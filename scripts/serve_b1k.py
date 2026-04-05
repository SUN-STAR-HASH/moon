"""정책 서버 실행 스크립트.

이 파일은 A100에서 정책 모델을 띄우고,
5070에서 도는 OmniGibson 시뮬레이터가 웹소켓으로 행동을 받아갈 수 있게 한다.
이번 수정에서는 실행을 가볍게 만들기 위해 기본값을 줄였다.
"""

import dataclasses
import enum
import logging
import os
import pathlib
import socket

import numpy as np
import tyro

# JAX를 가져오기 전에 GPU 메모리 사용 기본값을 정한다.
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')  # GPU 메모리를 절반 정도만 기본으로 사용
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')  # 플랫폼 기본 할당기 사용

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy

# 우리 프로젝트 전용 모듈들
from b1k.policies import policy_config as _policy_config  # 프로젝트용 policy_config 사용
from b1k.policies.checkpoint_switcher import CheckpointSwitcher
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config


class EnvMode(enum.Enum):
    # 현재는 거의 쓰지 않지만 기존 인터페이스 호환을 위해 남겨둔 값
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default prompt.
    default_prompt: str | None = None
    
    # For PI_BEHAVIOR models: task ID (0-49) instead of text prompt
    task_id: int | None = None

    # Dataset root, used to retrieve the prompt of the task if taskname is not None.
    dataset_root: str | None = "/scr/behavior/2025-challenge-demos"
    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # B1K Wrapper execution parameters
    actions_to_execute: int = 16
    actions_to_keep: int = 4
    execute_in_n_steps: int = 12
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 8
    apply_eval_tricks: bool = True  # 평가용 보정 규칙과 gripper 변화 검사 사용 여부
    
    # Multi-checkpoint support for PI_BEHAVIOR models (optional)
    task_checkpoint_mapping: str | None = None  # Path to task-checkpoint mapping JSON file


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    sample_kwargs = {"num_steps": args.num_steps}
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), 
        args.policy.dir, 
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs
    )


def main(args: Args) -> None:
    # B1K에서는 텍스트 대신 task embedding을 쓰는 PI_BEHAVIOR 계열만 사용한다.
    config = _config.get_config(args.policy.config)
    
    # task_id를 직접 줄 수도 있고, observation에서 읽어 올 수도 있다.
    if args.task_id is not None:
        logging.info(f"Using PI_BEHAVIOR model with task_id: {args.task_id}")
        task_id = args.task_id
    else:
        logging.info(f"Using PI_BEHAVIOR model - task_id will be extracted from observations")
        task_id = None
    
    # 이 모델은 실제로 텍스트 프롬프트를 쓰지 않지만, 인터페이스 호환을 위해 자리만 둔다.
    prompt = "PI_BEHAVIOR model (task-conditioned)"
    logging.info(f"Using prompt: {prompt}")

    # 먼저 기본 정책 하나를 로드한다.
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # task별로 체크포인트를 바꾸고 싶으면 여기서 스위처를 만든다.
    checkpoint_switcher = None
    if args.task_checkpoint_mapping:
        logging.info(f"Multi-checkpoint mode enabled: {args.task_checkpoint_mapping}")
        
        sample_kwargs = {"num_steps": args.num_steps}
        
        try:
            checkpoint_switcher = CheckpointSwitcher(
                config_path=args.task_checkpoint_mapping,
                training_config=config,
                sample_kwargs=sample_kwargs
            )
            logging.info("Checkpoint switcher initialized - will switch checkpoints based on task_id")
        except Exception as e:
            logging.error(f"Failed to initialize checkpoint switcher: {e}")
            raise
    else:
        logging.info("Single checkpoint mode - using one checkpoint for all tasks")

    # 디버깅용으로 정책의 입출력을 기록하고 싶으면 recorder를 감싼다.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # wrapper 설정: 한 번 예측한 행동 중 몇 step을 실행할지 등을 정한다.
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=args.apply_eval_tricks,
    )
    
    logging.info(f"Wrapper config: execute={wrapper_config.actions_to_execute}, keep={wrapper_config.actions_to_keep}, steps={wrapper_config.execute_in_n_steps}, num_steps={wrapper_config.num_steps}")
    
    if wrapper_config.apply_eval_tricks:
        logging.info("Eval tricks ENABLED - correction rules and gripper variation checks active")
    else:
        logging.info("Eval tricks DISABLED (default behavior)")

    # B1K 전용 wrapper를 감싸서 rolling inpainting, stage 관리 등을 사용한다.
    policy = B1KPolicyWrapper(
        policy, 
        text_prompt=prompt,  # Not used by PI_BEHAVIOR, kept for compatibility
        task_id=task_id,
        config=wrapper_config,
        checkpoint_switcher=checkpoint_switcher
    )
    
    if checkpoint_switcher:
        logging.info("Multi-checkpoint mode: checkpoints will switch based on task_id from observations")
    else:
        logging.info("Rolling inpainting enabled: will use initial_actions from input batch when provided")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
