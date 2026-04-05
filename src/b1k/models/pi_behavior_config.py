import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from b1k.models.observation import Observation

if TYPE_CHECKING:
    from b1k.models.pi_behavior import PiBehavior

# 태스크별 stage 개수. 데모 길이를 바탕으로 5~15 범위에서 정해 둔 값
# tuple로 두면 바뀌지 않고, import 시점에 불필요한 JAX 메모리 할당도 줄일 수 있다.
TASK_NUM_STAGES = (
    5, 6, 15, 15, 14, 12, 9, 15, 10, 15,  # Tasks 0-9
    7, 13, 10, 15, 15, 15, 15, 11, 13, 12,  # Tasks 10-19
    14, 15, 9, 15, 15, 15, 15, 15, 15, 15,  # Tasks 20-29
    11, 10, 10, 13, 5, 5, 14, 6, 8, 10,  # Tasks 30-39
    5, 15, 8, 15, 12, 11, 9, 14, 15, 15,  # Tasks 40-49
)

MAX_NUM_STAGES = 15  # Maximum stages per task
TOTAL_TASK_STAGE_EMBEDDINGS = sum(TASK_NUM_STAGES)  # 596 total embeddings

# Cumulative offsets for indexing into task_stage_embeddings (as tuple)
TASK_STAGE_OFFSETS = tuple([0] + [sum(TASK_NUM_STAGES[:i+1]) for i in range(len(TASK_NUM_STAGES) - 1)])


@dataclasses.dataclass(frozen=True)
class PiBehaviorConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 30
    max_token_len: int = 200  # Only used for compatibility, not for actual tokenization

    # Number of tasks in the behavior dataset
    # 기존 웨이트를 그대로 쓰기 위해 반드시 50을 유지한다.
    num_tasks: int = 50

    # Task embedding dimension - will match the paligemma width
    task_embedding_dim: int = None  # type: ignore

    # Maximum number of subtask states across all tasks
    max_num_subtask_states: int = MAX_NUM_STAGES

    # Path to task data JSON file for initialization
    task_data_path: str = "b1k/BEHAVIOR-1K/docs/challenge/task_data.json"

    # 행동 간 상관관계를 반영한 correlated noise를 사용할지 여부
    # 사용하려면 norm_stats 안에 상관행렬 정보가 미리 들어 있어야 한다.
    use_correlated_noise: bool = False

    # 상관행렬을 너무 강하게 믿지 않도록 섞는 비율
    # 실제 계산은 beta * S + (1-beta) * I 형태로 한다.
    # beta=1.0이면 상관행렬을 그대로 사용
    # beta=0.7이면 상관관계 70%, 독립 잡음 30%를 섞는다.
    # beta=0.0이면 완전히 독립 잡음만 사용
    correlation_beta: float = 0.5

    # FAST 보조 학습 관련 설정
    use_fast_auxiliary: bool = False  # 학습 시 FAST 보조 경로를 켤지 여부
    fast_loss_weight: float = 0.1  # 전체 손실에서 FAST 손실 비중

    # Action dimensions to encode with FAST (default: 0:6, 7:23 = 22 dims)
    # Format: "0:6,7:23" or list of tuples [(0, 6), (7, 23)]
    fast_encoded_dims: str | list[tuple[int, int]] = "0:6,7:23"

    # FAST tokenizer vocab size
    fast_vocab_size: int = 1024

    # Max FAST tokens to predict (truncate if exceeded)
    max_fast_tokens: int = 32

    # FAST tokenizer path (set during initialization, relative to assets_dir/asset_id)
    fast_tokenizer_path: str | None = None

    # VLM 층과 action expert 층 사이의 KV cache를 섞는 기능
    # 각 action expert 층이 VLM 여러 층의 정보를 섞어 보게 한다.
    use_kv_transform: bool = False

    # action expert의 gradient가 VLM 본체로 흘러가는 것을 막는 옵션
    # VLM은 FAST 쪽만, action expert는 flow matching 쪽만 보도록 분리할 때 쓴다.
    use_knowledge_insulation: bool = False

    # subtask/stage 예측 보조 손실의 가중치
    subtask_loss_weight: float = 0.0

    # 추론 중 inpainting 제약을 언제 풀지 정하는 기준
    # t가 이 값보다 작아지면 마지막 단계에서는 모델이 더 자유롭게 행동하게 둔다.
    time_threshold_inpaint: float = 0.3

    # vision backbone을 고정할지 여부
    freeze_vision_backbone: bool = True

    def __post_init__(self):
        if self.task_embedding_dim is None:
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            object.__setattr__(self, "task_embedding_dim", paligemma_config.width)

    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """PI_BEHAVIOR 모델 설정 파일.

주의:
- 실제로는 12개 태스크만 써도, 모델 파라미터 모양은 원래 50개 태스크 기준을 유지해야 한다.
- 이 파일에서 num_tasks, action_horizon 같은 값을 함부로 줄이면 기존 웨이트를 그대로 불러올 수 없게 된다.
"""
        if isinstance(self.fast_encoded_dims, str):
            ranges = []
            for range_str in self.fast_encoded_dims.split(','):
                start, end = map(int, range_str.strip().split(':'))
                ranges.append((start, end))
            return ranges
        return self.fast_encoded_dims

    def get_total_fast_dims(self) -> int:
        """Get total number of dimensions encoded by FAST."""
        return sum(end - start for start, end in self.get_fast_dim_ranges())

    @property
    @override
    def model_type(self):
        return "pi_behavior"

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiBehavior":
        from b1k.models.pi_behavior import PiBehavior
        return PiBehavior(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            obs_kwargs = {
                "images": {
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                "image_masks": {
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                "tokenized_prompt": jax.ShapeDtypeStruct([batch_size, 2], jnp.int32),
                "tokenized_prompt_mask": jax.ShapeDtypeStruct([batch_size, 2], bool),
            }

            if self.use_fast_auxiliary:
                obs_kwargs["fast_tokens"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], jnp.int32
                )
                obs_kwargs["fast_token_mask"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], bool
                )

            observation_spec = Observation(**obs_kwargs)
            action_spec = jax.ShapeDtypeStruct(
                [batch_size, self.action_horizon, self.action_dim], jnp.float32
            )
            return observation_spec, action_spec