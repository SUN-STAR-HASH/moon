"""
Training script for BEHAVIOR-1K solution.

Based on https://github.com/PhysicalIntelligence/openpi/blob/behavior/openpi/scripts/train.py with custom modifications.
"""

import subprocess # 4/8 
import dataclasses
import functools
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

# Configure JAX memory allocation to prevent OOM errors
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

# Configure OpenBLAS to prevent thread creation errors
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Import B1K-specific modules
from b1k.training import checkpoints as _checkpoints  # Use our custom checkpoints (not openpi's!)
from b1k.training import config as _config
from b1k.training import data_loader as _data_loader
from b1k.training import weight_loaders as _weight_loaders
from b1k.models.pi_behavior import PiBehavior
from b1k.models.pi_behavior_config import PiBehaviorConfig
from b1k.models.observation import Observation


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# wandb 초기화 함수.
# 새 실험이면 run id를 저장하고, resume이면 기존 run id를 다시 읽어 같은 실험으로 이어 붙인다.
#
# 여기서 중요한 점은 "checkpoint resume"과 "wandb run resume"를 같이 맞춰 준다는 것이다.
# 둘 중 하나만 이어지고 다른 하나가 새로 시작되면,
# 로그와 체크포인트 기록이 서로 어긋날 수 있다.
def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

# 4/8 추가
def log_gpu_mem(tag: str):
    """현재 GPU 메모리 사용량을 로그로 찍는다.

    주의:
    - 이 값은 '모델만의 메모리'가 아니라 현재 GPU 전체 사용량이다.
    - 그래도 단계별 증감을 보기에는 충분히 유용하다.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]

        used, total = [int(x.strip()) for x in out.split(",")]
        logging.info(f"[GPU MEM] {tag}: {used} MiB / {total} MiB")
    except Exception as e:
        # nvidia-smi가 없거나 실패해도 학습을 막지는 않게 한다.
        logging.info(f"[GPU MEM] {tag}: unavailable ({e})")


def block_and_log(tag: str, x=None):
    """JAX 연산을 실제로 끝까지 실행시킨 뒤 메모리 로그를 찍는다.

    JAX는 lazy / async 성격이 있어서,
    그냥 함수만 호출하면 실제 GPU 계산이 아직 안 끝났을 수 있다.
    그래서 block_until_ready를 걸어준 뒤 메모리를 보는 게 더 정확하다.
    """
    if x is not None:
        jax.block_until_ready(x)
    log_gpu_mem(tag)


# pretrained / partial checkpoint를 현재 모델 구조에 맞춰 불러오는 함수.
# 단순 load가 아니라,
# 1) nnx.Intermediate 같은 비-파라미터 필드를 비교 대상에서 빼고
# 2) 나머지 파라미터 트리의 shape / dtype 일치를 검사한 뒤
# 3) 실제로 주입 가능한 subset만 반환한다.
#
# 즉, 이 함수의 목적은 "불러오기"보다도
# "지금 모델 구조와 체크포인트가 안전하게 호환되는지 검증"하는 데 가깝다.
def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    
    # Filter out nnx.Intermediate fields from both sides (they're not params, excluded from checkpoints)
    # This allows loading old checkpoints that didn't have these fields
    # Intermediate 필드는 학습 파라미터라기보다
    # runtime cache / 통계 / 보조 상태에 가까워 체크포인트에 없을 수 있다.
    # 그래서 이 필드까지 엄격 비교하면,
    # 실제로는 문제없는 old checkpoint도 shape mismatch처럼 보일 수 있다.
    def filter_intermediate_fields(params_dict):
        flat = traverse_util.flatten_dict(params_dict)
        # List of field names that are nnx.Intermediate (excluded from checkpoints)
        intermediate_field_names = [
            'action_correlation_cholesky',  # Legacy full correlation matrix
            'L_spatial',                     # Separable spatial correlation
            'L_temporal',                    # Separable temporal correlation
            'cached_num_inpaint_actions',    # Conditional sampling cache
            'cached_input_action_dim',       # Conditional sampling cache
            'cached_Sigma_uo_Sigma_oo_inv',  # Conditional sampling cache
            'cached_L_cond_free',            # Conditional sampling cache
            'cached_Sigma_ou_Sigma_uu_inv',  # Conditional sampling cache
            'cached_L_cond_inp',             # Conditional sampling cache
        ]
        filtered = {k: v for k, v in flat.items() 
                   if not any(field in str(k) for field in intermediate_field_names)}
        return traverse_util.unflatten_dict(filtered)
    
    # Validate loaded params structure  
    params_shape_filtered = filter_intermediate_fields(params_shape)
    loaded_params_filtered = filter_intermediate_fields(loaded_params)
    at.check_pytree_equality(expected=params_shape_filtered, got=loaded_params_filtered, check_shapes=True, check_dtypes=True)
    
    # Remove jax.ShapeDtypeStruct and Intermediate fields from the loaded params
    def should_exclude(k, v):
        if isinstance(v, jax.ShapeDtypeStruct):
            return True
        # Exclude all intermediate fields
        intermediate_field_names = [
            'action_correlation_cholesky', 'L_spatial', 'L_temporal',
            'cached_num_inpaint_actions', 'cached_input_action_dim',
            'cached_Sigma_uo_Sigma_oo_inv', 'cached_L_cond_free',
            'cached_Sigma_ou_Sigma_uu_inv', 'cached_L_cond_inp',
        ]
        return any(field in str(k) for field in intermediate_field_names)
    
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() 
         if not should_exclude(k, v)}
    )


@at.typecheck
# 학습 시작에 필요한 TrainState를 만드는 함수.
# 순서는 대략
# 1) 모델 생성
# 2) 필요하면 correlation matrix 로드
# 3) partial pretrained weights 주입
# 4) freeze_filter 반영
# 5) optimizer / EMA state 초기화
# 이다.
#
# 즉, "모델 객체를 만든다"에서 끝나는 게 아니라
# "현재 실험 설정에 맞는 학습 가능 상태"까지 완성하는 단계다.
def init_train_state(
    config: _config.TrainConfig, 
    init_rng: at.KeyArrayLike, 
    mesh: jax.sharding.Mesh, 
    *, 
    resume: bool,
    norm_stats: dict | None = None
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)
        
        # Load correlation matrix into PiBehavior models BEFORE creating graphdef
        if isinstance(model, PiBehavior) and norm_stats is not None:
            # correlated noise용 correlation matrix는 graphdef를 고정하기 전에 미리 넣어 둔다.
            # 그래야 이후 state / sharding / checkpoint 구조가
            # 이미 correlation 정보가 반영된 형태로 정리된다.
            #
            # 나중에 넣으면 model state와 graphdef 타이밍이 어긋나 복잡해질 수 있다.
            model.load_correlation_matrix(norm_stats)
            logging.info("Loaded correlation matrix during model initialization")

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    
    # Log KV transform coefficients for PiBehavior models
    model = nnx.merge(train_state.model_def, train_state.params)
    if isinstance(model, PiBehavior) and hasattr(model, 'kv_transform') and model.kv_transform is not None:
        logging.info("KV Transform Coefficients (after loading):")
        logging.info("=" * 80)
        
        k_coeffs = model.kv_transform.k_coeffs.value
        v_coeffs = model.kv_transform.v_coeffs.value
        
        logging.info("K Coefficients (each layer attends to all VLM layers):")
        for i in range(k_coeffs.shape[0]):
            coeffs_str = ", ".join([f"{float(c):.2f}" for c in k_coeffs[i]])
            logging.info(f"  Layer {i:2d}: [{coeffs_str}]")
        
        logging.info("")
        logging.info("V Coefficients (each layer attends to all VLM layers):")
        for i in range(v_coeffs.shape[0]):
            coeffs_str = ", ".join([f"{float(c):.2f}" for c in v_coeffs[i]])
            logging.info(f"  Layer {i:2d}: [{coeffs_str}]")
        
        logging.info("=" * 80)

    return train_state, state_sharding


@at.typecheck
# jitted one-step 학습 함수.
# 현재 state에서 model을 복원하고,
# detailed loss를 계산한 뒤,
# gradient / optimizer update / EMA 갱신까지 한 번에 수행한다.
#
# 즉, 이 함수 하나가 "forward + backward + update" 전체를 담당한다.
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck  
    # 현재 학습은 model.compute_loss()가 아니라 compute_detailed_loss()를 사용한다.
    # 이유는 total_loss 외에도
    # fast_loss, fast_accuracy, subtask 관련 지표 등
    # 세부 항목을 함께 로깅하기 위해서다.
    def loss_fn(
        model: PiBehavior, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions
    ):
        losses_dict = model.compute_detailed_loss(rng, observation, actions, train=True, num_flow_samples=config.num_flow_samples)
        total_loss = jnp.mean(losses_dict["total_loss"])
        return total_loss, losses_dict

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, losses_dict), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)
    
    # Knowledge insulation gradient monitoring
    # knowledge insulation을 켰을 때 실제로 gradient가 어느 쪽으로 흐르는지 확인하기 위한 모니터링 코드.
    # 파라미터 이름 기준으로
    # - action expert 쪽
    # - 그 외 VLM 쪽
    # 을 나눠서 gradient norm을 따로 기록한다.
    #
    # 즉, 이 부분은 학습 로직 자체를 바꾸는 게 아니라
    # "설계한 gradient 차단이 실제로 먹는지"를 관찰하기 위한 계측 장치다.
    if config.model.use_knowledge_insulation:
        # Helper functions to identify parameter groups
        def is_action_expert_param(path_str):
            # Action expert parameters: 
            # - Second LLM expert (300M params, marked with _1 suffix)
            # - Action projections, time MLPs, kv_transform
            return any(x in path_str for x in [
                "_1",  # All second expert parameters
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
                "kv_transform"
            ])
        
        def is_vlm_param(path_str):
            # VLM parameters: everything else (first expert, img, FAST, task modules)
            return not is_action_expert_param(path_str)
        
        # Compute gradient norms for monitoring only (no scaling applied)
        def compute_group_norm(grads_state, predicate):
            """Compute norm for gradients matching predicate."""
            flat_grads = []
            for path, value in jax.tree_util.tree_flatten_with_path(grads_state.to_pure_dict())[0]:
                path_str = "/".join(str(k) for k in path)
                if predicate(path_str):
                    if hasattr(value, 'value'):
                        flat_grads.append(value.value if hasattr(value, 'value') else value)
                    else:
                        flat_grads.append(value)
            
            if flat_grads:
                return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads))
            return 0.0
        
        grad_norm_vlm = compute_group_norm(grads, is_vlm_param)
        grad_norm_action = compute_group_norm(grads, is_action_expert_param)
    else:
        grad_norm_vlm = None
        grad_norm_action = None

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    
    # Add gradient norm breakdown for knowledge insulation monitoring
    if grad_norm_vlm is not None:
        info["grad_norm_vlm"] = grad_norm_vlm
        info["grad_norm_action_expert"] = grad_norm_action

    # Add detailed loss components to info
    for key, value in losses_dict.items():
        if isinstance(value, (float, int)) or (hasattr(value, 'ndim') and value.ndim == 0):
            info[key] = value
        else:
            info[key] = jnp.mean(value)
    return new_state, info


# 전체 학습 실행 엔트리포인트.
# config 검증 -> seed 준비 -> sharding 구성 -> checkpoint/wandb 초기화
# -> data loader 준비 -> 첫 batch sanity check -> train state 초기화
# -> ptrain_step 반복 실행
# 순서로 돌아간다.
#
# 즉, train.py를 읽을 때는 main을 "실험 orchestration 스크립트"로 보면 된다.
def main(config: _config.TrainConfig):
    init_logging()

    # 4/8 추가 #############
    logging.info(f"Running on: {platform.node()}")

    # 프로그램 시작 직후 GPU 사용량
    log_gpu_mem("start")

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
    config.checkpoint_dir,
    keep_period=config.keep_period,
    overwrite=config.overwrite,
    resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # checkpoint manager / wandb 초기화 후 메모리 상태
    log_gpu_mem("after checkpoint/wandb init")
    ######################################################

    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Generate random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for JAX RNG: {seed}")
    
    rng = jax.random.key(seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_behavior_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    data_iter = iter(data_loader)
    # 첫 batch를 학습 전에 한 번 바로 뽑아 보는 이유는
    # data loader / transform / sharding 문제가 있으면 여기서 빨리 터뜨리기 위해서다.
    # 즉, 긴 초기화가 끝난 뒤 첫 step에서 죽는 것보다
    # 입력 파이프라인 문제를 가능한 앞단에서 확인하려는 sanity check다.
    batch = next(data_iter)

    # 4/8 추가 ############
    # 데이터 로더 생성 자체가 메모리를 얼마나 쓰는지 확인
    log_gpu_mem("before data_loader create")

    data_loader = _data_loader.create_behavior_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    log_gpu_mem("after data_loader create")

    data_iter = iter(data_loader)

    # 첫 batch를 실제로 뽑아보는 순간
    # 여기서 죽으면 dataset / transform / batching 단계 문제일 가능성이 크다.
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    log_gpu_mem("after first batch")
    #######################################
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    if config.wandb_enabled:
        # 첫 batch의 카메라 이미지를 wandb에 남겨
        # "입력 shape는 맞는데 내용이 뒤집히거나 채널 순서가 이상한" 문제를 빠르게 잡기 위한 시각적 sanity check다.
        # 다만 smoke/debug 환경에서는 오버헤드가 될 수 있어 wandb_enabled일 때만 수행한다.
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
        
    # Get norm_stats for correlation matrix loading
    data_config = data_loader.data_config()
    # 기본적으로 correlated noise / normalization 흐름을 위해 norm_stats를 요구한다.
    # 하지만 fake smoke는 실제 데이터셋 검증이 아니라 구조 smoke가 목적이므로,
    # repo_id == "fake"일 때만 이 요구사항을 예외적으로 완화한다.
    #
    # 즉, "norm_stats가 없어도 괜찮다"가 아니라
    # "fake smoke라는 특수한 디버그 경로에서만 통과시킨다"는 예외 처리다.
    if data_config.norm_stats is None:
        if data_config.repo_id == "fake":
            logging.warning(
                "norm_stats not found for fake laptop smoke config; "
                "skipping normalization stats requirement."
            )
        else:
            raise ValueError(
                "norm_stats not found. Run compute_norm_stats.py to generate normalization statistics."
            )
    norm_stats = data_config.norm_stats

    # 4/8 추가 #########
    # model init / weight restore / optimizer state 생성 직전 메모리 확인
    log_gpu_mem("before init_train_state")

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming, norm_stats=norm_stats
    )

    # 4/8 추가 #############
    # 기존 jax.block_until_ready(train_state) 대신 사용
    # JAX 계산이 실제 끝난 뒤 메모리를 확인하기 위해 block_and_log 사용
    block_and_log("after init_train_state", train_state)

    # model init / optimizer state / pretrained restore 직전
    log_gpu_mem("before init_train_state")

    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming, norm_stats=norm_stats
    )

    # 실제 계산이 끝난 뒤 메모리를 본다.
    block_and_log("after init_train_state", train_state)

    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    ################################

    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 저장은 save_interval마다 하되,
    # 시작 step 바로 직후의 중복 저장은 피하고,
    # 마지막 step은 interval과 무관하게 반드시 저장한다.
    #
    # 그래서 smoke처럼 짧은 실험에서도
    # 최소 한 번은 복구 가능한 checkpoint가 남도록 한다.
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        
        # 4/8 추가 #############
        # checkpoint restore 직후 메모리 확인
        # restore 과정에서 파라미터/옵티마 상태가 얼마나 메모리를 먹는지 확인 가능
        block_and_log("after restore_state", train_state)
        #######################

        # Reload correlation matrix after restore
        model = nnx.merge(train_state.model_def, train_state.params)
        model.load_correlation_matrix(norm_stats)
        logging.info("Reloaded correlation matrix after checkpoint restore")
        train_state = dataclasses.replace(train_state, model_def=nnx.graphdef(model))

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 4/8 추가 ###########
    # 첫 학습 step이 실제로 어디서 터지는지 보기 위한 디버그 실행
    # 위치는 ptrain_step 생성 직후, 본격 loop(start_step/pbar) 전에 둔다.
    logging.info("[TRACE] about to run first ptrain_step")
    log_gpu_mem("before first ptrain_step")

    try:
        # 첫 step을 명시적으로 한 번 실행
        train_state, info = ptrain_step(train_rng, train_state, batch)

        # loss까지 실제 계산 완료된 뒤 메모리 확인
        block_and_log("after first ptrain_step", info["loss"])

        logging.info(f"[TRACE] first step loss={float(jax.device_get(info['loss'])):.6f}")

    except Exception:
        logging.exception("[TRACE] failed during first ptrain_step")
        raise
    ##########################

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # Create a concise console log with main metrics
            main_metrics = {
                k: v for k, v in reduced_info.items()
                if "loss" in k or "accuracy" in k or k in [
                    "grad_norm", "param_norm", "grad_norm_vlm", "grad_norm_action_expert"
                ]
            }
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in main_metrics.items())
            pbar.write(f"Step {step}: {info_str}")

            if config.wandb_enabled:
                wandb.log(reduced_info, step=step)

            infos = []
        
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())