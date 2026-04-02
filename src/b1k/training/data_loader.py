"""Data loader utilities for BEHAVIOR-1K training.
This file mirrors openpi.training.data_loader, but swaps the dataset creation
path to use OmniGibson's BehaviorLeRobotDataset when available.

The goal is:
1. Load BEHAVIOR-1K from a local root if provided.
2. Reuse OpenPI's transform / batching / sharding pipeline.
3. Avoid requiring videos for smoke tests (download_videos=False).
"""

import importlib
import logging
import os
import time
from typing import Literal

import numpy as np
from torch.utils.data import Dataset
import jax
import torch
import lerobot.datasets.lerobot_dataset as lerobot_dataset

LeRobotDatasetMetadata = getattr(lerobot_dataset, "LeRobotDatasetMetadata", None)

import openpi.models.model as _model
import openpi.training.data_loader as _openpi_data_loader
from b1k.training import config as _config
from b1k.models.observation import Observation

logger = logging.getLogger(__name__)


class DataLoaderImpl(_openpi_data_loader.DataLoader):
    """Custom DataLoader using our Observation with fast_tokens.

    OpenPI 기본 DataLoaderImpl은 openpi.models.model.Observation을 반환할 수 있는데,
    현재 train_step은 b1k.models.observation.Observation을 기대하므로
    여기서 Observation.from_dict(batch)로 변환해서 넘긴다.
    """
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: _openpi_data_loader.TorchDataLoader | _openpi_data_loader.RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def _expand_root(root: str | None) -> str | None:
    """Expand '~' and return an absolute path."""
    if root is None:
        return None
    return os.path.abspath(os.path.expanduser(root))


def _get_dataset_fps(repo_id: str, default_fps: float = 30.0) -> float:
    """Try to read dataset fps from LeRobot metadata, otherwise fall back."""
    if LeRobotDatasetMetadata is None:
        logger.warning(
            "LeRobotDatasetMetadata is not available in this lerobot install. "
            "Falling back to %.1f fps.",
            default_fps,
        )
        return default_fps
    try:
        dataset_meta = LeRobotDatasetMetadata(repo_id)
        fps = getattr(dataset_meta, "fps", None)
        if fps is not None:
            return float(fps)
    except Exception as exc:
        logger.warning(
            "Could not read dataset fps from LeRobot metadata for %s. "
            "Falling back to %.1f fps. (%s)",
            repo_id,
            default_fps,
            exc,
        )
    return default_fps


def _build_delta_timestamps(
    action_sequence_keys: tuple[str, ...] | list[str],
    action_horizon: int,
    fps: float,
) -> dict[str, list[float]]:
    """Build delta timestamps expected by LeRobot / BehaviorLeRobotDataset."""
    return {
        key: [t / fps for t in range(action_horizon)]
        for key in action_sequence_keys
    }


def _get_behavior_lerobot_dataset_cls():
    """Import BehaviorLeRobotDataset from any known OmniGibson path.

    OmniGibson 버전에 따라 import path가 달라질 수 있으므로 여러 후보를 시도한다.
    """
    candidate_modules = [
        "omnigibson.learning.datas.lerobot_dataset",
        "omnigibson.learning.data.lerobot_dataset",
        "omnigibson.learning.utils.lerobot_dataset",
    ]
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        dataset_cls = getattr(module, "BehaviorLeRobotDataset", None)
        if dataset_cls is not None:
            logger.info("Using BehaviorLeRobotDataset from %s", module_name)
            return dataset_cls
    return None


def _instantiate_behavior_dataset(
    dataset_cls,
    repo_id: str,
    root: str | None,
    delta_timestamps: dict[str, list[float]],
    episodes_index: list[int] | None,
    seed: int | None,
):
    """Instantiate BehaviorLeRobotDataset with a few signature fallbacks.

    OmniGibson / BEHAVIOR 버전에 따라 생성자 인자가 조금씩 다를 수 있어서
    여러 kwargs 조합을 순차적으로 시도한다.
    """
    common_kwargs = {
        "repo_id": repo_id,
        "delta_timestamps": delta_timestamps,
    }

    candidate_kwarg_sets: list[dict[str, object]] = [
        {
            "root": root,
            "episodes": episodes_index,
            "download_videos": False,
            "local_only": False,
            "chunk_streaming_using_keyframe": False,
            "seed": seed,
        },
        {
            "root": root,
            "episodes": episodes_index,
            "download_videos": False,
            "local_only": False,
            "chunk_streaming_using_keyframe": False,
        },
        {
            "root": root,
            "episodes": episodes_index,
            "download_videos": False,
            "local_only": False,
        },
        {
            "root": root,
            "episodes": episodes_index,
            "download_videos": False,
        },
        {
            "root": root,
            "episodes": episodes_index,
        },
        {
            "root": root,
        },
        {},
    ]

    errors: list[str] = []
    for extra_kwargs in candidate_kwarg_sets:
        kwargs = {
            **common_kwargs,
            **{k: v for k, v in extra_kwargs.items() if v is not None},
        }
        try:
            dataset = dataset_cls(**kwargs)
            logger.info(
                "Created BehaviorLeRobotDataset with kwargs: %s",
                sorted(kwargs.keys()),
            )
            return dataset
        except TypeError as exc:
            errors.append(f"{sorted(kwargs.keys())}: {exc}")

    # 랩탑 / fake smoke 목적에서는 HF remote fallback이 문제를 일으켜서 의도적으로 비활성화    
    raise TypeError(
        "Could not instantiate BehaviorLeRobotDataset with any known signature.\n"
        + "\n".join(errors)
    )


def _instantiate_fallback_lerobot_dataset(
    repo_id: str,
    root: str | None,
    delta_timestamps: dict[str, list[float]],
):
    raise RuntimeError(
        "OmniGibson is not installed in this laptop environment, "
        "and LeRobot fallback is disabled because it triggers remote HF downloads."
    )


def create_behavior_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    seed: int | None = None,
) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.

    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of
    BEHAVIOR-1K data when real data is available.

    For laptop smoke tests (`repo_id == "fake"`), returns a lightweight
    dummy dataset that mimics the raw BEHAVIOR field structure expected by
    the existing repack/data/model transform pipeline.
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    logging.info(f"Using random seed for dataset: {seed}")

    # ------------------------------------------------------------------
    # Laptop / fake smoke path: bypass OmniGibson entirely
    # ------------------------------------------------------------------
    if data_config.repo_id == "fake":
        logging.warning(
            "Using laptop fake dataset for smoke test "
            "(bypassing OmniGibson / BehaviorLeRobotDataset)."
        )

        class _FakeMeta:
            def __init__(self, episodes):
                self.episodes = episodes

        class _LaptopFakeBehaviorDataset:
            """Smoke test용 경량 fake dataset.

            기존 transform 파이프라인이 기대하는 raw schema:
            - 3개 RGB 카메라
            - 256차원 proprio
            - 23차원 action (이후 transform에서 32차원으로 패딩됨)
            - task_index / timestamp / episode_index / index
            """
            def __init__(self, num_samples: int = 16):
                self.num_samples = num_samples
                self.episodes = np.array([0, 1], dtype=np.int32)
                self.meta = _FakeMeta(self.episodes)
                half = num_samples // 2
                self.episode_data_index = {
                    "from": np.array([0, half], dtype=np.int64),
                    "to": np.array([half, num_samples], dtype=np.int64),
                }

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx: int):
                sample_rng = np.random.default_rng(seed + idx)

                # 두 개의 에피소드가 있는 것처럼 index를 나눠서 구성
                half = self.num_samples // 2
                if idx < half:
                    episode_index = 0
                    timestep_in_episode = idx
                else:
                    episode_index = 1
                    timestep_in_episode = idx - half

                # proprio는 256차원으로 만들고,
                # 실제 gripper 위치처럼 보이도록 일부 인덱스에 작은 상수값을 넣음
                proprio = np.zeros(256, dtype=np.float32)
                proprio += sample_rng.normal(0.0, 0.005, size=(256,)).astype(np.float32)
                proprio[193:195] = 0.02
                proprio[232:234] = 0.02

                # 3개 카메라 입력
                head_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)
                left_wrist_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)
                right_wrist_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)

                # 원본 action은 23차원으로 만들고,
                # 이후 transform에서 32차원으로 pad되도록 둠
                action = sample_rng.normal(
                    loc=0.0,
                    scale=0.05,
                    size=(action_horizon, 23),
                ).astype(np.float32)

                return {
                    "observation.images.rgb.head": head_rgb,
                    "observation.images.rgb.left_wrist": left_wrist_rgb,
                    "observation.images.rgb.right_wrist": right_wrist_rgb,
                    "observation.state": proprio,
                    "action": action,
                    "task_index": np.int32(idx % 50),
                    "timestamp": np.float32(timestep_in_episode),
                    "episode_index": np.int32(episode_index),
                    "index": np.int64(idx),
                }

        return _LaptopFakeBehaviorDataset()

    # ------------------------------------------------------------------
    # Real dataset path
    # ------------------------------------------------------------------
    # 참고: 이 부분은 현재 네 버전대로 direct import를 유지한 형태
    # 어제 fake smoke 기준으로는 문제 없었지만,
    # 나중엔 _get_behavior_lerobot_dataset_cls()를 써서 더 안전하게 바꿀 수 있음
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset

    tasks = [
        "picking_up_trash",
        "putting_away_Halloween_decorations",
        "cleaning_up_plates_and_food",
        "setting_mousetraps",
        "hiding_Easter_eggs",
        "set_up_a_coffee_station_in_your_kitchen",
        "putting_dishes_away_after_cleaning",
        "preparing_lunch_box",
        "loading_the_car",
        "carrying_in_groceries",
        "turning_on_radio",
        "picking_up_toys",
        "can_meat",
        "rearranging_kitchen_furniture",
        "putting_up_Christmas_decorations_inside",
        "bringing_in_wood",
        "moving_boxes_to_storage",
        "bringing_water",
        "tidying_bedroom",
        "outfit_a_basic_toolbox",
        "sorting_vegetables",
        "collecting_childrens_toys",
        "putting_shoes_on_rack",
        "boxing_books_up_for_storage",
        "storing_food",
        "clearing_food_from_table_into_fridge",
        "assembling_gift_baskets",
        "getting_organized_for_work",
        "clean_up_your_desk",
        "setting_the_fire",
        "clean_boxing_gloves",
        "wash_a_baseball_cap",
        "wash_dog_toys",
        "hanging_pictures",
        "attach_a_camera_to_a_tripod",
        "clean_a_patio",
        "clean_a_trumpet",
        "spraying_for_bugs",
        "spraying_fruit_trees",
        "make_microwave_popcorn",
        "cook_cabbage",
        "make_pizza",
        "chop_an_onion",
        "slicing_vegetables",
        "chopping_wood",
        "canning_food",
        "cook_hot_dogs",
        "cook_bacon",
        "freeze_pies",
    ]

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=False,
        shuffle=True,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = _openpi_data_loader.TransformedDataset(
            dataset,
            [_openpi_data_loader._transforms.PromptFromLeRobotTask(dataset.meta.tasks)],
        )

    return dataset


def create_behavior_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a BEHAVIOR-1K torch-backed data loader."""
    dataset = create_behavior_dataset(
        data_config,
        action_horizon=action_horizon,
        seed=seed,
    )
    dataset = _openpi_data_loader.transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
    )

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logger.info("local_batch_size: %s", local_batch_size)

    data_loader = _openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    # 핵심: OpenPI 기본 DataLoaderImpl이 아니라
    # B1K Observation을 반환하는 커스텀 래퍼를 사용
    return DataLoaderImpl(data_config, data_loader)


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a BEHAVIOR-1K data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info("data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        return _openpi_data_loader.create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    return create_behavior_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=0 if config.seed is None else config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )