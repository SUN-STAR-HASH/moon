"""Observation 자료형과 전처리 함수.

기존 openpi Observation에 FAST 관련 필드를 추가한 버전이다.
이미지는 [-1, 1] 범위로 맞추고, 필요하면 학습 시 증강도 수행한다.
"""

from collections.abc import Sequence
from typing import Generic, TypeVar
import dataclasses

import augmax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.shared import image_tools
from openpi.shared import array_typing as at

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
IMAGE_RESOLUTION = (224, 224)


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """모델이 읽는 observation 묶음.

    이미지, 상태(state), task/stage 프롬프트 토큰, 그리고 필요하면 FAST 토큰까지 함께 담는다.
    """
    
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    
    fast_tokens: at.Int[ArrayT, "*b t"] | None = None
    fast_token_mask: at.Bool[ArrayT, "*b t"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """Convert dict to Observation."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        
        # uint8 이미지를 모델이 바로 쓰기 쉬운 float32 [-1, 1] 범위로 바꾼다.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            fast_tokens=data.get("fast_tokens"),
            fast_token_mask=data.get("fast_token_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert Observation to dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """이미지 크기를 맞추고, 학습 시에는 증강을 적용하며, FAST 관련 필드는 그대로 보존한다."""
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # augmax 증강 함수는 [0, 1] 범위를 기대하므로 잠시 바꿔 준다.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 증강이 끝나면 다시 [-1, 1] 범위로 돌린다.
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 이미지가 실제로 존재하는지 나타내는 mask를 만든다.
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        fast_tokens=getattr(observation, 'fast_tokens', None),
        fast_token_mask=getattr(observation, 'fast_token_mask', None),
    )
