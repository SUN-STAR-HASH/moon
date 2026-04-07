# 한글 주석/설명 반영 내용

이번 압축본은 아래 원칙을 기준으로 정리했습니다.

- 웨이트 shape 유지
- 50-task 구조 유지
- 실제 학습/평가에서는 12개 task만 사용
- 5070은 OmniGibson/실행 담당
- A100은 정책 추론/학습 담당
- task embedding, stage 관련 입력 형식, flow matching 구조 유지

## 특히 많이 손본 파일
- `src/b1k/models/pi_behavior.py`
- `src/b1k/models/pi_behavior_config.py`
- `src/b1k/models/observation.py`
- `src/b1k/training/config.py`
- `src/b1k/training/data_loader.py`
- `src/b1k/transforms.py`
- `src/b1k/shared/eval_b1k_wrapper.py`
- `scripts/serve_b1k.py`
- `src/b1k/policies/checkpoint_switcher.py`

## 새로 추가한 파일
- `src/b1k/configs/task_subset.py`
- `src/b1k/configs/__init__.py`
- `task_checkpoint_mapping_light12.json`

## 참고
코드의 클래스 이름, 함수 이름, 라이브러리 이름은 실행을 위해 영어 그대로 두었습니다.
대신 주석과 설명 문장은 가능한 한 한글로 바꿨습니다.

#  2026/4/07 변경 요약

목표: "가장 기본적인 pi0 + task embedding + flow matching만" 남기기

## 핵심 변경
- `src/b1k/models/pi_behavior.py`
  - task embedding은 유지
  - stage-conditioned prefix 경로 제거
  - subtask/stage 보조 손실 제거
  - FAST 보조 손실 비활성 기본 경로로 정리
  - KV transform 실제 사용 경로 제거
  - `sample_actions()`는 actions만 반환하도록 단순화

- `src/b1k/models/observation.py`
  - 문법 오류 수정
  - 한글 주석 정리

- `src/b1k/policies/pi_behavior_policy.py`
  - 모델이 actions만 반환해도 동작하도록 수정

- `src/b1k/policies/b1k_policy.py`
  - 출력에서 stage 관련 필드 제거

- `src/b1k/shared/eval_b1k_wrapper.py`
  - stage 추적/투표/correction rule 제거
  - task_id만 넣는 단순 wrapper로 재작성

- `src/b1k/training/config.py`
  - 기본 모델 transform에서 stage 계산 transform 제거

- `src/b1k/policies/policy_config.py`
  - inference 시 `TaskIndexToTaskId`는 유지되도록 수정

## 현재 남아 있는 것
- 파일 내부 일부 stage 관련 모듈/파라미터는 체크포인트 호환 때문에 남아 있을 수 있음
- 하지만 기본 실행 경로에서는 사용하지 않도록 바꿔 둠
