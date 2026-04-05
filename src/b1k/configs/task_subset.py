"""12개 태스크만 사용하기 위한 설정 파일.

중요:
- 모델은 50개 태스크 기준으로 학습된 웨이트를 그대로 사용한다.
- 따라서 데이터셋에서 0~11처럼 작은 번호로 태스크를 관리하더라도,
  모델에 넣기 전에는 반드시 원래의 전역 task id(0~49)로 되돌려야 한다.
"""

# 실제로 사용할 12개 태스크의 원래 전역 id 목록
SELECTED_TASKS = [0, 1, 3, 5, 6, 7, 11, 18, 19, 20, 21, 22]

# 12개 subset 안에서의 로컬 번호(0~11) -> 원래 전역 task id(0~49)
LOCAL_TO_GLOBAL = {i: t for i, t in enumerate(SELECTED_TASKS)}

# 원래 전역 task id(0~49) -> 12개 subset 안에서의 로컬 번호(0~11)
GLOBAL_TO_LOCAL = {t: i for i, t in enumerate(SELECTED_TASKS)}

def map_local_to_global(local_task_id: int) -> int:
    """subset 로컬 번호를 모델이 아는 원래 task id로 바꾼다."""
    return LOCAL_TO_GLOBAL[local_task_id]
