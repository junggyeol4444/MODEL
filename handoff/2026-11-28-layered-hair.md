# 2026-11-28 제작 인계

최신 실제 모델 후보: `xuanyuan_layered_hair_candidate.blend` / `.glb`.
- 실행: https://github.com/junggyeol4444/MODEL/actions/runs/46111807103 (success)
- 아티팩트: https://github.com/junggyeol4444/MODEL/actions/runs/46111807103/artifacts/15249990695 (11,621,525 bytes, 다운로드/ZIP/GLB 검증)
- 제작 코드: `scripts/refine_layered_hair.py`, 실행 커밋 `0ecefd6017efdf395085f01f31f2fb34e29ec2b3`.
- 직전 독립 장발 모델: run 46111630780 / artifact 15249153983. 이전 결과 보존.

검증: GLB에 몸통 1개, 장발 밑층 1개, 독립 머리카락 곡면 13개가 존재한다. 원본 측면 수정 모델 대비 몸통 정점/면/UV 배열 동일. 로그상 추가 곡면 8,736 정점, 비다양체 경계 0. 4방향 렌더링에서 장발 분리와 교차 흐름 확인.

미완료: 정수리의 각진 연결부, 옆면 연결·충돌, 기존 몸통 표면에 그려진 머리카락 잔존, 원본과 비교한 머리카락 세부 윤곽/색상 및 자연스러운 층 구조, 얼굴·눈·손, 의상, 리토폴로지, UV, 리깅/표정/물리, VRM/VRChat 구동. BOOTH급 완성품 아님. 다음 실행은 최신 성공 결과를 조회하고 이 후보보다 최신 결과가 없으면 위 아티팩트를 입력으로 실제 제작을 계속한다.
