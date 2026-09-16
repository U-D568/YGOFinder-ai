# Continuous model evaluation

이미지 수집은 학습 트리거가 아닙니다. 6시간마다 운영 체크포인트를 정답이 있는
덱 스크린샷으로 평가하고, 최소 100장의 micro F1이 0.95 미만일 때만 student를
재학습합니다. 검출 누락, 오인식, 과검출, 중복 카드 수량을 모두 계산합니다.
정답 없는 사용자 업로드나 모델 자신의 예측은 학습 정답으로 사용하지 않습니다.

## 준비

Python 3.11 환경에 `pip install -r requirements-training.txt`를 실행합니다.
운영 `.pt`, 해당 벡터 컬렉션을 만든 teacher `.h5`, 학습 CSV와 카드 이미지를
준비합니다. CSV는 기존 `id,type` 필드를 포함하며 이미지는 `<id>.jpg`입니다.
teacher와 Chroma 컬렉션은 한 평가 주기 동안 고정하고, 운영과 동일한 Chroma 환경
설정(`db/chroma_db.py`의 `.env`/환경 변수)을 평가 runner에도 제공합니다.

정답 manifest는 JSON 배열입니다. 이미지 경로는 manifest 기준이며, 같은 카드가
여러 번 보이면 ID도 그 횟수만큼 기입합니다. 중복 이미지나 빈 정답은 거부합니다.

```json
[
  {"image": "screenshots/deck-001.png", "card_ids": [89631139, 89631139, 46986414]}
]
```

실제 평가에서는 100장 이상을 준비합니다. 평가 스크린샷은 학습에 넣지 않으며,
반복 튜닝에 쓰지 않는 별도 최종 테스트셋으로 배포 전 확인합니다. 평가 manifest를
변경하면 기존 점수와 직접 비교하지 않습니다. 한 주기 안에서는 이미지 내용과
정답의 SHA-256이 일치해야 후보를 평가할 수 있습니다.

## 실행

저장소 루트에서 실행합니다. 경로는 예시이므로 실제 경로로 바꿉니다.

```bash
python -m continuous.run \
  --manifest /data/evaluation/manifest.json \
  --checkpoint /models/production.pt \
  --teacher /models/embedding.h5 \
  --train-csv /data/train.csv \
  --image-dir /data/card_images_small \
  --state-dir /data/continuous-state
```

`--threshold`, `--min-samples`, `--cooldown-hours`, `--epochs`, `--batch-size`,
`--min-improvement`, `--timeout`으로 정책을 조정할 수 있습니다. 기본 cooldown은
24시간이며 실패한 학습도 적용됩니다. 같은 state 디렉터리를 사용하는 프로세스는
파일 잠금으로 중복 실행을 막습니다. 모든 scheduler는 동일한 영속 state 경로를
사용해야 합니다. 입력/추론 오류는 저성능 점수로 바꾸지 않고 실패 처리합니다.

## 자동 실행

기본 브랜치에 workflow를 반영하고, 신뢰할 수 있는 self-hosted runner에
`yugioh-training` 라벨을 붙입니다. runner 외부 영속 경로에 데이터·모델·state와
Python 환경을 둡니다. repository variables에 다음을 설정합니다.

- `TRAINING_PYTHON`: 학습 가상환경 Python 실행 파일의 절대 경로
- `EVAL_MANIFEST`, `MODEL_PATH`, `TEACHER_PATH`: manifest, 운영 student, 고정 teacher 경로
- `TRAIN_CSV`, `CARD_IMAGE_DIR`, `EVAL_STATE_DIR`: 학습 데이터와 영속 state 경로
- `CONTINUOUS_EVALUATION_ENABLED=true`: 준비가 끝난 뒤 활성화

GitHub schedule은 기본 브랜치에서 실행되며 fork에서는 Actions를 활성화해야 합니다.
workflow_dispatch도 동일한 평가 정책을 사용합니다. scheduler는 GPU 학습을 수행하므로
API inference 프로세스와 분리된 runner를 사용합니다.

## 결과와 배포

state 하위 실행별 디렉터리에 `baseline.json`, `candidate.json`, `result.json` 및
`candidate.pt`가 저장됩니다. baseline/candidate는 학습 전에 존재하던 동일 평가셋과
동일 벡터 컬렉션, `server.py`의 실제 예측 경로로 측정합니다.

후보 F1이 0.95 이상이고 기존보다 0.01 이상 개선되며 precision/recall이 나빠지지
않아야 `candidate_ready`가 됩니다. 그 외에는 `candidate_rejected`입니다.
운영 모델을 덮어쓰거나 자동 배포하지 않습니다. 준비된 후보는 최종 테스트셋 검증 후
`MODEL_PATH`를 새 파일로 바꾸고 API 프로세스를 재시작하여 반영합니다.

student는 운영 가중치 전체를 이어받고 `embed_gain=1`로 식별 임베딩을 학습합니다.
teacher를 고정하여 기존 256차원 검색 공간을 유지합니다. Transformer/teacher 교체는
student 재증류와 전체 카드 벡터 재생성이 필요한 별도 변경입니다.

`main:app`은 `server:app`의 호환 alias입니다. 중복 서버 구현과 사용되지 않는
distillation 초기화/optimizer/검증 주석은 제거했습니다. 수동 teacher 실험용
`embedding/train.py`는 자동 pipeline에 포함되지 않습니다.

정책/실패 처리 검증: `python -m unittest discover -s tests -v`.
학습 의존성을 설치한 환경에서 gradient, 빈 foreground, 병렬 이미지 로딩 정렬을
검증하려면 `python -m unittest discover -s tests/ml -v`를 실행합니다.
