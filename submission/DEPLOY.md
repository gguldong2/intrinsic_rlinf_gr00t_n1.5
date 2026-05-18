# AIC 제출 패키지 — GR00T Policy (step 19000)

평가 점수 (단일 평가 기준):
- 1차: 73.24 (sweep, distrobox)
- 2차 (self-contained 이미지 검증): 67.02
- 분산 범위 ±8점, 평균 ~70점대

## 1. 패키지 내용

| 파일 | 용도 |
|------|------|
| `aic-policy-gr00t-step19000.tar.gz` (22GB) | self-contained docker image |
| `README_DEPLOY.md` (본 문서) | 운영자 가이드 |
| `SWEEP_REPORT.md` | 모든 체크포인트 평가 결과 (별첨) |

## 2. 사전 준비물 (팀장)

- Docker 설치된 머신 (디스크 가용 ~50GB 이상)
- AWS CLI v2 (`sudo apt install awscli`)
- AIC 운영자가 보낸 onboarding 이메일의 다음 항목:
  - AWS Access Key ID
  - AWS Secret Access Key
  - 팀의 ECR Repository URI (`973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>`)
  - 제출 portal 로그인 정보

## 3. 배포 절차

### 단계 1 — 이미지 로드

```bash
gunzip -c aic-policy-gr00t-step19000.tar.gz | docker load
# 또는: docker load < aic-policy-gr00t-step19000.tar.gz (gzip 자동 감지)

# 확인
docker images aic-policy-gr00t
# → aic-policy-gr00t   submit-step19000   ...   ~40GB
```

### 단계 2 — 로컬 검증 (선택, 권장)

AIC `docs/submission.md`의 "Verify Locally" 지침에 따라:

```bash
# aic 레포가 필요 — git clone https://github.com/Phy-Lab-aic/aic (또는 본인 fork)
cd ~/aic  # aic 레포 위치

# docker-compose.yaml의 model 서비스를 우리 이미지로 변경
# 또는 직접 docker run으로 검증 (아래 명령)

docker network create aic-verify || true

docker run -d --rm --name aic-eval-verify --network aic-verify --gpus all \
  -e AIC_EVAL_PASSWD=CHANGE_IN_PROD -e AIC_MODEL_PASSWD=CHANGE_IN_PROD \
  ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  gazebo_gui:=false launch_rviz:=false ground_truth:=false \
  start_aic_engine:=true shutdown_on_aic_engine_exit:=false \
  model_discovery_timeout_seconds:=600

docker run --rm --name aic-policy-self --network aic-verify --gpus all \
  -e AIC_ROUTER_ADDR=aic-eval-verify:7447 \
  -e AIC_MODEL_PASSWD=CHANGE_IN_PROD \
  aic-policy-gr00t:submit-step19000

# eval 로그에서 Total Score 확인:
docker logs aic-eval-verify 2>&1 | grep -E "Trial.*Score|Total Score"

# 정리
docker rm -f aic-eval-verify aic-policy-self 2>/dev/null
docker network rm aic-verify 2>/dev/null
```

### 단계 3 — ECR 푸시

```bash
# 1) AWS credentials 설정 (한 번만)
aws configure --profile <team_name>
#   Access Key ID:      (onboarding 이메일에서)
#   Secret Access Key:  (onboarding 이메일에서)
#   Default region:     us-east-1
#   Output format:      json

# 2) ECR 로그인
export AWS_PROFILE=<team_name>
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
    973918476471.dkr.ecr.us-east-1.amazonaws.com

# 3) 이미지 태그 (이메일에 받은 정확한 URI 사용)
docker tag aic-policy-gr00t:submit-step19000 \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1

# 4) 푸시 (첫 푸시는 15~30분 소요, 약 22GB)
docker push \
  973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
```

> ⚠️ **태그 immutable 주의**: 같은 태그(v1)로 다시 푸시 불가. 재제출 필요 시 v2, v3 등 사용.

### 단계 4 — 제출 portal 등록

1. 제출 portal 로그인 (이메일에 받은 credentials)
2. `AI for Industry Challenge` → `Submit` → `Qualification` phase 선택
3. `OCI Image` 필드에 위 URI 붙여넣기:
   ```
   973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1
   ```
4. `Submit` 클릭

## 4. 모델 동작 요약

- **베이스 모델**: NVIDIA GR00T-N1.5-3B (Vision-Language-Action)
- **Fine-tune**: SFT, `liqejdy/aic_cable_insert_sft_gr00t_n15_pretrained_v3_bs96_mb24_step5596` step 19000
- **학습 데이터**: `Phy-lab/pretrained_dataset_v3` (12 task, 2152 episodes)
- **정책 구조**:
  - ROS2 Lifecycle node `aic_model` (Python 3.12, pixi env)
  - GR00T inference worker는 별도 Python 3.11 venv subprocess + ZMQ 통신
  - `Task` 객체 → 학습 데이터 instruction 형식으로 변환:
    `"Insert the SFP-to-SC cable's <plug_type>_tip into <port_name> on <target_module_name>."`
- **액션 출력**: 16-step joint chunk @ 10 Hz, stiffness/damping 컨트롤러 호환

## 5. 환경변수 (모두 이미지 기본값으로 박힘, 변경 불필요)

| 변수 | 기본값 |
|------|--------|
| `GROOT_MODEL_PATH` | `/opt/models/gr00t-n1.5-3b` |
| `GROOT_CHECKPOINT_PATH` | `/opt/checkpoints/full_weights.pt` |
| `AIC_UR5E_STATS_JSON` | `/opt/stats/ur5e_stats.json` |
| `RLINF_PATH` | `/home/graphai/project/aic/RLinf` |
| `RLINF_PYTHON` | `/home/graphai/project/aic/RLinf/.venv/bin/python` |

평가 시 외부에서 다음만 주입 (docker-compose.yaml에 이미 명시):
- `AIC_ROUTER_ADDR` (zenoh router 주소, 예: `eval:7447`)
- `AIC_MODEL_PASSWD` (ACL 활성화 시)
- `AIC_ENABLE_ACL=true` (선택, ACL 활성화)

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `ur5e stats already present` | 정상 — 빌드 시 주입 완료 |
| `[worker] checkpoint loaded: missing=0 unexpected=0` | 정상 — fine-tuned weights 완전 매칭 |
| `Unable to connect to any locator of scouted peer ...` | 정상 — zenoh가 router 찾는 중 (timeout 600s까지 대기) |
| 평가가 시작 안 됨 | `AIC_ROUTER_ADDR` 값 확인, eval 컨테이너 이름과 일치하는지 |

---

문의: 이 이미지 빌드 일자 2026-05-16, 빌드 환경 정보는 `SWEEP_REPORT.md` 참고.
