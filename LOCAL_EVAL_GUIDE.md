# GR00T 정책 로컬 평가 가이드

GR00T N1.5-3B SFT 체크포인트를 로컬 머신에서 Gazebo 시뮬레이션으로 평가하는 절차.

---

## 필수 조건 (로컬 머신)

| 항목 | 요구사항 |
|------|----------|
| OS | Ubuntu 22.04 이상 (24.04 권장) |
| GPU | NVIDIA 12GB VRAM 이상 권장 (최소 8GB) |
| NVIDIA Container Toolkit | 설치됨 |
| Docker Engine | 미설치 시 아래에서 설치 |
| Distrobox | 미설치 시 아래에서 설치 |
| pixi | 미설치 시 아래에서 설치 |

---

## Step 1: 필수 도구 설치

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA 런타임 연결 (Container Toolkit이 이미 설치된 경우)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi  # 확인

# Distrobox
curl -s https://raw.githubusercontent.com/89luca89/distrobox/main/install \
  | sh -s -- --prefix ~/.local
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# pixi
curl -fsSL https://pixi.sh/install.sh | sh && source ~/.bashrc
```

---

## Step 2: 레포 클론

### 2-1. aic 레포 (평가 프레임워크 + 정책)
```bash
mkdir -p ~/ws_aic/src && cd ~/ws_aic/src
git clone https://github.com/intrinsic-dev/aic
cd aic
git checkout feature/gr00t-policy   # RunGR00T.py 포함된 브랜치
pixi install                        # ROS 2 + aic 패키지 설치
```

### 2-2. RLinf 레포 (Python 코드 — rlinf 패키지, gr00t 모델 코드)
> ⚠️ RLinf를 클론하는 이유는 체크포인트 파일 때문이 아닙니다.
> `rlinf` Python 패키지 (get_model, simulation_io 등)가 필요합니다.
> 체크포인트(.pt)는 어디에 두어도 무관합니다.

```bash
cd ~
git clone https://github.com/Phy-lab-aic/RLinf.git
cd RLinf
git checkout feature/1-gr00tn15-ur5e
bash requirements/install.sh embodied --model gr00t --env maniskill_libero
```

---

## Step 3: 파일 확보

### 3-1. GR00T 베이스 모델 (HuggingFace에서 다운로드)
```bash
~/RLinf/.venv/bin/huggingface-cli download nvidia/GR00T-N1.5-3B \
    --local-dir ~/models/gr00t-n1.5-3b
# 약 6GB
```

### 3-2. SFT 체크포인트 (학습 서버에서 복사)
체크포인트는 RLinf 레포 내부일 필요가 없습니다. 원하는 경로에 저장하세요.

```bash
# 방법 A: scp (SSH 접근 가능한 경우)
mkdir -p ~/checkpoints
scp -P <PORT> elicer@central-01.tcp.tunnel.elice.io:\
"/home/elicer/project/RLinf/logs/20260511-20:09:35/pretrained_v3_sft_gr00t/checkpoints/global_step_1399/actor/model_state_dict/full_weights.pt" \
~/checkpoints/gr00t_pretrained_v3_step1399.pt

# 방법 B: HuggingFace (업로드해둔 경우)
huggingface-cli download <username>/gr00t-checkpoint full_weights.pt \
  --local-dir ~/checkpoints/
```

---

## Step 4: RunGR00T.py 경로 수정

파일: `~/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/RunGR00T.py`

상단 3개 상수를 로컬 경로에 맞게 수정:

```python
_RLINF_PATH      = "/home/<username>/RLinf"                          # RLinf 클론 경로
_BASE_MODEL_PATH = "/home/<username>/models/gr00t-n1.5-3b"          # GR00T 베이스 모델
_CHECKPOINT_PATH = "/home/<username>/checkpoints/gr00t_pretrained_v3_step1399.pt"  # SFT 체크포인트
```

---

## Step 5: aic_eval 컨테이너 준비

```bash
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest

# NVIDIA GPU 사용 시
distrobox create -r --nvidia \
  -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval
```

---

## Step 6: 평가 실행 (터미널 3개)

### Terminal 1 — Gazebo 시뮬레이션 + 채점 엔진
```bash
distrobox enter -r aic_eval
/entrypoint.sh ground_truth:=false start_aic_engine:=true
# Gazebo 창 + RViz 창이 뜨면 정상
# "Waiting for aic_model..." = policy 연결 대기 중
```

### Terminal 2 — GR00T 정책 실행
```bash
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.RunGR00T
# 모델 로드 (~30초) 후 자동으로 Trial 1 시작
```

### Terminal 3 — 결과 확인 (선택)
```bash
watch -n 2 cat ~/aic_results/scoring.yaml
```

---

## 평가 구조

3 Trials 자동 진행, 총 최대 300점:

| Trial | 태스크 |
|-------|--------|
| 1 | SFP Module → SFP_PORT (NIC 카드 랜덤 위치) |
| 2 | SFP Module → SFP_PORT (다른 랜덤 seed) |
| 3 | SC Plug → SC_PORT (다른 커넥터 타입) |

점수 구조 (Trial당 최대 100점):
- Tier 1: 모델 유효성 (0~1점)
- Tier 2: 동작 품질 (smoothness, duration, efficiency, 페널티)
- Tier 3: 삽입 성공 (성공 75점, 오삽입 -12점)

결과: `~/aic_results/scoring.yaml`

---

## 파이프라인 검증 (권장)

첫 실행 전, CheatCode 정책으로 파이프라인이 정상인지 먼저 확인:

```bash
# Terminal 1: Gazebo 실행 (위와 동일)

# Terminal 2: CheatCode (정답 좌표를 알고 삽입 — 만점 기준)
cd ~/ws_aic/src/aic
pixi run ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

CheatCode 실행 후 scoring.yaml에서 Tier 3 = 75점이 나오면 평가 환경 정상.
이후 RunGR00T로 전환하여 실제 점수 측정.

---

## 요약 — 클론해야 하는 레포

| 레포 | 브랜치 | 이유 |
|------|--------|------|
| `intrinsic-dev/aic` | `feature/gr00t-policy` | 평가 프레임워크 + RunGR00T 정책 |
| `Phy-lab-aic/RLinf` | `feature/1-gr00tn15-ur5e` | rlinf Python 패키지 (모델 코드) |

체크포인트 (`full_weights.pt`, 5.1GB)와 GR00T 베이스 모델 (~6GB)은 별도 확보 필요.
