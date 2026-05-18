# AIC Cable-Insert GR00T Submission Materials

This directory contains everything our team produced for the AIC cable-insert
challenge that is not directly part of the upstream `aic` codebase: evaluation
automation, analysis reports, sample trial videos, the self-contained-image
Dockerfile, and the deployment guide.

The actual policy code (the only modification we made inside the upstream
codebase) lives in `aic_example_policies/aic_example_policies/ros/`:

- `RunGR00T.py` (modified)
- `gr00t_worker.py` (new)
- `record_trial_videos.py` (new)

And the verification-time Docker assets are in `docker/aic_policy_gr00t/`
and `docker/docker-compose-policy.yaml` at the repo root.

## Layout

```
submission/
├── README.md                    ← this file
├── DEPLOY.md                    ← team-lead deployment guide (load → ECR push → portal)
├── docker/
│   └── Dockerfile.submission    ← self-contained build (~25 GB image)
├── scripts/
│   ├── run_checkpoint_sweep.sh  ← multi-checkpoint evaluation sweep
│   └── watch_sweep.sh           ← live progress dashboard
├── reports/
│   ├── SWEEP_REPORT.md          ← 16-checkpoint detailed evaluation
│   ├── sweep_results.jsonl      ← per-step Total + trial details
│   ├── sweep_analysis.json      ← parsed per-trial tier breakdown
│   └── sweep_results.jsonl.log  ← sweep execution log
└── videos/                      ← trial videos for top-scoring steps
    ├── step_13000/  (68.95)
    ├── step_15000/  (76.16 — best)
    ├── step_18000/  (72.39)
    └── step_19000/  (73.24 — chosen for submission)
```

## TL;DR — Build and ship

1. **Verify locally** (see verification Dockerfile in `../docker/aic_policy_gr00t/`).
2. **Build the submission image** using `submission/docker/Dockerfile.submission`:
   - assemble a build context with the layout shown at the top of the Dockerfile
   - `docker build -f Dockerfile.submission -t aic-policy-gr00t:submit-stepN .`
3. **Follow `DEPLOY.md`** for `docker load` → ECR tag → push → portal registration.

## Selected checkpoint

`step 19000` from
`liqejdy/aic_cable_insert_sft_gr00t_n15_pretrained_v3_bs96_mb24_step5596`.

Rationale (full detail in `reports/SWEEP_REPORT.md`):

- Total ≈ 73.24 (1 run) / 67.02 (verify run inside submission image) — within
  the ±8 variance observed across the sweep.
- step 18000 (72.39) immediately before it — paired high band 18k–19k
  suggests the model is consistently in the 70+ regime at that point.
- step 15000 scored 76.16 (highest single run) but the very next snapshot
  (16788) dropped to 63.18, so 15000 is more likely a positive outlier.

## How to reproduce the sweep

Inside this repo (with the pixi env and an aic_eval container available):

```bash
# Download checkpoints (HF auth required for the source repo):
hf auth login
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
REPO = 'liqejdy/aic_cable_insert_sft_gr00t_n15_pretrained_v3_bs96_mb24_step5596'
TARGET = '/path/to/checkpoints'
for step in [4000, 6000, 11000, 13000, 15000, 18000, 19000, 26000]:
    hf_hub_download(repo_id=REPO,
        filename=f'global_step_{step}/actor/model_state_dict/full_weights.pt',
        local_dir=TARGET)
EOF

# Run the sweep
STEPS_ARG="4000 6000 11000 13000 15000 18000 19000" bash scripts/run_checkpoint_sweep.sh

# Live dashboard in another terminal
bash scripts/watch_sweep.sh
```

Sweep produces per-step engine logs, per-step `videos/step_<N>/` MP4s, and
appends results to `reports/sweep_results.jsonl`.
