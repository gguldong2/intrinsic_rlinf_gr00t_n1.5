#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# AIC GR00T policy container entrypoint
#
# Required env vars (provided by docker-compose):
#   AIC_ROUTER_ADDR        zenoh router for aic_eval container (e.g. eval:7447)
#   GROOT_MODEL_PATH       path to GR00T-N1.5-3B base model dir
#   GROOT_CHECKPOINT_PATH  path to fine-tuned full_weights.pt
#   AIC_UR5E_STATS_JSON    path to LeRobot stats.json (for metadata injection)
#   RLINF_PATH             path to RLinf repo root (worker uses its venv)
#   RLINF_PYTHON           path to RLinf venv python (Python 3.11)
# Optional:
#   AIC_ENABLE_ACL=true    enable zenoh ACL (with AIC_MODEL_PASSWD)
# ---------------------------------------------------------------------------

export RMW_IMPLEMENTATION=rmw_zenoh_cpp

# -- sanity check ----------------------------------------------------------
for v in AIC_ROUTER_ADDR GROOT_MODEL_PATH GROOT_CHECKPOINT_PATH RLINF_PATH RLINF_PYTHON; do
  if [ -z "${!v}" ]; then
    echo "[entrypoint] FATAL: env var $v must be set"
    exit 1
  fi
done
for p in "$GROOT_MODEL_PATH" "$GROOT_CHECKPOINT_PATH" "$RLINF_PATH" "$RLINF_PYTHON"; do
  if [ ! -e "$p" ]; then
    echo "[entrypoint] FATAL: path does not exist: $p"
    exit 1
  fi
done

# -- inject ur5e statistics into metadata.json if not yet present ----------
META_PATH="$GROOT_MODEL_PATH/experiment_cfg/metadata.json"
if [ -f "$META_PATH" ] && [ -n "$AIC_UR5E_STATS_JSON" ] && [ -f "$AIC_UR5E_STATS_JSON" ]; then
  need_inject=$(python3 - <<PY
import json, sys
m = json.load(open("$META_PATH"))
ur5e = m.get("ur5e", {})
stats = ur5e.get("statistics", {}).get("action", {}).get("shoulder_pan_joint", {})
# Inject if missing, or if max==min==0 (zero-stat fallback)
mn, mx = stats.get("min", [0])[0], stats.get("max", [0])[0]
print(int(not stats or (mn == 0 and mx == 0)))
PY
)
  if [ "$need_inject" = "1" ]; then
    echo "[entrypoint] injecting ur5e stats into $META_PATH ..."
    python3 /opt/inject_ur5e_stats.py --model-dir "$GROOT_MODEL_PATH" --stats-json "$AIC_UR5E_STATS_JSON"
  else
    echo "[entrypoint] ur5e stats already present"
  fi
fi

# -- zenoh config ----------------------------------------------------------
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_ROUTER_ADDR"'"]'
ZENOH_CONFIG_OVERRIDE+=';transport/shared_memory/enabled=false'

if [[ "$AIC_ENABLE_ACL" == "true" || "$AIC_ENABLE_ACL" == "1" ]]; then
  if [[ -z "$AIC_MODEL_PASSWD" ]]; then
    echo "[entrypoint] FATAL: AIC_ENABLE_ACL set but AIC_MODEL_PASSWD missing"
    exit 1
  fi
  echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/user="model"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'"'
  ZENOH_CONFIG_OVERRIDE+=';transport/auth/usrpwd/dictionary_file="/credentials.txt"'
fi
export ZENOH_CONFIG_OVERRIDE
echo "[entrypoint] ZENOH_CONFIG_OVERRIDE=$ZENOH_CONFIG_OVERRIDE"

# Pass through GR00T paths
export GROOT_MODEL_PATH GROOT_CHECKPOINT_PATH RLINF_PATH RLINF_PYTHON

# -- launch policy ---------------------------------------------------------
cd /ws_aic/src/aic
echo "[entrypoint] starting aic_model with GR00T policy ..."
exec pixi run --as-is ros2 run aic_model aic_model "$@"
