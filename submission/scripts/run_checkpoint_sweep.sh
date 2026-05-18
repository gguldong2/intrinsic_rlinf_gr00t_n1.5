#!/usr/bin/env bash
# Automated sweep over GR00T checkpoints.
# Per checkpoint:
#   1. clean all leftover processes (host + container)
#   2. start record_trial_videos.py with per-step OUTPUT_DIR
#   3. start pixi ros2 node with GROOT_CHECKPOINT_PATH
#   4. wait for worker "model ready"
#   5. run /entrypoint.sh inside aic_eval container (with timeout)
#   6. parse score, append to results JSONL
#   7. cleanup

set -u

# ---- config ----
CKPT_ROOT="/home/graphai/checkpoints/aic_sft_v3"
STEPS=(${STEPS_ARG:-5000 6000 7000 8000 9000 11000})
# Each step uses a different ROS_DOMAIN_ID so zenoh graphs don't bleed
# stale "aic_model" entries from prior iterations into the new run.
DOMAIN_BASE=120
RESULTS_FILE="/home/graphai/checkpoints/aic_sft_v3/sweep_results.jsonl"
PIXI_LOG="/tmp/pixi_run.log"
WORKER_LOG="/tmp/gr00t_worker.log"
RECORD_LOG="/tmp/record_trial_videos.log"
ENGINE_LOG_DIR="/tmp/engine_logs"
VIDEO_ROOT="/home/graphai/aic_results/videos"
mkdir -p "$ENGINE_LOG_DIR" "$VIDEO_ROOT"

# Per-eval entrypoint timeout. 3 trials × ~40s + setup ~60s ≈ 200s. Give ample buffer.
EVAL_TIMEOUT=420

cleanup_host() {
  pkill -f "ros2 run aic_model" 2>/dev/null
  pkill -f "gr00t_worker.py"     2>/dev/null
  pkill -f "record_trial_videos.py" 2>/dev/null
  pkill -f "docker exec aic_eval"   2>/dev/null
  sleep 2
}

cleanup_container() {
  # The aic_engine launches a ros2_launch that doesn't always exit cleanly.
  # Restarting the whole container is the only reliable way to get fresh
  # state (~10s overhead, worth it for sweep stability).
  docker restart aic_eval >/dev/null 2>&1
  # Wait for container daemons to settle
  sleep 8
}

cleanup_ros2_daemon() {
  ~/ws_aic/src/aic/.pixi/envs/default/bin/ros2 daemon stop >/dev/null 2>&1 || true
}

wait_worker_ready() {
  local deadline=$(( $(date +%s) + 300 ))
  while [ $(date +%s) -lt $deadline ]; do
    if grep -q "\[worker\] model ready" "$WORKER_LOG" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_one_eval() {
  local engine_log="$1"
  local domain_id="$2"
  # timeout ensures we don't hang forever if entrypoint never returns
  timeout --signal=KILL "$EVAL_TIMEOUT" \
    docker exec -e ROS_DOMAIN_ID="$domain_id" \
                -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
                aic_eval bash -lc "/entrypoint.sh ground_truth:=false start_aic_engine:=true" \
    > "$engine_log" 2>&1
  return $?
}

parse_score() {
  local engine_log="$1"
  local total
  total=$(grep -oE "Total Score:[[:space:]]+[0-9.]+" "$engine_log" | tail -1 | awk '{print $3}')
  echo "${total:-FAILED}"
}

parse_trials() {
  local engine_log="$1"
  python3 -c "
import re, json, sys
with open('$engine_log') as f:
    data = f.read()
# strip ANSI color codes
data = re.sub(r'\x1b\[[0-9;]*m', '', data)
result = {}
total_m = re.search(r'Total Score:\s*([0-9.]+)', data)
if total_m: result['total'] = float(total_m.group(1))
for tid in ('trial_1','trial_2','trial_3'):
    block = re.search(rf'{tid}:(.+?)(?=trial_\d:|\Z)', data, re.DOTALL)
    if not block: continue
    blk = block.group(1)
    tier_scores = re.findall(r'(tier_[123]):\s*\n\s*score:\s*([0-9.]+)', blk)
    distance_m = re.search(r'Final plug port distance:\s*([0-9.]+)m', blk)
    succ_m = re.search(r'✓ Trial.*Score:\s*([0-9.]+)', blk)
    trial_score_m = re.search(r'completed successfully! Score:\s*([0-9.]+)', blk)
    result[tid] = {
        'tiers': {k: float(v) for k,v in tier_scores},
        'final_distance_m': float(distance_m.group(1)) if distance_m else None,
    }
print(json.dumps(result))
" 2>/dev/null
}

# ---- main loop ----
echo "[$(date '+%F %T')] sweep start (steps: ${STEPS[*]})" | tee -a "$RESULTS_FILE.log"
iter=0
for step in "${STEPS[@]}"; do
  iter=$((iter + 1))
  domain_id=$(( DOMAIN_BASE + iter ))  # 43, 44, 45, ...  each iteration unique
  ckpt="$CKPT_ROOT/global_step_${step}/actor/model_state_dict/full_weights.pt"
  video_dir="$VIDEO_ROOT/step_${step}"
  mkdir -p "$video_dir"

  if [ ! -f "$ckpt" ]; then
    echo "[$(date '+%F %T')] SKIP step $step (file not found: $ckpt)" | tee -a "$RESULTS_FILE.log"
    continue
  fi

  echo "" | tee -a "$RESULTS_FILE.log"
  echo "=== [$(date '+%F %T')] step $step (ROS_DOMAIN_ID=$domain_id) ===" | tee -a "$RESULTS_FILE.log"
  echo "checkpoint: $ckpt" | tee -a "$RESULTS_FILE.log"
  echo "videos    : $video_dir" | tee -a "$RESULTS_FILE.log"

  cleanup_host
  cleanup_ros2_daemon
  cleanup_container

  : > "$WORKER_LOG"
  : > "$RECORD_LOG"

  # Start video recorder for this step (same ROS_DOMAIN_ID as everything else)
  (
    cd /home/graphai/ws_aic/src/aic && \
    AIC_VIDEO_DIR="$video_dir" \
    ROS_DOMAIN_ID="$domain_id" \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
      pixi run python3 \
        /home/graphai/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/record_trial_videos.py
  ) > "$RECORD_LOG" 2>&1 &
  record_pid=$!
  echo "record PID: $record_pid → $video_dir" | tee -a "$RESULTS_FILE.log"

  # Start pixi ros2 node in background
  (
    cd /home/graphai/ws_aic/src/aic && \
    GROOT_CHECKPOINT_PATH="$ckpt" \
    ROS_DOMAIN_ID="$domain_id" \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
      pixi run ros2 run aic_model aic_model \
        --ros-args -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.RunGR00T
  ) > "$PIXI_LOG" 2>&1 &
  pixi_pid=$!
  echo "pixi PID: $pixi_pid" | tee -a "$RESULTS_FILE.log"

  echo "Waiting for worker model ready..." | tee -a "$RESULTS_FILE.log"
  if ! wait_worker_ready; then
    echo "TIMEOUT waiting for model ready" | tee -a "$RESULTS_FILE.log"
    cleanup_host
    cleanup_container
    echo "{\"step\":$step,\"total\":null,\"error\":\"worker_not_ready\"}" >> "$RESULTS_FILE"
    continue
  fi
  echo "Model ready." | tee -a "$RESULTS_FILE.log"

  # Trigger eval
  engine_log="$ENGINE_LOG_DIR/engine_step_${step}.log"
  echo "Running engine → $engine_log (timeout ${EVAL_TIMEOUT}s, domain=$domain_id)" | tee -a "$RESULTS_FILE.log"
  run_one_eval "$engine_log" "$domain_id"
  rc=$?
  if [ $rc -eq 137 ] || [ $rc -eq 124 ]; then
    echo "WARN: engine timed out (rc=$rc) — will still parse partial scores" | tee -a "$RESULTS_FILE.log"
  fi

  # Parse score(s)
  total=$(parse_score "$engine_log")
  trials_json=$(parse_trials "$engine_log")
  echo "Step $step total: $total" | tee -a "$RESULTS_FILE.log"

  if [ -z "$trials_json" ]; then
    echo "{\"step\":$step,\"total\":${total:-null},\"engine_log\":\"$engine_log\",\"video_dir\":\"$video_dir\"}" >> "$RESULTS_FILE"
  else
    echo "{\"step\":$step,\"engine_log\":\"$engine_log\",\"video_dir\":\"$video_dir\",\"details\":$trials_json}" >> "$RESULTS_FILE"
  fi

  cleanup_host
  cleanup_ros2_daemon

  # Show video file count for this step
  n_videos=$(ls -1 "$video_dir"/*.mp4 2>/dev/null | wc -l)
  echo "Videos saved: $n_videos" | tee -a "$RESULTS_FILE.log"
done

echo "" | tee -a "$RESULTS_FILE.log"
echo "[$(date '+%F %T')] sweep done. Results: $RESULTS_FILE" | tee -a "$RESULTS_FILE.log"
echo "Summary:"
cat "$RESULTS_FILE"
