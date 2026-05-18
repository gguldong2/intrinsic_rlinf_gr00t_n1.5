#!/usr/bin/env bash
# Live monitor of the checkpoint sweep evaluation.
# Usage:  bash ~/watch_sweep.sh    (Ctrl+C to exit)

watch -tn 2 '
  printf "\033[1;36m========== AIC Checkpoint Sweep — $(date +%H:%M:%S) ==========\033[0m\n\n"

  printf "\033[1;33m[1] 현재 진행 중인 체크포인트\033[0m\n"
  latest=$(ls -t /tmp/engine_logs/engine_step_*.log 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
      step=$(basename "$latest" .log | sed "s/engine_step_//")
      printf "    step %s  →  %s\n" "$step" "$latest"
      printf "\n\033[1;33m[2] 이 체크포인트의 trial 진행\033[0m\n"
      grep -E "Trial [0-9]/3|✓ Trial .* Score:|Total Score" "$latest" 2>/dev/null \
          | sed "s/\x1b\[[0-9;]*m//g" \
          | tail -8 \
          | awk "{printf \"    %s\n\", \$0}"
  else
      printf "    (아직 평가 시작 전)\n"
  fi

  printf "\n\033[1;33m[3] sweep 전체 흐름 (마지막 10줄)\033[0m\n"
  tail -10 /home/graphai/checkpoints/aic_sft_v3/sweep_results.jsonl.log 2>/dev/null \
      | sed "s/^/    /"

  printf "\n\033[1;33m[4] 완료된 체크포인트 점수 (sweep_results.jsonl)\033[0m\n"
  if [ -s /home/graphai/checkpoints/aic_sft_v3/sweep_results.jsonl ]; then
      cat /home/graphai/checkpoints/aic_sft_v3/sweep_results.jsonl | sed "s/^/    /"
  else
      printf "    (아직 없음)\n"
  fi

  printf "\n\033[1;33m[5] 워커 로딩 상태\033[0m\n"
  tail -3 /tmp/gr00t_worker.log 2>/dev/null | grep -E "\[worker\]" | sed "s/^/    /"
'
