#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="/mnt/data/lcx1/yiqinworkspace/openpi"
LIBERO_PYTHON="/mnt/data/lcx1/yiqinworkspace/clone_env_smoke_test/rlinf_env/bin/python"
CHECKPOINT="${OPENPI_DIR}/asset_pi05_base/pytorch/model.safetensors"
NORM_STATS_DIR="${NORM_STATS_DIR:-${OPENPI_DIR}/RLinf-Pi05-LIBERO-SFT}"
GPUS=(${GPUS:-4 5 6 7})
EPISODES_PER_TASK="${EPISODES_PER_TASK:-10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
TASK_SUITES="${TASK_SUITES:-libero_spatial libero_object libero_goal libero_10}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-240}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${OPENPI_DIR}/logs/pi05_base_libero_eval/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

suite_index=0
for suite in ${TASK_SUITES}; do
    gpu="${GPUS[$((suite_index % ${#GPUS[@]}))]}"
    port=$((18000 + gpu))
    server_log="${OUTPUT_DIR}/${suite}_server_gpu${gpu}.log"
    worker_log="${OUTPUT_DIR}/${suite}_worker_gpu${gpu}.log"
    result_file="${OUTPUT_DIR}/${suite}.jsonl"

    echo "Starting ${suite} on physical GPU ${gpu} (visible cuda:0)"
    (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        export UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_DIR}/.uv-cache}"
        cd "${OPENPI_DIR}"
        uv run python src/openpi/rl/run_pi05_openpi_server.py \
            --checkpoint-path "${CHECKPOINT}" \
            --reference-dir "${NORM_STATS_DIR}" \
            --assets-dir "${OPENPI_DIR}/assets" \
            --device cuda:0 \
            --sample-method flow_ode \
            --num-denoise-steps 10 \
            --host 127.0.0.1 \
            --port "${port}" \
            --checkpoint-interval 0 \
            --tensorboard-log-dir "" \
            >"${server_log}" 2>&1 &
        server_pid=$!
        trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

        ready=0
        for _ in $(seq 1 180); do
            if ! kill -0 "${server_pid}" 2>/dev/null; then
                tail -n 80 "${server_log}" >&2
                exit 1
            fi
            if "${LIBERO_PYTHON}" -c "import requests; raise SystemExit(0 if requests.get('http://127.0.0.1:${port}/health', timeout=2).ok else 1)" 2>/dev/null; then
                ready=1
                break
            fi
            sleep 2
        done
        [[ "${ready}" -eq 1 ]] || { echo "Server timeout for ${suite}" >&2; exit 1; }

        MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="${OPENPI_DIR}/src:${PYTHONPATH:-}" \
            "${LIBERO_PYTHON}" "${OPENPI_DIR}/scripts/base_model_scripts/eval_pi05_base_libero.py" \
            --server-url "http://127.0.0.1:${port}" \
            --task-suite "${suite}" \
            --task-ids ${TASK_IDS} \
            --episodes "${EPISODES_PER_TASK}" \
            --max-episode-steps "${MAX_EPISODE_STEPS}" \
            --execute-horizon "${EXECUTE_HORIZON}" \
            --checkpoint "${CHECKPOINT}" \
            --norm-stats-dir "${NORM_STATS_DIR}" \
            --output "${result_file}" \
            2>&1 | tee "${worker_log}"
    ) &
    pids+=("$!")
    suite_index=$((suite_index + 1))
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
[[ "${failed}" -eq 0 ]] || { echo "At least one evaluation worker failed; inspect ${OUTPUT_DIR}/*.log" >&2; exit 1; }

"${LIBERO_PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("libero_*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
successes = sum(row["successful_episodes"] for row in rows)
episodes = sum(row["completed_episodes"] for row in rows)
rate = successes / episodes if episodes else 0.0
stderr = math.sqrt(rate * (1 - rate) / episodes) if episodes else 0.0
summary = {
    "checkpoint": "/mnt/data/lcx1/yiqinworkspace/openpi/asset_pi05_base/pytorch/model.safetensors",
    "tasks_evaluated": len(rows),
    "completed_episodes": episodes,
    "successful_episodes": successes,
    "success_rate": rate,
    "success_rate_percent": 100 * rate,
    "approx_95_percent_ci_percent": [100 * max(0, rate - 1.96 * stderr), 100 * min(1, rate + 1.96 * stderr)],
    "per_task": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "Final success-rate report: ${OUTPUT_DIR}/summary.json"
