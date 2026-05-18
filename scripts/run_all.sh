#!/bin/bash
set -euo pipefail

# Usage: bash scripts/run_all.sh [video_dir] [project_name]
# Defaults: video_dir=stored_vids/, project_name=default_project
# Requires: apptainer, uv, GPU node (sinteractive or sbatch)

SCRIPT_DIR_EARLY="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR_EARLY="$(cd "$SCRIPT_DIR_EARLY/.." && pwd)"

VIDEO_DIR="${1:-$REPO_DIR_EARLY/stored_vids}"
PROJECT="${2:-default_project}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STORE_DIR="$REPO_DIR/stored_vids"

echo "PosePipeline — Full Run"
echo "  Repo:    $REPO_DIR"
echo "  Videos:  $VIDEO_DIR"
echo "  Project: $PROJECT"

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. You likely need a GPU node."
    echo "  sinteractive --gres=gpu:1 --mem=32G --time=04:00:00"
    read -p "Continue anyway? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

if command -v module &>/dev/null; then
    module load apptainer 2>/dev/null || true
fi

cd "$REPO_DIR/docker"
bash run_datajoint.sh start
cd "$REPO_DIR"

mkdir -p "$STORE_DIR"

if [ ! -f "$REPO_DIR/dj_local_conf.json" ]; then
    cat > "$REPO_DIR/dj_local_conf.json" <<EOF
{
    "database.host": "127.0.0.1",
    "database.password": "pose",
    "database.port": 3306,
    "database.user": "root",
    "database.reconnect": false,
    "database.use_tls": false,
    "enable_python_native_blobs": true,
    "fetch_format": "array",
    "loglevel": "INFO",
    "safemode": true,
    "stores": {
        "localattach": {
            "protocol": "file",
            "location": "$STORE_DIR"
        }
    },
    "custom": {
        "database.prefix": ""
    }
}
EOF
    echo "Created dj_local_conf.json"
fi

uv sync --inexact

uv run python scripts/run_pipeline.py --video_dir "$VIDEO_DIR" --project "$PROJECT" --output_dir "$VIDEO_DIR/output"

echo "Done! Results saved to: $VIDEO_DIR/output/"
