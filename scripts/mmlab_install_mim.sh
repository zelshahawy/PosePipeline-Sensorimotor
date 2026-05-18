#!/bin/bash
set -euo pipefail

uv run mim install mmcv==2.1.0
uv pip install platformdirs yapf opencv-python-headless
