#!/bin/bash
set -euo pipefail

uv run mim install mmengine
uv run mim install mmcv==2.1.0
uv run mim install mmpretrain==1.2.0
uv run mim install mmdet==3.2.0
uv run mim install mmpose==1.3.2

# mim uses pip which can upgrade numpy past our pin — restore it
uv sync
