#!/bin/bash
set -euo pipefail

IMAGE="docker://datajoint/mysql:8.0"
SIF="mysql_8.0.sif"
INSTANCE_NAME="datajoint-db"
DATA_DIR="./data"
MYSQL_PORT=3306
MYSQL_ROOT_PASSWORD="pose"

start() {
    [ ! -f "$SIF" ] && apptainer pull "$SIF" "$IMAGE"
    mkdir -p "$DATA_DIR"

    if apptainer instance list | grep -q "$INSTANCE_NAME"; then
        echo "Already running."
        exit 0
    fi

    apptainer instance start \
        --bind "$DATA_DIR":/var/lib/mysql \
        --env MYSQL_ROOT_PASSWORD="$MYSQL_ROOT_PASSWORD" \
        --env MYSQL_TCP_PORT="$MYSQL_PORT" \
        --writable-tmpfs \
        --no-home \
        "$SIF" "$INSTANCE_NAME"

    echo "Started on port $MYSQL_PORT"
}

stop()   { apptainer instance stop "$INSTANCE_NAME"; }
status() { apptainer instance list; }
shell()  { apptainer shell instance://"$INSTANCE_NAME"; }

case "${1:-}" in
    start)  start ;;
    stop)   stop ;;
    status) status ;;
    shell)  shell ;;
    *)      echo "Usage: $0 {start|stop|status|shell}"; exit 1 ;;
esac

