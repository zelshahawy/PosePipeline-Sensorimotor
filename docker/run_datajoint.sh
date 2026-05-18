#!/bin/bash
set -euo pipefail

IMAGE="docker://datajoint/mysql:8.0"
SIF="mysql_8.0.sif"
INSTANCE_NAME="datajoint-db"
DATA_DIR="./data"
MYSQL_PORT=3306
MYSQL_ROOT_PASSWORD="pose"

init_db() {
    echo "Initializing MySQL data directory..."
    mkdir -p "$DATA_DIR"

    # Initialize with no password first
    apptainer exec --writable-tmpfs --bind "$DATA_DIR":/var/lib/mysql \
        "$SIF" mysqld --initialize-insecure --user="$(whoami)" --datadir=/var/lib/mysql

    # Start mysqld temporarily to set the root password and allow remote connections
    apptainer exec --writable-tmpfs --bind "$DATA_DIR":/var/lib/mysql \
        "$SIF" bash -c "
            mkdir -p /var/run/mysqld
            mysqld --user=$(whoami) --datadir=/var/lib/mysql --port=$MYSQL_PORT --socket=/var/run/mysqld/mysqld.sock &
            MPID=\$!
            for i in \$(seq 1 60); do
                if mysqladmin ping --socket=/var/run/mysqld/mysqld.sock --silent 2>/dev/null; then
                    mysql --socket=/var/run/mysqld/mysqld.sock -u root <<SQL
ALTER USER 'root'@'localhost' IDENTIFIED BY '$MYSQL_ROOT_PASSWORD';
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY '$MYSQL_ROOT_PASSWORD';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
SQL
                    echo 'Database initialized with password.'
                    mysqladmin --socket=/var/run/mysqld/mysqld.sock -u root -p$MYSQL_ROOT_PASSWORD shutdown
                    wait \$MPID 2>/dev/null || true
                    exit 0
                fi
                sleep 1
            done
            echo 'ERROR: mysqld did not start during init.'
            kill \$MPID 2>/dev/null || true
            exit 1
        "
}

start() {
    [ ! -f "$SIF" ] && apptainer pull "$SIF" "$IMAGE"

    # Initialize data directory if empty
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        init_db
    fi

    if apptainer instance list | grep -q "$INSTANCE_NAME"; then
        echo "Already running."
        return 0
    fi

    apptainer instance start \
        --bind "$DATA_DIR":/var/lib/mysql \
        --writable-tmpfs \
        --no-home \
        "$SIF" "$INSTANCE_NAME"

    # Apptainer doesn't run Docker entrypoints, so start mysqld manually
    apptainer exec instance://"$INSTANCE_NAME" bash -c \
        "mkdir -p /var/run/mysqld && mysqld --user=$(whoami) --datadir=/var/lib/mysql --port=$MYSQL_PORT --socket=/var/run/mysqld/mysqld.sock &"

    echo "Waiting for MySQL..."
    for i in $(seq 1 60); do
        if apptainer exec instance://"$INSTANCE_NAME" mysqladmin ping -h 127.0.0.1 -u root -p"$MYSQL_ROOT_PASSWORD" --silent 2>/dev/null; then
            echo "MySQL is ready (took ${i}s)."
            return 0
        fi
        sleep 1
    done

    echo "ERROR: MySQL did not start within 60s."
    exit 1
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
