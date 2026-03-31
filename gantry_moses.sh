#!/usr/bin/env bash

set -euo pipefail

beaker_whoami="${BEAKER_WHOAMI:-$(beaker account whoami --format json | jq -r '.[0].name')}"
python_args=("$@")

if [ "${#python_args[@]}" -eq 0 ]; then
    python_args=("test")
fi

gantry run \
    --workspace ai2/oe-adapt-code \
    --budget ai2/oe-adapt \
    --cluster ai2/neptune \
    --priority high \
    --gpus 1 \
    --timeout 5h \
    --task-name gantry-moses \
    --beaker-image docker://molecularsets/moses \
    --venv base \
    --install "echo 'Skipping Gantry Python setup'" \
    --weka=oe-adapt-default:/weka/oe-adapt-default \
    -- /bin/bash -lc 'mkdir -p /root && ln -sfn "/weka/oe-adapt-default/'"${beaker_whoami}"'/.config" /root/.config && exec /opt/miniconda/bin/python "$@"' bash "${python_args[@]}"
