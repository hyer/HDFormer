#!/bin/sh
# LOG
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
set -o pipefail

export OMP_NUM_THREADS=10
export KMP_INIT_AT_FORK=FALSE

PYTHON=python

dataset=H36M
#dataset=MPIINF3DHP
TEST_CODE=test.py

config=$1
EXP_HOME=$2

exp_dir=${EXP_HOME}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
now=$(date +"%Y%m%d_%H%M%S")

cp scripts/test.sh main/${TEST_CODE} ${exp_dir}

export PYTHONPATH=./

#$PYTHON -u ${exp_dir}/${TEST_CODE} \
# --config=${config} \
# save_folder ${result_dir}/last \
# model_path ${model_dir}/model_last.pth.tar \
# 2>&1 | tee ${exp_dir}/test_last-$now.log

$PYTHON -u ${exp_dir}/${TEST_CODE} \
 --config=${config} \
 save_folder ${result_dir}/best \
 model_path ${model_dir}/model_best.pth.tar \
 2>&1 | tee ${exp_dir}/test_best-$now.log
