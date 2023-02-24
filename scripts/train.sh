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

PYTHON=python3
dataset=H36M
# dataset=MPIINF3DHP
TRAIN_CODE=train.py
TEST_CODE=test.py
exp_name=$1
config=$2
EXP_HOME=$3

exp_dir=${EXP_HOME}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p -m 775 ${exp_dir}
mkdir -p -m 775 ${model_dir}
mkdir -p -m 775 ${result_dir}
mkdir -p -m 775 ${result_dir}/last
mkdir -p -m 775 ${result_dir}/best

cp scripts/train.sh scripts/test.sh main/${TRAIN_CODE} main/${TEST_CODE} ${config} ${exp_dir}
config=${exp_dir}/${config##*/}

export PYTHONPATH=.
echo $OMP_NUM_THREADS | tee -a ${exp_dir}/train-$now.log
nvidia-smi | tee -a ${exp_dir}/train-$now.log
which pip3 | tee -a ${exp_dir}/train-$now.log

echo "start train"
$PYTHON -u ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  log_dir ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log

# $PYTHON -u ${exp_dir}/test.py \
#  --config=${config} \
#  save_folder ${result_dir}/last \
#  model_path ${model_dir}/model_last.pth.tar \
#  2>&1 | tee ${exp_dir}/test_last-$now.log

$PYTHON -u ${exp_dir}/test.py \
 --config=${config} \
 save_folder ${result_dir}/best \
 model_path ${model_dir}/model_best.pth.tar \
 2>&1 | tee ${exp_dir}/test_best-$now.log
