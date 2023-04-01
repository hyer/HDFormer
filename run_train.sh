set +x
DATASET_TYPE=H36M
EXP_ROOT=output


EXP_CONFIG=hdformer.yaml
bash scripts/train.sh config/${EXP_CONFIG} ${EXP_ROOT}
