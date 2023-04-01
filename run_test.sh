set +x
DATASET_TYPE=H36M

EXP_CONFIG=hdformer.yaml
bash scripts/test.sh config/${EXP_CONFIG} checpoints
