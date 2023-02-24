# Intro
Code for HDFormer: High-order Directed Transformer for 3D Human Pose Estimation

# Training
```bash
set +x
DATASET_TYPE=H36M
EXP_ROOT=output


EXP_NAME=v52
EXP_CONFIG=hdformer.yaml
bash scripts/train.sh ${EXP_NAME} config/${EXP_CONFIG} ${EXP_ROOT}
```
