# Intro
Code for HDFormer: High-order Directed Transformer for 3D Human Pose Estimation

# Train
```bash
bash run_train.sh
```

# Test
```bash
bash run_test.sh
```
The pretrained model of Human3.6M with 2D GT input can be found in `checkpoints/model`.

# Dataset
Set the dataset path `data_path` in `config/hdformer.yaml`. Note that the Human3.6M dataset need licenses, the developer should apply for authorisation from [Human3.6M](http://vision.imar.ro/human3.6m/description.php).


# Citation
```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = { IEEE Transactions on Pattern Analysis and Machine Intelligence}，
  publisher = {IEEE Computer Society}，
  year = {2014}
}
```
```bibtex
@article{chen2023-hdformer,
  title = {HDFormer: High-order Directed Transformer for 3D Human Pose Estimation},
  author = {Chen, Hanyuan and He, Jun-Yan and Xiang, Wangmeng and Liu, Wei and Cheng, Zhi-Qi and Liu, Hanbing and Luo, Bin and Geng, Yifeng and Xie, Xuansong},
  year = {2023},
  eprint = {2302.01825},
  doi = {10.48550/arXiv.2302.01825},
}
```
