# DEGNN PAKDD2024
This is a PyTorch implementation of PAKDD 2024 paper [DEGNN: Dual Experts Graph Neural Network Handling Both Edge and Node Feature Noise](https://link.springer.com/chapter/10.1007/978-981-97-2253-2_30). This paper is also available at [arxiv](https://arxiv.org/abs/2404.09207).

## Requirements
```
python==3.10.9
numpy==1.23.5
scipy==1.9.3
torch==1.13.1
```

## Uasge
All the hyper-parameters are included in `best_parameters.json`.

You can run:
```
python main.py --dataset cora --model DEGNN1 --num_exp 10
```
To run experiments with extra noise, spesify them by the parameters `--edge_noise_ratio` and `--node_noise_ratio`.

## Cite
```
@inproceedings{hasegawa2024degnn,
  title={DEGNN: Dual Experts Graph Neural Network Handling both Edge and Node Feature Noise},
  author={Hasegawa, Tai and Yun, Sukwon and Liu, Xin and Phua, Yin Jun and Murata, Tsuyoshi},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={376--389},
  year={2024},
  organization={Springer}
}
```
