# Identifying and Understanding Cross-Class Features in Adversarial Training (ICML 2025)
This is the official repository for ICML 2025 paper *Identifying and Understanding Cross-Class Features in Adversarial Training*, including code for adversarial training on CIFAR-10/100, class-wise feature extraction, and class-wise feature attribution correlation matrix visualization.

### Requirements
- Python 3.9+
- Common libs like `torch`, `torchvision`, `numpy`, `tqdm`, `matplotlib`, `pandas`.
- Data: CIFAR datasets download to `/data/cifar_data` by default (edit dataset paths in code if needed).

### Usage
- Train (run from repo root)
  - `python -m train --fname AT_run --dataset cifar10 --mode AT --norm linf --ne 200 --lr 0.1 --bs 128`
  - Modes: `AT`, `TRADES`, `none`
  - Outputs: `train_log/<fname>/config.json`, `log.csv`, `model_best.pth`, `model_last.pth`, periodic `model_<epoch>.pth`, `opt_<epoch>.pth`
- Extract features
  - `python -m show_feature --fname AT_run --model train_log/AT_run/model_best.pth --arch PRN18 --dataset cifar10 --norm linf --eps 8`
  - Saves: `features/<fname>/test_clean.pth` and `features/<fname>/test_adv.pth`
- Correlation and figures
  - `python -m evaluate_matrix --fname AT_run --model train_log/AT_run/model_best.pth --arch PRN18 --dataset cifar10 --norm linf --eps 8`
  - Saves: correlation matrices and optional diffs to `figs/<fname>/`
  - Note: you can load features from `features/<fname>/test_clean.pth` and `features/<fname>/test_adv.pth` to avoid re-extracting features.


## Citation
```
@inproceedings{wei2025identifying,
  title={Identifying and Understanding Cross-Class Features in Adversarial Training},
  author={Wei, Zeming and Guo, Yiwen and Wang, Yisen},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```