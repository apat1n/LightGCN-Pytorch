### LightGCN on Pytorch

This is a implementation of LightGCN ([Paper in arXiv](https://arxiv.org/abs/2002.02126)) neural net from SIGIR 2020

### Supported datasets:
- [gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- [brightkite](https://snap.stanford.edu/data/loc-brightkite.html)

Use `prepare_<dataset_name>_dataset.py` for download and splitting by time

### Supported models:
- [iALS](https://implicit.readthedocs.io/en/latest/als.html) is matrix factorization model from `implicit` open-source library
- TopNModel recommends top items from all user feedback
- TopNPersonalized recommends top items from unique user feedback
- TopNNearestModel recommends nearest by last user location items (domain-specific for geo features)
- [LightGCN](https://arxiv.org/abs/2002.02126)
- [Catboost](https://catboost.ai) fitting with LogLoss/YetiRank and ranking candidates

### Training:

Main script is `train.py` which trains model from `MODEL` setting in `config.yaml` file

Also there is `fit_catboost.py` script which trains catboost ranking model