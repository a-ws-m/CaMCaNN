# Models

All of the models trained during the research are available in this directory. Each model's name starts with either `ecfp`, for the linear models based on extended-connectivity fingerprints (ECFPs), or `gnn`, for the graph neural network (GNN) models. The name also indicates which subset of the Qin data the models were trained/tested on: either all of the data, or just the nonionic surfactants.

The sensitivity models' names also contain the syntax `<num_splits>-splits-trial-<trial_idx>`. The four benchmarking models are stored in [ecfp-all](ecfp-all), [ecfp-nonionics](ecfp-nonionics), [gnn-search-all](gnn-search-all) and [gnn-search-nonionics](gnn-search-nonionics). The GNN benchmarking folders also contain the best hyperparameters found during the Hyperband search (`best_hps.json`).

All of the GNN models contain `logs` subdirectories with data that can be read by Tensorboard:

```bash
tensorboard --logdir logs/
```
