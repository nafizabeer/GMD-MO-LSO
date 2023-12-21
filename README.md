# GMD-MO-LSO
Codes for [Multi-Objective Latent Space Optimization of Generative Molecular Design Models](https://arxiv.org/abs/2203.00526) will be uploaded here.

<!-- <div align="center"> -->


</div>

## Description
This code repository contains scripts to run the multi-objective weighted retraining on molecular generative model, i.e. [JT-VAE](https://proceedings.mlr.press/v80/jin18a.html).

We followed the codebase from [Sample-Efficient Optimization in the Latent Space of Deep Generative Models via Weighted Retraining](https://github.com/cambridge-mlg/weighted-retraining). For the necessary dependencies, please follow their instructions.

## Running multi-objective weighted retraining

The bash script to run the multi-objective weighted retraining can be found in `scripts/opt/mo_opt_chem.sh`. Currently it has the commands for all six possible pairs of logP, SAS, NP_score and DRD2. One pair can be run at a time. For this reason other 5 commands are commented out.

The important arguments for `mo_opt_chem.sh`:

- `--pretrained_model_file` is the path to baseline model. For the case of complete dataset, it is same for all pairs. For reduced dataset, the models for each pair can be found in `assets/pretrained_model`
- `--train_path` points to the directory where the tensor data for training molecules are stored. Note that, for the case of reduced dataset, this directory must correpond to the appropriate property pairs.
- `--all_new` is set 1 if we want to put all the new molecules in the next stage of weighted retraining. If it is 0, then all new molecules are added to the current training dataset from which only 10% random samples are used in weighted retraining stage. 
<!-- - `--weight_type` controls the type of weighting (either rank weighting or one of the baseline strategies) -->
- `--rank_weight_k` controls the degree of weighting in formulation of weight from Pareto front rank
<!-- - `--query_budget` controls the budget of function evaluations -->
- `--retraining_frequency` controls the retraining frequencies (number of epochs between retrainings)
- `--result_root` is the directory to save the results in
<!--   (in particular, a file `results.npz` is created) in this location
  which contains the results to be plotted. -->


