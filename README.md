# Supporting material for NeuralCubes

This repository contains additional material that supports the manuscript [NeuralCubes: Deep Representations for Visual Data Exploration](https://arxiv.org/abs/1808.08983), by Zhe Wang, Dylan Cashman, Mingwei Li, Jixian Li, Matthew Berger, Joshua A. Levine, Remco Chang, and Carlos Scheidegger.

Specifically, the repository contains source code to define, train and test NeuralCubes.
* The architecture of a NeuralCubes model is (mostly) customizable through a json file. A sample configuration is included as `cfg_bk_nyc_10k.json`. Currently, only Fully Connected (FC) layers and `ReLU` activation function are supported.
* The implementation is developed and tested with `Pytorch 0.4`.
* This repo also provides preprocessed training and testing data generated from the [BrightKite social network dataset](https://snap.stanford.edu/data/loc-Brightkite.html).

## Training
To train a NeuralCubes model, execute: `./train_bk_nyc.sh`

## Testing
To test the trained model, execute: `./test_bk_nyc.sh`

# Acknowledgments

This material is based upon work supported or partially supported by the National Science Foundation under Grant Number 1815238, project titled "III: Small: An end-to-end pipeline for interactive visual analysis of big data"

Any opinions, findings, and conclusions or recommendations expressed in this project are those of author(s) and do not necessarily reflect the views of the National Science Foundation.
