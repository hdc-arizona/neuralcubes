# NeuralCubes

* This repo contains code to define, train and test NeuralCubes.
* The architecture of a NeuralCubes model is (mostly) customizable through a json file. A sample configuration is included as `cfg_bk_nyc_10k.json`. Currently, only Fully Connected (FC) layers and `ReLU` activation function are supported.
* The implementation is developed and tested with `Pytorch 0.4`.
* This repo also provides preprocessed training and testing data generated from the [BrightKite social network dataset](https://snap.stanford.edu/data/loc-Brightkite.html).

## Training
To train a NeuralCubes model, execute: `./train_bk_nyc.sh`

## Testing
To test the trained model, execute: `./test_bk_nyc.sh`
