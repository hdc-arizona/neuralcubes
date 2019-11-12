# NeuralCubes

This repo contains code to define, train and test NeuralCubes.
The architecture of a NeuralCubes model is (mostly) customizable through a json file.
A sample configuration is included as `cfg_bk_nyc_10k.json`.
Currently, only Fully Connected (FC) layers and `ReLU` activation function are supported.

## Training
`./train_bk_nyc.sh`

## Testing
`./test_bk_nyc.sh`
