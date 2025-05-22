# Error models

This folder contains the error models defined for NVIDIA GPU. Models have been derived by means of fault injection experiments on cudaDNN test applications running individual DNN operators and then mining frequently-observed patterns. Fault injection has been performed by means of  [NVBitFI](https://github.com/fernandoFernandeSantos/nvbitfi)), rmodels definition by means of the CNN error classifier tool (available in this [repository](https://github.com/D4De/cnn-error-classifier) ). We defined to sets of models, each one stored in a different folder:
- `faultythread_models`, derived by injecting a random value in a single CUDA thread
- `faultywarp_models`, derived by injecting a random value in all the CUDA threads of the same warp