# Modified Gpufit source code

This directory contains modified source code from the Gpufit project. The Gpufit code is licensed under the MIT license and its notice is included in here in LICENSE.

The model code is in the models subdirectory. The constants.h and models.cuh files link the model functions into the gpufit software.

It is also necessary to modify the gpufit.py file that resides with the compiled module to include the model ID number for the new fit models.