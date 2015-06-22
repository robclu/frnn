# cRNN

## Overview

cRNN is a C/C++ Recurrent Neural Network library. The long term aim is to provide support for all possible RNN configurations (Deep, LSTM, Bi-Directional ...), as well as CUDA and OpenCL. 

The long term goal is to provide support for both CUDA and OpenCL. CUDA provides support for modern C++ and is beneficial for systems which use Nvidia GPUs. Initially all parallel components will be implemented using CUDA. OpenCL is more similar to C, however, provides support for many vendors as well as for CPUs. OpenCL support will be implemented to take advantage of using the CPU and GPU in some places, as well as to provide support for powerful CPU only, or AMD GPU systems.

## Support 

* CUDA   : Cuda 7.0 API
* OpenCL : Not yet supported

## Current Functionality 

The current version is v1.0.0 and supports the following functionality:

__Note__: See the wiki for details on the functions and conceptual explanations. 

## Compiling

### Development system 

The following system parameters are used for testing:

