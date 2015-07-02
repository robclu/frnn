# cuRNN

## Overview

cuRNN is a Cuda Recurrent Neural Network library. The long term aim is to provide support for all possible RNN configurations (Deep, LSTM, Bi-Directional ...), as well as for OpenCL, eventually. 

The long term goal is to provide support for both CUDA and OpenCL. CUDA provides support for modern C++ and is beneficial for systems which use Nvidia GPUs. Initially all parallel components will be implemented using CUDA. OpenCL is more similar to C, however, provides support for many vendors as well as for CPUs. OpenCL support will be implemented to take advantage of using the CPU and GPU in some places, as well as to provide support for powerful CPU only, or AMD GPU systems.

## Support 

### Software

* C++ : 11 (Testing with gtest)
* CUDA   : Cuda 7.0 API
* OpenCL : Not yet supported
* Linux  : Tested and working
* Windows : Not yet tested
* OSX : Not yet tested

### Hardware

* CUDA Compute Capability >= 3.0 

## Current Functionality 

The current version is v1.0.0 and supports the following functionality:

__Note__: See the wiki for details on the functions and conceptual explanations (coming soon). 

* Math :
  * axpy
  * sum (sum of a vector, result is returned) 
  * sumVectorized (sum of a vector, each element gets result)
  * softmax 

## Compiling and Running 

__Note__: All makefiles assume that CUDA is installed as the default cuda-7.0 in /usr/local.
          If this is not the case you will need to change this in the Makefiles. 

Tests are written for each of the components of the library. The tests for the individual components can be
run or all tests can be run.

### Running All Tests

To run all tests, go to the __src__ directory and issue the following commands
```
make 
./all_tests
```

### Running Individual Component Tests

To run the tests for an indivdual component, got the __src/component__ directory, where __component__ is
the component you want to run the tests for (for example math), then similarly issue
```
make 
./component_tests
```

### Cleaning

Issuing 
```
make clean
```
will clean any executable after making it.

### Development system 

The following system parameters are used for testing:

OS &nbsp; : Linux Ubuntu 14.04  
CPU : Intel i7-3615QM (Quad-core @ 2.3GHz)  
GPU : Nvidia GeForce GT 650M (344 cores, 900Mhz)  
