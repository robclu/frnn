# fastRNN

## Overview

fastRNN is a fast Recurrent Neural Network library. The long term aim is to provide support for all possible RNN configurations (Deep, LSTM, Bi-Directional, etc...), with a fast implementation. 

Gpu support will be provided for CUDA and OpenCL (later, so that all gpu vendors are supported). The implementation will use cpu or gpu functions whereever the most speedup can be gained, and will provide full cpu support if there are no gpus present in the system.

## Support 

### Software

The currently supported/used software is:

* C++11 (Testing with gtest)
* CUDA 7.0
* Linux 

### Hardware

* CUDA Compute Capability >= 3.0 

## Current Functionality 

The current version is v1.0.0 and supports the following functionality:

__Note__: See the wiki for details on the functions and conceptual explanations (coming soon). 

* Data :
  * tensor4 (4D tensor)
* Layers :
  * softmax (negative log liklihood loss)
* Math :
  * axpy (a x X + Y) [GPU]
  * sum (sum of a vector, result is returned) [GPU]
  * sumVectorized (sum of a vector, each element gets result) [GPU]
  * softmax [GPU]
  * xmy (X - Y) [CPU]


## Compiling and Running 

__Note__: All makefiles assume that CUDA is installed as the default cuda-7.0 in /usr/local.
          If this is not the case you will need to change this in the Makefiles. 

__Note__: There will be a config file in the future that will allow for a custom install.

Tests are written for each of the components of the library. The tests for each component can be run individually, or all tests can be run at once.

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

__Note__: The development system will upgrade in the (near) future to include multiple cpu and gpus and will support mpi.
