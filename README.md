# fastRNN

## Overview

fastRNN is a fast Recurrent Neural Network library, which uses C++, CUDA and OpenMP. It supports all possible configurations of RNN's.

Support will be provided for both CPU-GPU and CPU-only systems, and will determine which device to use for each function based on its complexity.

## Documentation

The code is well documented, with descripitons being provided for all classes, structs, functions and variables. 

Additionally more thourough documentation will be provided on the [fastRNN website](http://robclu.github.io/fastRNN/), however, this will be slow as the primary focus is to first provide functionality with documentation in the code, and then for thorough documentation.

## Compiling and Running

### Pre-requisits

#### CUDA (currently required for GPU and CPU)

You will need to have the CUDA SDK installed as a large component of the code uses CUDA. Currently, there are not separate components in the makefiles for GPU and CPU versions, if there is no GPU the compiler will select the CPU implementations of the functions, thus the CUDA SDK is required for both versions (I realize this is a problem, and will work on rectifying it).

Additionally, the Makefiles assume that CUDA is installed as __/usr/local/cuda-7.0__, if this is not the case you will need to change this in the makefiles.

#### g++

The makefiles use g++, thus it is required to build the code (again this will be changed to support any compiler).

__Note__: The intention is to have a config file in the future to allow for a custom install.

### Individual Components

Tests are written for each of the component of the library and the makefile for the component can be found in its directory. For example, to test the functionality for the Tensor class, navigate to the Tensor directory (__/src/tensor__) and then (if the pre-requisits are installed) run ```make``` to make the component tests. 

The tests can then be run with ```./<conponent_name>_tests```

### All Components

All tests for all components can also be run at once. Navigate to the __/src__ directory and then run ```make`` which will make all tests.

All the test can then be run by ```./all_tests```

### Cleaning

To clean components or all the tests, run ```make clean``` from the same directory from which the ```make``` command was issued, which will clean the executable and the object files.
