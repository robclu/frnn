/*
 *  Header file for fastRNN tensor exception class.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */ 

#ifndef _FRNN_TENSOR_EXCEPTION_
#define _FRNN_TENSOR_EXCEPTION_

#include <iostream>
#include <exception>
#include <string>

namespace frnn {

/*
 * ==========================================================================================================
 * Class        : TensorOutOfRange
 * Description  : Class which provides the exception for when an element of a dimension of a tensor which 
 *                is out of the range of the dimension (accessing element 3 of dimension 2 if dimension 2 has
 *                a size of 1)
 * ==========================================================================================================
 */
class TensorOutOfRange : public std::exception {
public:
    std::string _message;           // Error message for exception
public:
    /*
     * ======================================================================================================
     * Function     : TensorOutOfRange
     * Description  : Constructor for the TensorOutOfRange class. Sets the error message using the inputs
     * Inputs       : dimension         : The dimension from which the element is being accessed
     *              : dimension_size    : The size of the dimension
     *              : index             : The index of the element which is trying to be accessed
     * ======================================================================================================
     */
    TensorOutOfRange(const int dimension, const int dimension_size, const int index) 
    : _message("Error : Out of range : Attempted to access invalid tensor element "    + 
                std::to_string(index)                                                  + 
                " of dimension "                                                       +
                std::to_string(dimension)                                              +
                " which has size "                                                     +
                std::to_string(dimension_size)                                         +
                " : Note : tensors are 0 indexed"                                      ) {} 
   
    /*
     * ======================================================================================================
     * Function     : TensorOutOfRange
     * Description  : Constructor for the TensorOutOfRange class. Sets the error message using the inputs
     * Inputs       : dimension     : The dimension which is trying to be accessed
     *              : rank          : The rank of the tensor
     * ======================================================================================================
     */
    TensorOutOfRange(const int dimension, const int rank) 
    : _message("Error : Out of range : Attempted to access invalid dimension "  +   
                std::to_string(dimension)                                       + 
                " of tensor with rank "                                         +
                std::to_string(rank)                                            +
                " returning value of 0"                                         ) {}
    
    /*
     * ======================================================================================================
     * Function     : what 
     * Description  : Function for specifying the error message for out of range tensor acccess
     * Outputs      : The error message
     * ======================================================================================================
     */
    const char* what() const throw() { return _message.c_str(); }
};

/*
 * ==========================================================================================================
 * Class        : TensorInvalidArguments
 * Description  : Class which provides the exception for when an invalid number of arguments are provided to a
 *                tensor function, and can be used for any of the variadic functions where the number of 
 *                arguments is known
 * ==========================================================================================================
 */
class TensorInvalidArguments : public std::exception {
public:
    std::string _message;
public:
    /*
     * ======================================================================================================
     * Function     : TensorInvalidArguments
     * Description  : Constructor which sets the error message
     * Inputs       : num_args_specified    : The number of arguments given to the function
     *              : num_args_required     : The number of arguments required by the function
     * ======================================================================================================
     */ 
    TensorInvalidArguments(const int num_args_specified, const int num_args_required) 
    : _message("Error : Invalid Arguments for tensor : "    +
                std::to_string(num_args_required)           +
                " arguments required, "                     +
                std::to_string(num_args_specified)          +
                " given"                                    ) {}
    
    /*
     * ======================================================================================================
     * Function     : what 
     * Description  : Function for specifying the error message for invalid arguments for tensor functions
     * Outputs      : The error message
     * ======================================================================================================
     */    
    const char* what() const throw() { return _message.c_str(); }
};

}   // End namspace frnn

#endif
