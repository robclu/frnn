// ==========================================================================================================
//! @file   Header file for fastRNN tensor class.
// ==========================================================================================================

/*
 * ==========================================================================================================
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
 * ==========================================================================================================
 */

#ifndef _FRNN_NEW_TENSOR_
#define _FRNN_NEW_TENSOR_

#include "tensor_expressions.h"
#include "tensor_exceptions.h"
#include "tensor_indices.h"
#include "../containers/tuple.h"

#include <iostream>
#include <cassert>
#include <initializer_list>
#include <numeric>
#include <type_traits>

namespace frnn {

// ==========================================================================================================
//! @class  Tensor 
//! @brief  Allows an N dimensional space to be created to store data and operate on the data.               \n
//!                                                                                                          \n
//!         Detailed usage of the Tensor class can be found in the unit tests for the class.                 \n
//!         The basic usage is:                                                                              \n
//!                                                                                                          \n
//!         Tensor<int, 3> tensor = {2, 2, 2}           // Create a 3-dimensional (rank 3) Tensor            \n
//!                                                                                                          \n
//!         The above Tensor is of int type and each dimension has a size of 2, thus the Tensor is a         \n
//!         matrix with 2 pages. Operations can be performed such as:                                        \n
//!                                                                                                          \n
//!         Tensor<int, 3> new_tensor = tensor + tensor + tensor    // Add 3 Tensors                         \n
//!         Tensor<int, 2> slice_tensor = tensor(j,i)               // New tensor from dim 1 and 2 of old 
//!                                                                    tensor                                \n
//! @tparam T   Type of data used by the Tensor.
//! @tparam R   Rank of the Tensor (the number of dimensions it has).
// ==========================================================================================================
template <typename T, const size_t R>
class Tensor : public TensorExpression<T, Tensor<T, R>> {
public:
    /* =========================================== Typedefs =============================================== */
    using typename TensorExpression<T, Tensor<T,R>>::container_type;
    using typename TensorExpression<T, Tensor<T,R>>::size_type;
    using typename TensorExpression<T, Tensor<T,R>>::value_type;
    using typename TensorExpression<T, Tensor<T,R>>::reference;
    /* ==================================================================================================== */
private:
    container_type          _data;                  //!< Container to hold Tensor data elements
    std::vector<size_type>  _dimensions;            //!< Sizes of each of the Tensor's dimensions
    size_type               _counter;               //!< Iteration of the elemen offset calculation 
    size_type               _offset;                //!< For accessing elements with operator()
public:
     // =====================================================================================================
     //! @brief     Sets the member variables to 0, and the number of dimensions to the rank.
     // =====================================================================================================
    Tensor() : _data(0), _dimensions(R), _counter(0), _offset(0) {}
    
    /*
     * ======================================================================================================
     * Function     : Tensor 
     * Description  : Constructor for the tensor class using an initializer list
     * Inputs       : dimensions    : The list of dimensions where each element in the list specifies how
     *                                many elements are in the dimension. The nth element specifies the 
     *                                size of the (n+1)th dimensions, ie element 0 specifies the size of 
     *                                the first dimension
     * ======================================================================================================
     */
    Tensor(std::initializer_list<int> dimensions) 
    : _data(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>())),
      _counter(0), _offset(0)
    {   
        ASSERT(dimensions.size(), ==, R); 
        for (auto& element : dimensions) _dimensions.push_back(element);
    }
    
    /*
     * ======================================================================================================
     * Function     : Tensor
     * Description  : Constructor from a tensor expression, which is used to create new tensors from
     *                operations on other tensors, minimizing the copy overhead
     * Params       : E         : The expression which is the operation on the other tensors (addition for
     *                            example)
     * ======================================================================================================
     */
    template <typename E>
    Tensor(TensorExpression<T,E> const& tensor) 
    : _dimensions(tensor.dimSizes()), _counter(0), _offset(0)
    {
        E const& t = tensor;
        _data.resize(t.size());
        for (size_type i = 0; i != t.size(); ++i) {
            _data[i] = t[i];
        }
    }
   
    /* 
     * ======================================================================================================
     * Function     : Tensor 
     * Description  : Constructor using vectors for the data and dimensions
     * ======================================================================================================
     */
    Tensor(std::vector<size_type>& dimensions, container_type& data) 
    : _dimensions(std::move(dimensions)), _data(std::move(data)), _counter(0), _offset(0) 
    {
        ASSERT(_dimensions.size(), ==, R);           // Check number of dimensions is equal to the rank
        ASSERT(_data.size(), ==,                     // Check total data size is consistent with dim sizes
               std::accumulate(_dimensions.begin()          , 
                               _dimensions.end()            , 
                               1                            , 
                               std::multiplies<size_type>() ));
    }
  
    /* 
     * ======================================================================================================
     * Function     : size
     * Description  : Gets the size of the tensor (the number of elements it holds)
     * Outputs      : The number of elements in the tensor
     * ======================================================================================================
     */
    size_type size() const { return _data.size(); }

    /* 
     * ======================================================================================================
     * Function     : size
     * Description  : Overloaded size function to return the size of a specified dimension 
     * Inputs       : dim   : The dimension for which the size must be returned
     * Outputs      : The size of the dimension dim
     * ======================================================================================================
     */
    size_type size(const int dim) const 
    {
        try {
            if (dim >= R) throw TensorOutOfRange(dim, R);
            return _dimensions[dim];
        } catch (TensorOutOfRange& e) {
            std::cout << e.what() << std::endl;
            return 0;
        }
    }
    
    /*
     * ======================================================================================================
     * Function     : rank
     * Description  : Returns the rank (number of dimensions) of the tensor
     * Outputs      : The rank of the tensor
     * ======================================================================================================
     */
    size_type rank() const { return R; }

    /*
     * ======================================================================================================
     * Function     : dimSizes
     * Description  : Gets the sizes of the dimensions of the tensor
     * Outputs      : A constant reference to the dimension sizes of the tensor
     * ======================================================================================================
     */
    const std::vector<size_type>& dimSizes() const { return _dimensions; }
     
    /*
     *  =====================================================================================================
     *  Function    : data
     *  Description : Returns a constant reference to the data 
     *  Outputs     : A counstant reference to the tensor data
     *  =====================================================================================================
     */
    const container_type& data() const { return _data; }

    /* 
     * ======================================================================================================
     * Function     : operator[]
     * Description  : Overloaded access operator to a reference to an element from the tensor data
     * Inputs       : i     : The index of the element to access 
     * Outputs      : A reference to the element
     * ======================================================================================================
     */
    reference operator[](size_type i) { return _data[i]; }
    
    /* 
     * ======================================================================================================
     * Function     : operator[]
     * Description  : Overloaded access operator to get an element from the tensor data
     * Inputs       : i     : The index of the element to access
     * Outputs      : An element of the tensor data
     * ======================================================================================================
     */
    value_type operator[](size_type i) const { return _data[i]; }

    /*
     * ======================================================================================================
     * Function     : operator() (for slicing)
     * Description  : Creates a tensor expression which is a slice of the tensor invoking the call
     * Inputs       : dims      : The dimensions to slice
     * Outputs      : A TensorSlice which is a slice of this tensor
     * Params       : Ts        : The types of the dims values
     * ======================================================================================================
     */
    template <typename... Ts>
    TensorSlice<T, Tensor<T,R>, Ts...> operator()(Ts... dims) const 
    {
        return TensorSlice<T, Tensor<T,R>, Ts...>(static_cast<Tensor<T,R> const&>(*this),
                                                  Tuple<Ts...>(dims...)                 );          
    }
    
    /*
     * ======================================================================================================
     * Function     : operator() (for setting)
     * Description  : Terminating case of operator(), for when there is only a single element
     * Inputs       : idx       : The index of the element in the last dimension (the variadic version 
     *                            would have been called fist for the other dimensions
     * Params       : I         : The type of the idx argument
     * ======================================================================================================
     */
    template <typename I>
    typename std::enable_if<std::is_arithmetic<I>::value, T&>::type  operator() (I idx) 
    {
        try {                                                           // Check in range
            if (idx >= _dimensions[_counter]) {                         // counter +1 in next line for
                throw TensorOutOfRange(_counter + 1, _dimensions[_counter], idx);   // 0 indexing offset
            }
        } catch (TensorOutOfRange& e) {
            std::cerr << e.what() << std::endl;
            _counter = 0;
            return _data[0];
        }
        try {                                                           // Make sure variadic version 
            if (_counter == 0) throw TensorInvalidArguments(1, R);      // has been called already
        } catch (TensorInvalidArguments& e) {
            std::cerr << e.what() << std::endl;
        }
        _offset += std::accumulate(_dimensions.begin()      , 
                                   _dimensions.end() - 1    ,           // Mult all elements except last
                                   1                        ,           // Start val for multiplication
                                   std::multiplies<int>()   ) * idx;    // Add offset due to idx for dim
        _counter = 0;                                                   // Reset counter for next iter
        return _data[_offset];
    }
    
    /* 
     * ======================================================================================================
     * Function     : operator() (for setting)
     * Description  : Variadic case of operator() so that the offset is determined for any rank tensor
     * Inputs       : idx       : The index for the current dimension so that the offset for the 
     *                            dimension can be added
     *              : indices   : The rest of the indecies for the other dimensions
     * Params       : I         : The type of the idx argument
     *              : Is        : The types for the remaining indecies
     * ======================================================================================================
     */
    template <typename I, typename... Is>
    typename std::enable_if<std::is_arithmetic<I>::value, T&>::type operator()(I idx, Is... indices) 
    {
        const int num_args = sizeof...(Is);
        try {                                                           // Check index in range
            if (idx >= _dimensions[_counter]) {                         // counter + 1 int next line for
                throw TensorOutOfRange(_counter + 1, _dimensions[_counter], idx);   // 0 indexing offset
            }
        } catch (TensorOutOfRange& e ) {
            std::cout << e.what() << std::endl;
            _counter = 0;
            return _data[0];
        }   
        if (_counter++ == 0) {                                          // Case for first index
            try {                                                       // Check correct number of arguments
                if (num_args + 1 !=  R) throw TensorInvalidArguments(num_args + 1, R);
                _offset = idx;
            } catch (TensorInvalidArguments& e) {
                std::cerr << e.what() << std::endl;
                return _data[0];
            }  
        } else {                                                        // Case for all other indecies
            _offset += std::accumulate(_dimensions.begin()              , 
                                       _dimensions.end() - num_args - 1 ,
                                       1                                , 
                                       std::multiplies<int>()           ) * idx;
        }
        return this->operator()(indices...);
    }  
   
    /*
     * ======================================================================================================
     * Function     : operator() (for getting)
     * Description  : Terminating case of operator(), for when there is only a single element
     * Inputs       : idx       : The index of the element in the last dimension (the variadic version 
     *                            would have been called fist for the other dimensions
     * Params       : I         : The type of the idx argument
     * ======================================================================================================
     */
    template <typename I>
    typename std::enable_if<std::is_arithmetic<I>::value, const T&>::type operator()(I idx) const 
    {
        try {                                                           // Check in range
            if (idx >= _dimensions[_counter]) {                         // counter +1 in next line for
                throw TensorOutOfRange(_counter + 1, _dimensions[_counter], idx);   // 0 indexing offset
            }
        } catch (TensorOutOfRange& e) {
            std::cerr << e.what() << std::endl;
            _counter = 0;
            return _data[0];
        }
        try {                                                           // Make sure variadic version 
            if (_counter == 0) throw TensorInvalidArguments(1, R);      // has been called already
        } catch (TensorInvalidArguments& e) {
            std::cerr << e.what() << std::endl;
        }
        _offset += std::accumulate(_dimensions.begin()      , 
                                   _dimensions.end() - 1    ,           // Mult all elements exepct last
                                   1                        ,           // Start val for multiplication
                                   std::multiplies<int>()   ) * idx;    // Add offset due to idx for dim
        _counter = 0;                                                   // Reset counter for next iter
        return _data[_offset]; 
    }
    
    /* 
     * ======================================================================================================
     * Function     : operator() (for getting)
     * Description  : Variadic case of operator() so that the offset is determined for any rank tensor
     * Inputs       : idx       : The index for the current dimension so that the offset for the 
     *                            dimension can be added
     *              : indecies  : The rest of the indecies for the other dimensions
     * Params       : I         : The type of the idx argument
     *              : Is        : The types for the remaining indecies
     * ======================================================================================================
     */
    template <typename I, typename... Is>
    typename std::enable_if<std::is_arithmetic<I>::value, const T&>::type 
    operator()(I idx, Is... indecies) const
    {
        const int num_args = sizeof...(Is);
        try {                                                           // Check index in range
            if (idx >= _dimensions[_counter]) {                         // counter + 1 int next line for
                throw TensorOutOfRange(_counter + 1, _dimensions[_counter], idx);   // 0 indexing offset
            }
        } catch (TensorOutOfRange& e ) {
            std::cout << e.what() << std::endl;
            _counter = 0;
            return _data[0];
        }   
        if (const_cast<int&>(_counter)++ == 0) {                        // Case for first index
            try {                                                       // Check correct number of arguments
                if (num_args + 1 !=  R) throw TensorInvalidArguments(num_args + 1, R);
                _offset = idx;
            } catch (TensorInvalidArguments& e) {
                std::cerr << e.what() << std::endl;
                return _data[0];
            }  
        } else {                                                        // Case for all other indecies
            _offset += std::accumulate(_dimensions.begin()              , 
                                       _dimensions.end() - num_args - 1 ,
                                       1                                , 
                                       std::multiplies<int>()           ) * idx;
        }
        return this->operator()(indecies...);
    }  
};

}   // End namespace frnn

#endif
