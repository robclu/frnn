/*
 *  Header file for fastRNN tensor class.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but w_ITHOUT AN_size.y WARRANTy_; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  y_ou should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_NEW_TENSOR_
#define _FRNN_NEW_TENSOR_

#include "tensor_expressions.h"
#include "tensor_exceptions.h"
#include "tensor_indices.h"

#include <iostream>
#include <cassert>
#include <initializer_list>
#include <numeric>

namespace frnn {

/*
 * ==========================================================================================================
 * Class        : Tensor
 * 
 * Description  : Tensor class for the fastRNN library. The class allows for a tensor of any dimension, thus
 *                providing great flexibility. For example, a 1D tensor is a vector or array, and a 2D tensor 
 *                is a matrix. 
 *
 * Params       : T     : The type of data used by the tensor
 *              : R     : The rank of the tensor 
 * ==========================================================================================================
 */
template <typename T, const size_t R>
class Tensor : public TensorExpression<T, Tensor<T, R>> 
{
    public:
        /* ======================================= Typedefs =============================================== */
        using typename TensorExpression<T, Tensor<T,R>>::container_type;
        using typename TensorExpression<T, Tensor<T,R>>::size_type;
        using typename TensorExpression<T, Tensor<T,R>>::value_type;
        using typename TensorExpression<T, Tensor<T,R>>::reference;
        /* ================================================================================================ */
    private:
        container_type          data_;                  // Data for tensor
        std::vector<size_type>  dimensions_;            // Sizes of the dimensions
        int                     counter_;               // Used for calculating the offset for operator()
        int                     offset_;                // For accessing elements with operator()
    public:
        /*
         * ==================================================================================================
         * Function     : Tensor
         * 
         * Description  : Default constructor for the tensor class 
         * ==================================================================================================
         */
        Tensor() : data_(0), dimensions_(R), counter_(0), offset_(0) {}
        
        /*
         * ==================================================================================================
         * Function     : Tensor 
         * 
         * Description  : Constructor for the tensor class using an initializer list 
         * 
         * Inputs       : dimensions    : The list of dimensions where each element in the list specifies how
         *                                many elements are in the dimension. The nth element specifies the 
         *                                size of the (n+1)th dimensions, ie element 0 specifies the size of 
         *                                the first dimension
         * ==================================================================================================
         */
        Tensor(std::initializer_list<int> dimensions) 
        : data_(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>())),
          counter_(0), offset_(0)
        {   
            ASSERT(dimensions.size(), ==, R); 
            for (auto& element : dimensions) dimensions_.push_back(element);
        }
        
        /*
         * ==================================================================================================
         * Function     : Tensor
         * 
         * Description  : Constructor from a tensor expression, which is used to create new tensors from
         *                operations on other tensors, minimizing the copy overhead
         *                
         * Params       : E         : The expression which is the operation on the other tensors (addition for
         *                            example)
         * ==================================================================================================
         */
        template <typename E>
        Tensor(TensorExpression<T,E> const& tensor) : counter_(0), dimensions_(tensor.dimSizes())
        {
            E const& t = tensor;
            data_.resize(t.size());
            for (size_type i = 0; i != t.size(); ++i) {
                data_[i] = t[i];
            }
        }
        
        /* 
         * ==================================================================================================
         * Function     : operator[]
         * 
         * Description  : Overloaded access operator to a reference to an element from the tensor data
         * 
         * Inputs       : i     : The index of the element to access 
         * 
         * Outputs      : A reference to the element
         * ==================================================================================================
         */
        reference operator[](size_type i) { return data_[i]; }
        
        /* 
         * ==================================================================================================
         * Function     : operator[]
         * 
         * Description  : Overloaded access operator to get an element from the tensor data
         * 
         * Inputs       : i     : The index of the element to access
         * 
         * Outputs      : An element of the tensor data
         * ==================================================================================================
         */
        value_type operator[](size_type i) const { return data_[i]; }
        
        /* 
         * ==================================================================================================
         * Function     : size
         * 
         * Description  : Gets the size of the tensor (the number of elements it holds)
         * 
         * Outputs      : The number of elements in the tensor
         * ==================================================================================================
         */
        size_type size() const { return data_.size(); }

        /* 
         * ==================================================================================================
         * Function     : size
         *  
         * Description  : Overloaded size function to return the size of a specified dimension 
         * 
         * Inputs       : dim   : The dimension for which the size must be returned
         * 
         * Outputs      : The size of the dimension dim
         * ==================================================================================================
         */
        size_type size(const int dim) const 
        {
            try {
                if (dim >= R) throw TensorOutOfRange(dim, R);
                return dimensions_[dim];
            } catch (TensorOutOfRange& e) {
                std::cout << e.what() << std::endl;
                return 0;
            }
        }
        
        /*
         * ==================================================================================================
         * Function     : rank
         * 
         * Description  : Returns the rank (number of dimensions) of the tensor
         * 
         * Outputs      : The rank of the tensor
         * ==================================================================================================
         */
        size_type rank() const { return R; }

        /*
         * ==================================================================================================
         * Function     : dimSizes
         * 
         * Description  : Gets the sizes of the dimensions of the tensor
         * 
         * Outputs      : A constant reference to the dimension sizes of the tensor
         * ==================================================================================================
         */
        const std::vector<size_type>& dimSizes() const { return dimensions_; }
         
        /*
         *  =================================================================================================
         *  Function    : data 
         *  
         *  Description : Returns a constant reference to the data 
         *  
         *  Outputs     : A counstant reference to the tensor data
         *  =================================================================================================
         */
        const container_type& data() const { return data_; }
        
        /*
         * ==================================================================================================
         * Function     : operator() (for setting)
         * 
         * Description  : Last case of operator(), for when there is only a single element
         * 
         * Inputs       : idx       : The index of the element in the last dimension (the variadic version 
         *                            would have been called fist for the other dimensions
         *                        
         * Params       : I         : The type of the idx argument
         * ==================================================================================================
         */
        template <typename I>
        T& operator() (I idx) 
        {
            try {                                                           // Check in range
                if (idx >= dimensions_[counter_]) {                         // counter +1 in next line for
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // 0 indexing offset
                }
            } catch (TensorOutOfRange& e) {
                std::cerr << e.what() << std::endl;
                counter_ = 0;
                return data_[0];
            }
            try {                                                           // Make sure variadic version 
                if (counter_ == 0) throw TensorInvalidArguments(1, R);      // has been called already
            } catch (TensorInvalidArguments& e) {
                std::cerr << e.what() << std::endl;
            }
            // If there are no errors
            offset_ += std::accumulate(dimensions_.begin()      , 
                                       dimensions_.end() - 1    ,           // Mult all elements exepct last
                                       1                        ,           // Start val for multiplication
                                       std::multiplies<int>()   ) * idx;    // Add offset due to idx for dim
            counter_ = 0;                                                   // Reset counter for next iter
            return data_[offset_];
        }
        
        /* 
         * ==================================================================================================
         * Function     : operator() (for setting)
         * 
         * Description  : Variadic case of operator() so that the offset is determined for any rank tensor
         * 
         * Inputs       : idx       : The index for the current dimension so that the offset for the 
         *                            dimension can be added
         *              : indices   : The rest of the indecies for the other dimensions
         *                        
         * Params       : I         : The type of the idx argument
         *              : Is        : The types for the remaining indecies
         * ==================================================================================================
         */
        template <typename I, typename... Is>
        T& operator()(I idx, Is... indices) 
        {
            const int num_args = sizeof...(Is);
            try {                                                           // Check index in range
                if (idx >= dimensions_[counter_]) {                         // counter + 1 int next line for
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // 0 indexing offset
                }
            } catch (TensorOutOfRange& e ) {
                std::cout << e.what() << std::endl;
                counter_ = 0;
                return data_[0];
            }   
            if (counter_++ == 0) {                                          // Case for first index
                try {                                                       // Check correct number of arguments
                    if (num_args + 1 !=  R) throw TensorInvalidArguments(num_args + 1, R);
                    offset_ = idx;
                } catch (TensorInvalidArguments& e) {
                    std::cerr << e.what() << std::endl;
                    return data_[0];
                }  
            } else {                                                        // Case for all other indecies
                offset_ += std::accumulate(dimensions_.begin()              , 
                                           dimensions_.end() - num_args - 1 ,
                                           1                                , 
                                           std::multiplies<int>()           ) * idx;
            }
            return this->operator()(indices...);
        }  
       
        /*
         * ==================================================================================================
         * Function     : operator() (for getting)
         * 
         * Description  : Last case of operator(), for when there is only a single element
         * 
         * Inputs       : idx       : The index of the element in the last dimension (the variadic version 
         *                            would have been called fist for the other dimensions
         *                        
         * Params       : I         : The type of the idx argument
         * ==================================================================================================
         */
        template <typename I>
        const T& operator()(I idx) const 
        {
            try {                                                           // Check in range
                if (idx >= dimensions_[counter_]) {                         // counter +1 in next line for
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // 0 indexing offset
                }
            } catch (TensorOutOfRange& e) {
                std::cerr << e.what() << std::endl;
                counter_ = 0;
                return data_[0];
            }
            try {                                                           // Make sure variadic version 
                if (counter_ == 0) throw TensorInvalidArguments(1, R);      // has been called already
            } catch (TensorInvalidArguments& e) {
                std::cerr << e.what() << std::endl;
            }
            // If there are no errors
            offset_ += std::accumulate(dimensions_.begin()      , 
                                       dimensions_.end() - 1    ,           // Mult all elements exepct last
                                       1                        ,           // Start val for multiplication
                                       std::multiplies<int>()   ) * idx;    // Add offset due to idx for dim
            counter_ = 0;                                                   // Reset counter for next iter
            return data_[offset_]; 
        }
        
        /* 
         * ==================================================================================================
         * Function     : operator() (for getting)
         * 
         * Description  : Variadic case of operator() so that the offset is determined for any rank tensor
         * 
         * Inputs       : idx       : The index for the current dimension so that the offset for the 
         *                            dimension can be added
         *              : indecies  : The rest of the indecies for the other dimensions
         *                        
         * Params       : I         : The type of the idx argument
         *              : Is        : The types for the remaining indecies
         * ==================================================================================================
         */
        template <typename I, typename... Is>
        const T& operator()(I idx, Is... indecies) const
        {
            const int num_args = sizeof...(Is);
            try {                                                           // Check index in range
                if (idx >= dimensions_[counter_]) {                         // counter + 1 int next line for
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // 0 indexing offset
                }
            } catch (TensorOutOfRange& e ) {
                std::cout << e.what() << std::endl;
                counter_ = 0;
                return data_[0];
            }   
            if (const_cast<int&>(counter_)++ == 0) {                                          // Case for first index
                try {                                                       // Check correct number of arguments
                    if (num_args + 1 !=  R) throw TensorInvalidArguments(num_args + 1, R);
                    offset_ = idx;
                } catch (TensorInvalidArguments& e) {
                    std::cerr << e.what() << std::endl;
                    return data_[0];
                }  
            } else {                                                        // Case for all other indecies
                offset_ += std::accumulate(dimensions_.begin()              , 
                                           dimensions_.end() - num_args - 1 ,
                                           1                                , 
                                           std::multiplies<int>()           ) * idx;
            }
            return this->operator()(indecies...);
        }  
};

}   // End namespace frnn

#endif
