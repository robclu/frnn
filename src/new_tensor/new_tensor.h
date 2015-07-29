/*
 *  Header file for fastRNN *NEW* tensor class.
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

#include "tensor_exception.h"
#include "../util/errors.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <initializer_list>
#include <numeric>

namespace frnn {

/*
 * ==========================================================================================================
 * Class        : TensorExpression 
 * 
 * Description  : Class used to define a tesnor expression, for example addition, subtraction ... which can
 *                then be used by the respective overloaded operator to provide fast operation on tensors 
 *                with elegant syntax
 *                
 * Params       : T     : The tpye to use for the expression
 *              : E     : The exprssion (TensorAddition etc...)
 * ==========================================================================================================
 */
template <typename T, typename E>
class TensorExpression {
public:
    /* =========================================== Typedefs =============================================== */
    typedef std::vector<T>                      container_type;
    typedef typename container_type::size_type  size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference  reference;
    /* ==================================================================================================== */
    
    /*
     * ======================================================================================================
     * Function     : size
     * 
     * Description  : Returns the size of the expression
     * 
     * Outputs      : The size of the tensor expression
     * ======================================================================================================
     */
    size_type size() const { return static_cast<E const&>(*this).size(); }
    
    /*
     * ======================================================================================================
     * Function     : operator[]
     * 
     * Description  : Operloaded access operator for getting an element of a tensor expression
     * 
     * Inputs       : i     : The element which must be accessed
     * 
     * Outputs      : The value of the element at position i
     * ======================================================================================================
     */
    value_type operator[](size_type i) const { return static_cast<E const&>(*this)[i]; }

    /*
     * ======================================================================================================
     * Function     : operator()
     * 
     * Description  : Returns a reference to the expression E
     * 
     * Outputs      : A (mutable) reference to the expression E, which is passed as a template to the class
     * ======================================================================================================
     */
    operator E&() { return static_cast<E&>(*this); }

    /*
     * ======================================================================================================
     * Function     : oeprator()
     * 
     * Description  : Returns a constant reference to the expression E
     * 
     * Outptus      : A constant (immutable) reference to the expression E, which is passed as a template to
     *                the class
     * ======================================================================================================
     */
    operator E const&() const   { return static_cast<const  E&>(*this); }
};

/*
 * ==========================================================================================================
 * Class        : Tensor
 * 
 * Description  : Tensor class for the fastRNN library. The class allows for a tensor of any dimension, thus
 *                providing great flexibility. For example, a 1D tensor is a vector or array, and a 2D tensor i
 *                is a matrix. 
 * ==========================================================================================================
 */
template <typename T, int R>
class Tensor : public TensorExpression<T, Tensor<T, R>> {
    using typename TensorExpression<T, Tensor<T,R>>::container_type;
    using typename TensorExpression<T, Tensor<T,R>>::size_type;
    using typename TensorExpression<T, Tensor<T,R>>::value_type;
    using typename TensorExpression<T, Tensor<T,R>>::reference;
 
    private:
        container_type      data_;
        std::vector<int>    dimensions_;
        int                 counter_;
        int                 offset_;
    public:
        reference   operator[](size_type i)         { return data_[i]; }
        value_type  operator[](size_type i) const   { return data_[i]; }
        size_type   size()                  const   { return data_.size(); }

        Tensor() : data_(0), dimensions_(0), counter_(0) {}
        
        // Construct Tensor for a given rank
        Tensor(std::initializer_list<int> dimensions) 
            : data_(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>())),
              counter_(0) {
                  assert(dimensions.size() == R); 
                  for (auto& element : dimensions) dimensions_.push_back(element);
            }
        
        // Consstructor from an expression
        template <typename E>
        Tensor(TensorExpression<T,E> const& tensor) : counter_(0) {
            E const& t = tensor;
            data_.resize(t.size());
            for (size_type i = 0; i != t.size(); ++i) {
                data_[i] = t[i];
            }
        }
        
        template <typename Index>
        T& operator() (Index idx) {
            try {                                                           // Check in range
                if (idx >= dimensions_[counter_]) {                         // counter +1 for 0 index offset 
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // in this line
                }
            } catch (TensorOutOfRange& e) {
                std::cerr << e.what() << std::endl;
                counter_ = 0;
                return data_[0];
            }
            offset_ += std::accumulate(dimensions_.begin()      , 
                                       dimensions_.end() - 1    ,           // Multiply all elements exepct last
                                       1                        ,           // Starting value for multiplication
                                       std::multiplies<int>()   ) * idx;    // Add offset due to idx for this dimension
            counter_ = 0;
            return data_[offset_];
        }
        
        template <typename Index, typename... Indecies>
        int operator()(Index idx, Indecies... indecies) {
            int num_args = sizeof...(Indecies);
            try {                                                           // Check index in range
                if (idx >= dimensions_[counter_]) {                         // counter + 1 for 0 index offset    
                    throw TensorOutOfRange(counter_ + 1, dimensions_[counter_], idx);   // in this line
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
            } else {
                offset_ += std::accumulate(dimensions_.begin()              , 
                                           dimensions_.end() - num_args - 1 ,
                                           1                                , 
                                           std::multiplies<int>()           ) * idx;
            }
            return this->operator()(indecies...);
        }   
};

template <typename T, typename E1, typename E2>
class TensorDifference : public TensorExpression<T, TensorDifference<T,E1,E2>> {
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::container_type;
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::size_type;
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::value_type;
    
    private:
        E1 const& x_;
        E2 const& y_;
    public:
        TensorDifference(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) :
            x_(x), y_(y) { 
                assert(x.size() == y.size());
            }
        size_type size() const { return x_.size(); }
        value_type operator[](size_type i) const { return x_[i] - y_[i]; }
};

/* ==================================== Operator Overloads ================================================ */

template <typename T, typename E1, typename E2>
TensorDifference<T,E1,E2> const operator-(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) {
    return TensorDifference<T,E1,E2>(x, y);
}

}
#endif
