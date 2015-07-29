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

#include "../util/errors.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <initializer_list>
#include <numeric>

namespace frnn {
  
template <typename T, typename E>
class TensorExpression {
public:
    typedef std::vector<T>                      container_type;
    typedef typename container_type::size_type  size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference  reference;
    
    size_type   size()                  const { return static_cast<E const&>(*this).size(); }
    value_type  operator[](size_type i) const { return static_cast<E const&>(*this)[i];     }
    
    operator E&()               { return static_cast<       E&>(*this); }
    operator E const&() const   { return static_cast<const  E&>(*this); }
};

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
        int operator() (Index idx) {
            assert(idx < dimensions_[counter_]);                            // Check in range
            offset_ += std::accumulate(dimensions_.begin()      , 
                                       dimensions_.end() - 1    ,           // Multiply all elements exepct last
                                       1                        ,           // Starting value for multiplication
                                       std::multiplies<int>()   ) * idx;    // Add offset due to idx for this dimension
            counter_ = 0;
            return offset_;
        }
        
        template <typename Index, typename... Indecies>
        int operator()(Index idx, Indecies... indecies) {
            int num_args = sizeof...(Indecies);
            if (counter_++ == 0) {  
                ASSERT(num_args + 1, ==,  R);
                offset_ = idx;
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
