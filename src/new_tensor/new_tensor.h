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

#include <vector>
#include <iostream>
#include <cassert>
#include <initializer_list>

namespace frnn {
    
inline int product(std::initializer_list<int>& list) {
    int elementProduct = 1;
    for (auto& element : list ) elementProduct *= element;
    return elementProduct;
}

int offset() { return 0; }

template <typename... D>
int offset(std::vector<int>& dimVect, int counter, int first, D... dims) {
    return first * dimVect[counter] + offset(dimVect, counter + 1, dims...);
}       

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
    public:
        reference   operator[](size_type i)         { return data_[i]; }
        value_type  operator[](size_type i) const   { return data_[i]; }
        size_type   size()                  const   { return data_.size(); }

        Tensor() : data_(0), dimensions_(0) {}
        
        // Construct Tensor for a given rank
        Tensor(std::initializer_list<int> dimensions) 
            : data_(product(dimensions)) {   
                 assert(dimensions.size() == R ); 
                for (auto& element : dimensions) dimensions_.push_back(element);
            }
        
        // Consstructor from and expression
        template <typename E>
        Tensor(TensorExpression<T,E> const& tensor) {
            E const& t = tensor;
            data_.resize(t.size());
            for (size_type i = 0; i != t.size(); ++i) {
                data_[i] = t[i];
            }
        }
        
        template <typename... D>
        T& operator() (D... dims) {}
            
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
