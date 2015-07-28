/*
 *  Header file for frnn Accessor class.
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

#ifndef _FRNN_TENSOR_ACCESSOR_
#define _FRNN_TENSOR_ACCESSOR_

#include <vector>
#include <cassert>
#include <initializer_list>
#include <numeric>

namespace frnn {

template <typename E>
class AccessorExpression {
    public:
        typedef std::vector<int>                    container_type;
        typedef typename container_type::size_type  size_type;
        typedef typename container_type::value_type value_type;
        typedef typename container_type::reference  reference;
    
        size_type   size()                  const { return static_cast<E const&>(*this).size(); }
        value_type  operator[](size_type i) const { return static_cast<E const&>(*this)[i];     }
        
        operator E&()               { return static_cast<       E&>(*this); }
        operator E const&() const   { return static_cast<const  E&>(*this); }
};
        
class Accessor : public AccessorExpression<Accessor> {
    public:
        using typename AccessorExpression<Accessor>::container_type;
        using typename AccessorExpression<Accessor>::size_type;
        using typename AccessorExpression<Accessor>::value_type;
        using typename AccessorExpression<Accessor>::reference;
    public:
        container_type values_;
    public:   
        reference   operator[](size_type i)         { return values_[i];     }
        value_type  operator[](size_type i) const   { return values_[i];     }
        size_type   size()                  const   { return values_.size(); }
        
        // Construct from initializer list
        Accessor(const std::initializer_list<int>& values) 
            : values_(values) {}
       
        // Construct from random arguments 
        template <typename... Args>
        explicit Accessor(Args&&... args) 
            : values_(std::forward<Args>(args)...) {};
        
        // Construct from expression
        template <typename E>
        Accessor(AccessorExpression<E> const& accessor) {
            E const& a = accessor;
            values_.resize(a.size());
            for (size_type i = 0; i != a.size(); ++i) {
                values_[i] = a[i];
            }
        }
       
        int sum() {
            return std::accumulate(values_.begin(), values_.end(), 0);
        }
        // Return sum of 
};

template <typename E1, typename E2>
class AccessorMultiply : public AccessorExpression<AccessorMultiply<E1,E2>> {
    public:
        using typename AccessorExpression<AccessorMultiply<E1,E2>>::container_type;
        using typename AccessorExpression<AccessorMultiply<E1,E2>>::size_type;
        using typename AccessorExpression<AccessorMultiply<E1,E2>>::value_type; 
    private:
        E1 const& x_;
        E2 const& y_;
    public:
        AccessorMultiply(AccessorExpression<E1> const& x, AccessorExpression<E2> const& y)
            : x_(x), y_(y) {
                assert(x.size() == y.size());
            }
        size_type size() const { return x_.size(); }
        value_type operator[](size_type i) const { return x_[i] * y_[i]; }
};

/* ================================= Operator Overloads =================================================== */
template <typename E1, typename E2>
AccessorMultiply<E1, E2> const operator*(AccessorExpression<E1> const& x, AccessorExpression<E2> const& y) {
    return AccessorMultiply<E1,E2>(x, y);
}

}

#endif
