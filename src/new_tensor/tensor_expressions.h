/*
 *  Header file for fastRNN tensor expression classes.
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

#ifndef _FRNN_TENSOR_EXPRESSIONS_
#define _FRNN_TENSOR_EXPRESSIONS_

#include "../util/errors.h"

#include <vector>

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
 *              : E     : The type of exprssion (Tensor, TensorAddition etc...)
 * ==========================================================================================================
 */
template <typename T, typename E>
class TensorExpression 
{
    public:
        /* =========================================== Typedefs =========================================== */
        typedef std::vector<T>                      container_type;
        typedef typename container_type::size_type  size_type;
        typedef typename container_type::value_type value_type;
        typedef typename container_type::reference  reference;
        /* ================================================================================================ */
        
        /*
         * ==================================================================================================
         * Function     : size
         * 
         * Description  : Returns the size of the expression
         * 
         * Outputs      : The size of the tensor expression
         * ==================================================================================================
         */
        size_type size() const { return static_cast<E const&>(*this).size(); }

        /*
         * ==================================================================================================
         * Function     : dimSizes
         * 
         * Description  : Gets the sizes of the dimensions of the expression E
         * 
         * Output       : A constant reference to the dimension size vector of the expression 
         * ==================================================================================================
         */
        const std::vector<size_type>& dimSizes() const { return static_cast<E const&>(*this).dimSizes(); }
        
        /*
         * ==================================================================================================
         * Function     : operator[]
         * 
         * Description  : Operloaded access operator for getting an element of a tensor expression
         * 
         * Inputs       : i     : The element which must be accessed
         * 
         * Outputs      : The value of the element at position i
         * ==================================================================================================
         */
        value_type operator[](size_type i) const { return static_cast<E const&>(*this)[i]; }

        /*
         * ==================================================================================================
         * Function     : operator()
         * 
         * Description  : Returns a reference to the expression E
         * 
         * Outputs      : A (mutable) reference to the expression E, which is passed as a template to the 
         *                class
         * ==================================================================================================
         */
        operator E&() { return static_cast<E&>(*this); }

        /*
         * ==================================================================================================
         * Function     : oeprator()
         * 
         * Description  : Returns a constant reference to the expression E
         * 
         * Outptus      : A constant (immutable) reference to the expression E, which is passed as a 
         *                template to the class
         * ==================================================================================================
         */
        operator E const&() const   { return static_cast<const  E&>(*this); }
};

/*
 * ==========================================================================================================
 * Class        : TensorDifference
 * 
 * Description  : Expression class for calculating the difference of two tensors
 * 
 * Params       : T     : The type of the data used by the tesors
 *              : E1    : The type of the  first expression for subtraction
 *              : E2    : The type of the second expression for subtraction
 * ==========================================================================================================
 */
template <typename T, typename E1, typename E2>
class TensorDifference : public TensorExpression<T, TensorDifference<T,E1,E2>> 
{
    public:
        /* ==================================== Typedefs ================================================== */
        using typename TensorExpression<T, TensorDifference<T,E1,E2>>::container_type;
        using typename TensorExpression<T, TensorDifference<T,E1,E2>>::size_type;
        using typename TensorExpression<T, TensorDifference<T,E1,E2>>::value_type;
        /* ================================================================================================ */
    private:
        E1 const& x_;       // Reference to first expression 
        E2 const& y_;       // Reference to second expression
    public:
        /*
         * ==================================================================================================
         * Function     : TensorDifference 
         * 
         * Description  : Constructor for the TensorDifference class, using the two expressions
         * 
         * Inputs       : x     : The first expression for subtraction
         *              : y     : The second expression for subtraction
         * ==================================================================================================
         */
        TensorDifference(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) : x_(x), y_(y) 
        { 
            ASSERT(x.size(), ==, y.size());                         // Check total sizes
            for (int i = 0; i < x.dimSizes().size(); i++) {        // Check each dimension size
                ASSERT(x.dimSizes()[i], ==, y.dimSizes()[i]);
            }
        }
       
       /*
        * ===================================================================================================
        */
        const std::vector<size_type>& dimSizes() const { return x_.dimSizes(); }
        
        /*
         * ==================================================================================================
         * Function     : size 
         * 
         * Description  : Returns the size of the result of the subtraction expression
         * ==================================================================================================
         */
        size_type size() const { return x_.size(); }
        
        /*
         * ==================================================================================================
         * Function     : operator[]
         * 
         * Description  : Overloaded access operator to return an element 
         * 
         * Inputs       : i     : The index of the element to get
         * 
         * Outputs      : The tensor element at the specified index
         * ==================================================================================================
         */
        value_type operator[](size_type i) const { return x_[i] - y_[i]; }
};      

} // End namespace frnn

/* =========================== Global Operator Overloads using Tensor Expressions ========================= */

namespace {
    
/*
 * ==========================================================================================================
 * Function     : operator-
 * 
 * Description  : Overloaded - operator to subtract two tensor expressions, which improves readability of the
 *                subtraction of two tensors, and improves performance through the expression templates
 *               
 * Inputs       : x     : The first expression for the subtraction (this could be a tensor addition, a tensor
 *                        itself etc...)
 *              : y     : The second expression for the subtraction
 *              
 * Outputs      : The result of the subtraction of the expressions
 * 
 * Params       : T     : The type of data used by the expressions
 *              : E1    : The type of the first expression
 *              : E2    : The type of the second expression
 * ==========================================================================================================
 */
template <typename T, typename E1, typename E2>
frnn::TensorDifference<T,E1,E2> const operator-(frnn::TensorExpression<T,E1> const& x, 
                                                frnn::TensorExpression<T,E2> const& y)    
{
    return frnn::TensorDifference<T,E1,E2>(x, y);
}

} // End global namespace

#endif
