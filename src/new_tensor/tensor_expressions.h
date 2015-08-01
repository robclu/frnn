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
#include "tensor_utils.h"

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
        * Function      : dimSizes
        * 
        * Description   : Returns the sizes of the dimensions of the expressions
        * 
        * Output        : A constant reference to the dimension sizes vector of the expression
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

/*
 * ==========================================================================================================
 * Class        : TensorSlice
 * 
 * Description  : Class used to get a slice of a tensor
 * 
 * Params       : T     : The type of data used by the tensor
 *              : E     : The type of the expression to slice
 *              : Ds    : The dimensions of the expression E which must be sliced to make this tensor
 * ==========================================================================================================
 */
template <typename T, typename E>
class TensorSlice : public TensorExpression<T, TensorSlice<T, E>>
{
    public:
        /* ==================================== Typedefs ================================================== */
        using typename TensorExpression<T, TensorSlice<T,E>>::container_type;
        using typename TensorExpression<T, TensorSlice<T,E>>::size_type;
        using typename TensorExpression<T, TensorSlice<T,E>>::value_type;
        /* ================================================================================================ */ 
    private:
        E const&                x_;                 // Reference to expression
        std::vector<size_type>  mapping_;           // Vector of new dimension mapping
        VariadicVector<size_type> maps;
    public:        
        /*
         * ==================================================================================================
         * Function         : Construct for the TensorSlice class
         * 
         * Description      : Initializes member variables and build the mapping for the new tensor from the
         *                    tensor which is being sliced
         *                    
         * Inputs           : x         : The TensorExpression which is being sliced
         *                  : dims      : The dimension of x which are being sliced 
         * ==================================================================================================
         */
        TensorSlice(TensorExpression<T, E> const& x, VariadicVector<size_t>&& dims)
        : x_(x), mapping_(0), maps(std::move(dims))
        {
            std::cout << "SIZE : " << maps.size() << " " << maps[0] << std::endl;
        }
        
       /*
        * ===================================================================================================
        * Function      : dimSizes
        * 
        * Description   : Returns the sizes of the dimensions of the expressions
        * 
        * Output        : A constant reference to the dimension sizes vector of the expression
        * ===================================================================================================
        */
        const std::vector<size_type>& dimSizes() const { return x_.dimSizes(); }
       
        /*
         * ==================================================================================================
         * Function         : buildMapping
         * 
         * Description      : Base case for creating a vector of dimension for slicing a tensor
         * 
         * Inputs           : dim       : The dimension to add to the mapping
         * 
         * Params           : D         : The type for the dim variable
         * ==================================================================================================
         */
        template <typename D>
        void buildMapping(D dim) { mapping_.push_back(dim); }
        
        /*
         * ==================================================================================================
         * Function         : buildMapping
         * 
         * Description      : all other cases for creating a vector of dimension for slicing a tensor
         * 
         * Inputs           : dim       : The dimension to add to the mapping
         *                  : dims      : The rest of the dims which must still be added to the mapping
         * 
         * Params           : D         : The type for the dim variable
         *                  : Dz        : The types for the rest of the dimensions
         * ==================================================================================================
         */ 
        template <typename D, typename... Dz>
        void buildMapping(D dim, Dz... dims) 
        {
            mapping_.push_back(dim);            // Add this dimension to the list
            buildMapping(dims...);              // Recurse until no dimensions left
        }
          
        /*
         * ==================================================================================================
         * Function     : mapIndex (non terminating cases)
         * 
         * Description  : Takes the index of an element in a tensor which is a slice of another tensor, and 
         *                maps the index of the element in the new, sliced tensor, to the index in the tensor 
         *                which is being sliced.
         *                
         * Inputs       : idx       : The index of the element in the new, sliced tensor to map to the old
         *                            tensor
         *                            
         * Outputs      : The value of the dimension dim in the old tesnor for index and the slice mapping
         *                given to the constructor. For example, if the mapping is from a Tensor[i,j,k] to a 
         *                Tensor[k,j], then if dim = j, for the element at index idx in the new tensor, the 
         *                mapping retuns the index in the j dimension of the original tensor
         * ==================================================================================================
         */
        size_type mapIndex(size_type idx) const 
        {
            size_type index = 0, offset = 0, dim = 0, mappedDim = 0, dimOffset = 0;
            std::vector<size_type> prevDimSizes;
            
            for (int i = 0; i < mapping_.size(); i++) {
                dim = mapping_[i];
            
                // Offset of element in original tensors memory for this dimension
                dimOffset = std::accumulate(x_.dimSizes().begin()         ,
                                            x_.dimSizes().begin() + dim   ,
                                            1                             ,
                                            std::multiplies<size_type>()  );
            
                if (i == 0) {
                    tensor::DimensionMapper<true> mapper;
                    mappedDim = mapper(idx, x_.dimSizes()[dim], prevDimSizes);
                } else {
                    tensor::DimensionMapper<false> mapper;
                    mappedDim = mapper(idx, x_.dimSizes()[dim], prevDimSizes);
                }
                    
                dim == 0 ? index   = mappedDim
                         : offset += dimOffset * mappedDim;
            
                prevDimSizes.push_back(x_.dimSizes()[dim]);
            }
           
            // Add offset due to all dimensions but the first, 
            // to the index in the first dimension
            return (index + offset);  
        }
        
        /*
         * ==================================================================================================
         * Function     : operator[]
         * 
         * Description  : Overloaded access operator to return an element of the new tensor, using the mapping
         *                to get the value from the old tensor 
         * 
         * Inputs       : i     : The index of the element to get
         * 
         * Outputs      : The tensor element at the specified index
         * ==================================================================================================
         */
        value_type operator[](size_type i) const { return x_[mapIndex(i)]; }
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
