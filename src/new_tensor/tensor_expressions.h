// ==========================================================================================================
//! @file   Header file for fastRNN tensor expression classes.
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

#ifndef _FRNN_TENSOR_EXPRESSIONS_
#define _FRNN_TENSOR_EXPRESSIONS_

#include "tensor_utils.h"
#include "../containers/tuple.h"
#include "../util/errors.h"

namespace frnn {

// ==========================================================================================================
//! @class      TensorExpression 
//! @brief      Define a Tesnor expression, for example a Tensor itself, addition, subtraction ... which can
//!             then be used by operators to make operations on Tensors look mathematic.
//! @tparam     T   The type of data the expression uses
//! @tparam     E   The type of exprssion (Tensor, TensorAddition etc...)
// ==========================================================================================================
template <typename T, typename E>
class TensorExpression {
public:
    /* ============================================= Typedefs ============================================= */
    typedef std::vector<T>                      container_type;
    typedef typename container_type::size_type  size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference  reference;
    /* ==================================================================================================== */
    
    // ======================================================================================================
    //! @brief     Returns the size of the expression
    //! @return    The size of the TensorExpression
    // ======================================================================================================
    size_type size() const { return static_cast<E const&>(*this).size(); }

    // ======================================================================================================
    //! @brief     Gets the sizes of the all the dimensions of the expression.
    //! @return    A constant reference to the dimension size vector of the expression 
    // ======================================================================================================
    const std::vector<size_type>& dimSizes() const { return static_cast<E const&>(*this).dimSizes(); }
    
    // ======================================================================================================
    //! @brief     Gets and element from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The value of the element at position i of the expression data.
    // ======================================================================================================
    value_type operator[](size_type i) const { return static_cast<E const&>(*this)[i]; }

    // ======================================================================================================
    //! @brief     Gets a reference to the Tensor expression.
    //! @return    A reference to the Tensor expression E.
    // ======================================================================================================
    operator E&() { return static_cast<E&>(*this); }

    // ======================================================================================================
    //! @brief     Gets a constant reference to the Tensor expression.
    //! @return    A constant reference to the Tensror expression E.
    // ======================================================================================================
    operator E const&() const   { return static_cast<const  E&>(*this); }
};

// ==========================================================================================================
//! @class      TensorDifference
//! @brief      Expression class for calculating the difference of two tensors.
//! @tparam     T       The type of the data used by the tensors.
//! @tparam     E1      The type of the  first expression for subtraction.
//! @tparam     E2      The type of the second expression for subtractio.n
// ==========================================================================================================
template <typename T, typename E1, typename E2>
class TensorDifference : public TensorExpression<T, TensorDifference<T,E1,E2>> {
public:
    /* ====================================== Typedefs ==================================================== */
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::container_type;
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::size_type;
    using typename TensorExpression<T, TensorDifference<T,E1,E2>>::value_type;
    /* ==================================================================================================== */
private:
    E1 const& _x;       //!< Reference to the first expression for subtraction
    E2 const& _y;       //!< Reference to the second expression for subtraction
public:
     // =====================================================================================================
     //! @brief         Constructs a TensorDifference class from the inputs.
     //! @tparam[in]    x   The first expression for subtraction
     //! @tparam[in]    y   The second expression for subtraction
     // =====================================================================================================
    TensorDifference(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) : _x(x), _y(y) 
    { 
        ASSERT(x.size(), ==, y.size());                         // Check total sizes
        for (int i = 0; i < x.dimSizes().size(); i++) {         // Check each dimension size
            ASSERT(x.dimSizes()[i], ==, y.dimSizes()[i]);
        }
    }

       
    // =====================================================================================================
    //! @brief     Gets the sizes of the all the dimensions of the expression.
    //! @return    A constant reference to the dimension size vector of the expression 
    // =====================================================================================================
    const std::vector<size_type>& dimSizes() const { return _x.dimSizes(); }
   
    // ======================================================================================================
    //! @brief     Gets and element from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The value of the element at position i of the expression data.
    // ======================================================================================================
    size_type size() const { return _x.size(); }
   
    // ======================================================================================================
    //! @brief     Subtracts two elements (one from each Tensor) from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The result of the subtraction of the Tensors.
    // ======================================================================================================
    value_type operator[](size_type i) const { return _x[i] - _y[i]; }
};      

/*
 * ==========================================================================================================
 * Class        : TensorAddition
 * Description  : Expression class for calculating the addition of two tensors
 * Params       : T     : The type of the data used by the tesors
 *              : E1    : The type of the  first expression for the addition
 *              : E2    : The type of the second expression for the addition
 * ==========================================================================================================
 */
template <typename T, typename E1, typename E2>
class TensorAddition : public TensorExpression<T, TensorAddition<T,E1,E2>> {
public:
    /* ====================================== Typedefs ==================================================== */
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::container_type;
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::size_type;
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::value_type;
    /* ==================================================================================================== */
private:
    E1 const& _x;       // Reference to first expression 
    E2 const& _y;       // Reference to second expression
public:
    /*
     * ======================================================================================================
     * Function     : TensorAddition
     * Description  : Constructor for the TensorAddition class, using the two expressions
     * Inputs       : x     : The first expression for addition
     *              : y     : The second expression for addition
     * ======================================================================================================
     */
    TensorAddition(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) : _x(x), _y(y) 
    { 
        ASSERT(x.size(), ==, y.size());                         // Check total sizes
        for (int i = 0; i < x.dimSizes().size(); i++) {         // Check each dimension size
            ASSERT(x.dimSizes()[i], ==, y.dimSizes()[i]);
        }
    }
   
   /*
    * =======================================================================================================
    * Function      : dimSizes
    * Description   : Returns the sizes of the dimensions of the expressions
    * Output        : A constant reference to the dimension sizes vector of the expression
    * =======================================================================================================
    */
    const std::vector<size_type>& dimSizes() const { return _x.dimSizes(); }
    
    /*
     * ======================================================================================================
     * Function     : size 
     * Description  : Returns the size of the result of the subtraction expression
     * ======================================================================================================
     */
    size_type size() const { return _x.size(); }
    
    /*
     * ======================================================================================================
     * Function     : operator[]
     * Description  : Overloaded access operator to return the addition of an element
     * Inputs       : i     : The index of the elements to add
     * Outputs      : The sum of the tensor elements at the specified index
     * ======================================================================================================
     */
    value_type operator[](size_type i) const { return _x[i] + _y[i]; }
};    

/*
 * ==========================================================================================================
 * Class        : TensorSlice
 * Description  : Class used to get a slice of a tensor
 * Params       : T     : The type of data used by the tensor
 *              : E     : The type of the expression to slice
 *              : Ts    : The types of the variables used to represent the dimensions to slice, this is used
 *                        to build a Tuple holding the dimensions to slice
 * ==========================================================================================================
 */
template <typename T, typename E, typename... Ts>
class TensorSlice : public TensorExpression<T, TensorSlice<T, E, Ts...>> {
public:
    /* ======================================== Typedefs ================================================== */
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::container_type;
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::size_type;
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::value_type;
    /* ==================================================================================================== */ 
private:
    E const&                        _x;                 // Reference to expression
    mutable Tuple<Ts...>            _slice_dims;        // Dimensions for the sliced tensor
    mutable std::vector<size_type>  _prev_slice_dims;   // Previous dimensions used for recursice map index
    mutable std::vector<size_type>  _slice_dim_sizes;   // Sizes of the dimensions for the slice
    mutable size_type               _index;             // Index of an element in the tensor to be sliced
    mutable size_type               _offset;            // Offset in the old tensor due to mapping to dimension 0
    size_type                       _slice_size;        // Size of the data of the slice
public:        
    /*
     * ======================================================================================================
     * Function         : Construct for the TensorSlice class
     * Description      : Initializes member variables and build the mapping for the new tensor from the
     *                    tensor which is being sliced
     * Inputs           : x         : The TensorExpression which is being sliced
     *                  : dims      : The dimension of x which are being sliced 
     * ======================================================================================================
     */
    TensorSlice(TensorExpression<T, E> const& x, Tuple<Ts...> slice_dims)
    : _x(x), _index(0), _offset(0), _prev_slice_dims(0), 
      _slice_size(mapDimensions()), _slice_dims(slice_dims)
    {}
   
    /*
     * ======================================================================================================
     * Function     : size 
     * Description  : Returns the size of sliced tensor
     *==== ==================================================================================================
     */
    size_type size() const { return _slice_size; }
    
   /*
    * =======================================================================================================
    * Function      : dimSizes
    * Description   : Returns the sizes of the dimensions of the expressions
    * Output        : A constant reference to the dimension sizes vector of the expression
    * ----===================================================================================================
    */
    const std::vector<size_type>& dimSizes() const { return _slice_dim_sizes; }

    /*
     * ======================================================================================================
     * Function     : operator[]
     * Description  : Overloaded access operator to return an element of the new tensor, using the mapping
     *                to get the value from the old tensor 
     * Inputs       : i     : The index of the element to get
     * Outputs      : The tensor element at the specified index
     * ======================================================================================================
     */
    value_type operator[](size_type i) const { return _x[mapIndex(i)]; }

    /*
     * ======================================================================================================
     * Function     : mapDimensions (general case)
     * Description  : Adds the size of a dimension from an old tensor to the sliced tensor dimension sizes
     *                vector
     * Params       : i     : The iteration of the mapping function
     * ======================================================================================================
     */
    template <size_type i = 0>
    typename std::enable_if<i != (sizeof...(Ts) - 1), size_type>::type mapDimensions() const
    {
        _slice_dim_sizes.push_back(_x.size(get<i>(_slice_dims)()));         // Add dimension i size
        return ( _x.size(get<i>(_slice_dims)())   *         // Get size of dimension i from old tensor
                 mapDimensions<i + 1>()          );         // Multiply with remaining dimensions
    }

    /*
     * ======================================================================================================
     * Function     : mapDimensions (terminating case)
     * Description  : Adds the size of a dimension from an old tensor to the sliced tensor dimension sizes
     *                vector
     * Params       : i     : The iteration of the mapping function
     * ======================================================================================================
     */
    template <size_type i>
    typename std::enable_if<i == (sizeof...(Ts) - 1), size_type>::type mapDimensions() const 
    {
        _slice_dim_sizes.push_back(_x.size(get<i>(_slice_dims)()));         // Add last dimension size 
        return _x.size(get<i>(_slice_dims)());              // Get size of last dimension old tensor   
    }
    
    /*
     * ======================================================================================================
     * Function     : mapIndex (non terminating case)
     * Description  : Takes the index of an element in a tensor which is a slice of another tensor, and 
     *                maps the index of the element in the new, sliced tensor, to the index in the tensor 
     *                which is being sliced.
     * Inputs       : idx       : The index of the element in the new, sliced tensor to map to the old
     *                            tensor
     * Outputs      : Calls itself to determine the mapping of the index to the other dimensions of the
     *                tensor which is being sliced
     * Params       : i         : The iteration of the function - which number element in the list of slice 
     *                            dimensions is being mapped to the sliced tensor
     * ======================================================================================================
     */ 
    template <size_type i = 0>
    typename std::enable_if<i != (sizeof...(Ts) - 1), size_type>::type mapIndex(size_type idx) const 
    {
        size_type mapped_dim = 0, dim = 0, dim_offset = 0;

        dim         = get<i>(_slice_dims)();                                // Size of dim i
        dim_offset  = std::accumulate(_x.dimSizes().begin()           ,     // Index offset of i in 
                                      _x.dimSizes().begin() + dim     ,     // original tensors memory
                                      1                               ,
                                      std::multiplies<size_type>()    );
        
        tensor::DimensionMapper<i> mapper;                                  // Get index in dimension i of
        mapped_dim = mapper(idx, _x.dimSizes()[dim]);                       // idx in tensor being sliced
                
        dim == 0  ? _index   = mapped_dim
                  : _offset += dim_offset * mapped_dim;
        
        _prev_slice_dims.push_back(dim);
        return mapIndex<i + 1>(idx);                        // Continue until all dimensions finished
    }
    
    /*
     * ======================================================================================================
     * Function     : mapIndex (terminating case)
     * Description  : Takes the index of an element in a tensor which is a slice of another tensor, and 
     *                maps the index of the element in the new, sliced tensor, to the index in the tensor 
     *                which is being sliced.
     * Inputs       : idx       : The index of the element in the new, sliced tensor to map to the old
     *                            tensor
     * Outputs      : The total offset of the index idx in the memory of the tensor to be sliced, so it is
     *                determining the index of idx in the sliced tensor
     * Params       : i         : The iteration of the function - which number element in the Tuple of 
     *                            slice dimensions is beig mapped
     * ======================================================================================================
     */
    template <size_type i>
    typename std::enable_if<i == (sizeof...(Ts) - 1), size_type>::type mapIndex(size_type idx) const 
    {
        size_type mapped_dim = 0, dim = 0, dim_offset = 0;
        
        dim         = get<i>(_slice_dims)();                                // Size of dim i 
        dim_offset  = std::accumulate(_x.dimSizes().begin()           ,     // Index offset of i in 
                                      _x.dimSizes().begin() + dim     ,     // original tensors memory
                                      1                               ,
                                      std::multiplies<size_type>()    );
        
        tensor::DimensionMapper<i> mapper;                                  // Get index of dimension i of
        mapped_dim = mapper(idx, _x.dimSizes()[dim], _prev_slice_dims);     // idx in tensor being sliced
                
        dim == 0  ? _index   = mapped_dim
                  : _offset += dim_offset * mapped_dim;
        
        size_type total_offset = _index + _offset;                          // Calculate final offset
        _prev_slice_dims.clear();                                           // Reset all class vars
        _index = 0; _offset = 0;
    
        return total_offset;
    }
};

}       // End namespace frnn

/* =========================== Global Operator Overloads using Tensor Expressions ========================= */

namespace {
    
/*
 * ==========================================================================================================
 * Function     : operator-
 * Description  : Overloaded - operator to subtract two tensor expressions, which improves readability of the
 *                subtraction of two tensors, and improves performance through the expression templates
 * Inputs       : x     : The first expression for the subtraction (this could be a tensor addition, a tensor
 *                        itself etc...)
 *              : y     : The second expression for the subtraction
 * Outputs      : The result of the subtraction of the expressions
 * Params       : T     : The type of data used by the expressions
 *              : E1    : The type of the first expression
 *              : E2    : The type of the second expression
 * ==========================================================================================================
 */
template <typename T, typename E1, typename E2>
frnn::TensorDifference<T, E1 ,E2> const operator-(frnn::TensorExpression<T, E1> const& x, 
                                                  frnn::TensorExpression<T, E2> const& y)    
{
    return frnn::TensorDifference<T, E1, E2>(x, y);
}

/*
 * ==========================================================================================================
 * Function     : operator+
 * Description  : Overloaded + operator to add two tensor expressions, which improves readability of the
 *                addition of two tensors, and improves performance through the expression templates
 * Inputs       : x     : The first expression for the addition (this could be a tensor addition, a tensor
 *                        itself etc...)
 *              : y     : The second expression for the addition
 * Outputs      : The result of the addition of the expressions
 * Params       : T     : The type of data used by the expressions
 *              : E1    : The type of the first expression
 *              : E2    : The type of the second expression
 * ==========================================================================================================
 */
template <typename T, typename E1, typename E2>
frnn::TensorAddition<T, E1 ,E2> const operator+(frnn::TensorExpression<T, E1> const& x, 
                                                frnn::TensorExpression<T, E2> const& y)    
{
    return frnn::TensorAddition<T, E1, E2>(x, y);
}

} // End global namespace

#endif
