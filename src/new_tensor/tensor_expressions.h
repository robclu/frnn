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
//! @tparam     E1      Expression to subtract from.
//! @tparam     E2      Expression to subtract with.
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
    E1 const& _x;       //!< First expression for subtraction
    E2 const& _y;       //!< Second expression for subtraction
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
       
    // ======================================================================================================
    //! @brief     Gets the sizes of the all the dimensions of the expression.
    //! @return    A constant reference to the dimension size vector of the expression 
    // ======================================================================================================
    const std::vector<size_type>& dimSizes() const { return _x.dimSizes(); }
   
    // ======================================================================================================
    //! @brief     Returns the size of the expression
    //! @return    The size of the TensorExpression
    // ====================================================================================================== 
    size_type size() const { return _x.size(); }
   
    // ======================================================================================================
    //! @brief     Subtracts two elements (one from each Tensor) from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The result of the subtraction of the Tensors.
    // ======================================================================================================
    value_type operator[](size_type i) const { return _x[i] - _y[i]; }
};      

// ==========================================================================================================
//! @class      TensorAddition
//! @brief      Expression class for calculating the addition of two tensors.
//! @tparam     T       Type of the data used by the tesors
//! @tparam     E1      Expression to add to.
//! @tparam     E2      Expression to add with.
// ==========================================================================================================
template <typename T, typename E1, typename E2>
class TensorAddition : public TensorExpression<T, TensorAddition<T,E1,E2>> {
public:
    /* ====================================== Typedefs ==================================================== */
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::container_type;
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::size_type;
    using typename TensorExpression<T, TensorAddition<T,E1,E2>>::value_type;
    /* ==================================================================================================== */
private:
    E1 const& _x;       //!< First expression for addition
    E2 const& _y;       //!< Second expression for addition
public:
     // =====================================================================================================
     //! @brief     Sets the expressions for addition and checks that they have the same ranks and dimension
     //!            sizes.
     //! @param[in] x       The first expression for addition.
     //! @param[in] y       The second expression for addition
     // =====================================================================================================
    TensorAddition(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) : _x(x), _y(y) 
    { 
        ASSERT(x.size(), ==, y.size());                         // Check total sizes
        for (int i = 0; i < x.dimSizes().size(); i++) {         // Check each dimension size
            ASSERT(x.dimSizes()[i], ==, y.dimSizes()[i]);
        }
    }
   
    // ======================================================================================================
    //! @brief     Gets the sizes of the all the dimensions of the expression.
    //! @return    A constant reference to the dimension size vector of the expression 
    // ======================================================================================================
    const std::vector<size_type>& dimSizes() const { return _x.dimSizes(); }
    
    // ======================================================================================================
    //! @brief     Returns the size of the expression.
    //! @return    The size of the TensorAddition.
    // ====================================================================================================== 
    size_type size() const { return _x.size(); }
    
    // ======================================================================================================
    //! @brief     Adds two elements (one from each Tensor) from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The result of the subtraction of the Tensors.
    // ======================================================================================================
    value_type operator[](size_type i) const { return _x[i] + _y[i]; }
};      

// ==========================================================================================================
//! @class      TensorMultiplier
//! @brief      Expression class for multiplying two Tensors
//! @tparam     T       Type of the data used by the Tensors.
//! @tparam     E1      Expression to multiply.
//! @tparam     E2      Expression to multilpy.
// ==========================================================================================================
template <typename T, typename E1, typename E2>
class TensorMultiplier : public TensorExpression<T, TensorMultiplier<T,E1,E2>> {
public:
    /* ====================================== Typedefs ==================================================== */
    using typename TensorExpression<T, TensorMultiplier<T,E1,E2>>::container_type;
    using typename TensorExpression<T, TensorMultiplier<T,E1,E2>>::size_type;
    using typename TensorExpression<T, TensorMultiplier<T,E1,E2>>::value_type;
    /* ==================================================================================================== */
private:
    E1 const& _x;       //!< First expression for multiplication
    E2 const& _y;       //!< Second expression for multiplication
public:
     // =====================================================================================================
     //! @brief     Sets the expressions for multiplication.
     //! @param[in] x       The first expression for addition.
     //! @param[in] y       The second expression for addition
     // =====================================================================================================
    TensorAddition(TensorExpression<T,E1> const& x, TensorExpression<T,E2> const& y) : _x(x), _y(y) 
    { 
        // Will need to check that the dimension checkout
    }
   
    // ======================================================================================================
    //! @brief     Gets the sizes of the all the dimensions of the expression.
    //! @return    A constant reference to the dimension size vector of the expression.
    // ======================================================================================================
    const std::vector<size_type>& dimSizes() const { return _x.dimSizes(); }
    
    // ======================================================================================================
    //! @brief     Returns the size of the expression.
    //! @return    The size of the TensorMultiplier.
    // ====================================================================================================== 
    size_type size() const { return _x.size(); }
    
    // ======================================================================================================
    //! @brief     Multiplies two elements (one from each Tensor) from the Tensor expression data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The result of the multiplication of the Tensors.
    // ======================================================================================================
    value_type operator[](size_type i) const { return _x[i] + _y[i]; }

private:
    // ======================================================================================================
    // ======================================================================================================
    template <typename... Ts>
    void buildReductionLists(Ts... dims) 
    {
        
    }
};    

// ==========================================================================================================
//! @class      TensorSlice
//! @brief      Expression class used to slice a TensorExpression by dimension.
//! @tparam     T   The type of data used by the tensor.
//! @tparam     E   The expression to slice.
//! @tparam     Ts  The types of the variables used to represent the dimensions to slice.
// ==========================================================================================================
template <typename T, typename E, typename... Ts>
class TensorSlice : public TensorExpression<T, TensorSlice<T, E, Ts...>> {
public:
    /* ======================================== Typedefs ================================================== */
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::container_type;
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::size_type;
    using typename TensorExpression<T, TensorSlice<T,E,Ts...>>::value_type;
    /* ==================================================================================================== */ 
private:
    // Note : These are mutable because this class is essentially a functor without state an is intended to 
    //        be used as a functor once. The mutable memebers are really just temporary variables which hold
    //        state during the dimension mapping computations, thus it is okay to modify them. However, the 
    //        function which modifies them need to be const to access _x, hence they are mutable.
    E const&                        _x;                 //!< Expression to slice
    mutable Tuple<Ts...>            _slice_dims;        //!< Dimensions of the sliced Expression
    mutable std::vector<size_type>  _prev_slice_dims;   //!< Dimensions used for iterative index mapping
    mutable std::vector<size_type>  _slice_dim_sizes;   //!< Sizes of the dimensions for the sliced Expression
    mutable size_type               _index;             //!< Mapped index from the slice to the Expression
    mutable size_type               _offset;            //!< Offset in the Expression due to its 0 dimension
    size_type                       _slice_size;        //!< Size (number of elements) of the slice
public:        
     // =====================================================================================================
     //! @brief     Initializes member variables, builds vector of dimensions of the Expression to slice, and 
     //!            determines the size of the slice.
     //! @param[in] x           The Expression to slice.
     //! @param[in] slice_dims  The dimension of Expression which make up the slice.
     // =====================================================================================================
    TensorSlice(TensorExpression<T, E> const& x, Tuple<Ts...> slice_dims)
    : _x(x), _index(0), _offset(0), _prev_slice_dims(0), 
      _slice_size(buildSliceDimSizes()), _slice_dims(slice_dims) 
    {}
  
    // ======================================================================================================
    //! @brief     Returns the size of the expression
    //! @return    The size of the TensorExpression
    // ======================================================================================================   
    size_type size() const { return _slice_size; }
    
    // ======================================================================================================
    //! @brief      Returns the sizes of each of the dimensions of the slice.
    //! @return     A constant reference to the dimension sizes vector of the slice.
    // ======================================================================================================
    const std::vector<size_type>& dimSizes() const { return _slice_dim_sizes; }

    // ======================================================================================================
    //! @brief     Gets an element from the Expression data which should be in position i of the slice's data.
    //! @param[in] i   The element in the expression which must be fetched.
    //! @return    The value of the element at position i of the expression data.
    // ======================================================================================================
    value_type operator[](size_type i) const { return _x[mapIndex(i)]; }
    
private:
    // =====================================================================================================
    //! @brief     Adds the size of a dimension from the Expression to the slice dimension sizes vector so 
    //!            that all dimension sizes of the slice are known. Case for all iterations but the last.
    //! @tparam    i   The iteration of the function.
    // =====================================================================================================
    template <size_type i = 0>
    typename std::enable_if<i != (sizeof...(Ts) - 1), size_type>::type buildSliceDimSizes() const
    {
        _slice_dim_sizes.push_back(_x.size(get<i>(_slice_dims)()));                 // Add dimension i's size
        return ( _x.size(get<i>(_slice_dims)()) *                   // Get size of dim i from the Expression
                 buildSliceDimSizes<i + 1>()    );                  // Multiply with the remaining dimensions
    }

    // =====================================================================================================
    //! @brief     Adds the size of a dimension from the Expression to the slice dimension sizes vector so 
    //!            that all dimensions sizes of the slice are known. Case for the last iteration.
    //! @tparam    i   The iteration of the function.
    // =====================================================================================================
    template <size_type i>
    typename std::enable_if<i == (sizeof...(Ts) - 1), size_type>::type buildSliceDimSizes() const 
    {
        _slice_dim_sizes.push_back(_x.size(get<i>(_slice_dims)()));                 // Add last dimension size 
        return _x.size(get<i>(_slice_dims)());                  // Get size of last dimension of the Expression
    }
    
    // =====================================================================================================
    //! @brief         Takes the index of an element in the slice, and maps the index to and element in the 
    //!                Expression being sliced. Case for all iterations but the last.
    //! @param[in]     idx     The index of the element in the slice.
    //! @return        The index of the element i in the slice, in the Expression's data variable.
    //! @tparam        i       The iteration of the function, essentially which element (in the vector of
    //!                slice dimensions) the offset in the Expression's is being determined.
    // =====================================================================================================
    template <size_type i = 0>
    typename std::enable_if<i != (sizeof...(Ts) - 1), size_type>::type mapIndex(size_type idx) const 
    {
        size_type mapped_dim = 0, dim = 0, dim_offset = 0;

        dim         = get<i>(_slice_dims)();                                        // Size of dim i
        dim_offset  = std::accumulate(_x.dimSizes().begin()           ,             // Index offset of i in 
                                      _x.dimSizes().begin() + dim     ,             // original tensors memory
                                      1                               ,
                                      std::multiplies<size_type>()    );
        
        tensor::DimensionMapper<i> mapper;                                      // Get index in dimension i of
        mapped_dim = mapper(idx, _x.dimSizes()[dim]);                           // idx in tensor being sliced
                
        dim == 0  ? _index   = mapped_dim
                  : _offset += dim_offset * mapped_dim;
        
        _prev_slice_dims.push_back(dim);
        return mapIndex<i + 1>(idx);                                // Continue until all dimensions finished
    }
    
    // =====================================================================================================
    //! @brief         Takes the index of an element in the slice, and maps the index to and element in the 
    //!                Expression being sliced. Case for the last iteration.
    //! @param[in]     idx     The index of the element in the slice.
    //! @return        The index of the element i in the slice, in the Expression's data variable.
    //! @tparam        i       The iteration of the function, essentially which element (in the vector of
    //!                slice dimensions) the offset in the Expression's is being determined.
    // =====================================================================================================
    template <size_type i>
    typename std::enable_if<i == (sizeof...(Ts) - 1), size_type>::type mapIndex(size_type idx) const 
    {
        size_type mapped_dim = 0, dim = 0, dim_offset = 0;
        
        dim         = get<i>(_slice_dims)();                                        // Size of dim i 
        dim_offset  = std::accumulate(_x.dimSizes().begin()           ,             // Index offset of i in 
                                      _x.dimSizes().begin() + dim     ,             // original tensors memory
                                      1                               ,
                                      std::multiplies<size_type>()    );
        
        tensor::DimensionMapper<i> mapper;                                      // Get index of dimension i of
        mapped_dim = mapper(idx, _x.dimSizes()[dim], _prev_slice_dims);         // idx in tensor being sliced
                
        dim == 0  ? _index   = mapped_dim
                  : _offset += dim_offset * mapped_dim;
        
        size_type total_offset = _index + _offset;                                  // Calculate final offset
        _prev_slice_dims.clear();                                                   // Reset all class vars
        _index = 0; _offset = 0;
    
        return total_offset;
    }
};

}       // End namespace frnn

/* =========================== Global Operator Overloads using Tensor Expressions ========================= */

namespace {
    
// ==========================================================================================================
//! @brief      Subtracts two TensorExpressions.
//!
//!             The expressions could be any type: Tensor, TensorSubtraction, TensorMultiplication ...
//! @param[in]  x   The TensorExpression to substract from.
//! @param[in]  y   The TensorExpression to subtract with. 
//! @return     The result of the subtraction of the two TensorExpressions.
//! @tparam     T   Type of data used by the expressions.
//! @tparam     E1  Type of the expression to subtract from.
//! @tparam     E2  Type of the expression to subtract with.
// ==========================================================================================================
template <typename T, typename E1, typename E2>
frnn::TensorDifference<T, E1 ,E2> const operator-(frnn::TensorExpression<T, E1> const& x, 
                                                  frnn::TensorExpression<T, E2> const& y)    
{
    return frnn::TensorDifference<T, E1, E2>(x, y);
}

// ==========================================================================================================
//! @brief      Adds two TensorExpressions. 
//!
//!             The expressions could be any Tensor type: Tensor, TensorAddition, TensorMultiplication ...
//! @param[in]  x   The TensorExpression to add to.
//! @param[in]  y   The TensorExpression to add with.
//! @return     The result of the addition of the two TensorExpressions.
//! @tparam     T   The type of data used by the expressions.
//! @tparam     E1  The type of the expression to add to.
//! @tparam     E2  The type of the expression to add with.
// ==========================================================================================================
template <typename T, typename E1, typename E2>
frnn::TensorAddition<T, E1 ,E2> const operator+(frnn::TensorExpression<T, E1> const& x, 
                                                frnn::TensorExpression<T, E2> const& y)    
{
    return frnn::TensorAddition<T, E1, E2>(x, y);
}

} // End global namespace

#endif
