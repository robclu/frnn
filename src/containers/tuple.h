// ==========================================================================================================
//! @file tuple.h
//!       Header file for the fastRNN Tuple class to hold and number of elements of any type. 
//!       A good tutorial on using variadic templates for this type of implementation is given at:
//!       http://eli.thegreenplace.net/2014/variadic-templates-in-c/
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
 *  =========================================================================================================
 */ 

#ifndef _FRNN_CONTAINERS_TUPLE_
#define _FRNN_CONTAINERS_TUPLE_

namespace frnn {

// ==========================================================================================================
//! @struct  Tuple 
//! @brief   Holds any number of elements of any type.
//! @details Usage : Tuple<type1, type2, ...> tuple(elem1 of type1, elem2 of type2, ...)
//! @tparam  Ts Types of all elements held in the Tuple.  
// ==========================================================================================================
template <typename... Ts> struct Tuple {};

// ==========================================================================================================
//! @struct  Tuple 
//! @tparam  T Type of the element to add to the Tuple. 
// ==========================================================================================================
template <typename T>
struct Tuple<T> {
public:
    T _back;            //!< Element of the Tuple
public:
    // ======================================================================================================
    //! @fn         Tuple
    //! @brief      Constuctor for a Tuple when there is only one element (base case of recursive 
    //!             construction).
    //! @param[in]  element The element of type T to add to the Tuple.
    // ======================================================================================================
    Tuple(T element) : _back(element) {}
};

// ==========================================================================================================
//! @struct  Tuple 
//! @tparam  T  Type of the element to add to the Tuple. 
//! @tparam  Ts The types of the elements to add to the Tuple on the following iterations of the recursive 
//!             construction.
// ==========================================================================================================
template <typename T, typename... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...> {
public: 
    T _back;            //!< Element of the Tuple
public:
     // =====================================================================================================
     //! @fn    Tuple 
     //! @brief Constructor for a Tuple when there is more than one element.
     //! @param[in] element     The element to add to the Tuple on this iteration of the recursive 
     //!                        construction.
     //! @param[in] elements    The rest of the elements to add to the Tuple on the remaining iterations.
     //  ====================================================================================================
    Tuple(T element, Ts... elements) : Tuple<Ts...>(elements...), _back(element) {}
};

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder 
 * Description  : Struct which holds the types of all the elements in a Tuple
 * Params       : i     : The index of the element for which the type must be stored
 *              : T     : The type of the element
 * ==========================================================================================================
 */
template <size_t i, typename T> struct TupleElementTypeHolder;

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder
 * Description  : Determines the type of the 0 element in the tuple
 * Params       : T     : The type of the 0 element in the Tuple
 *              : Ts    : The types of the rest of the elements in the Tuple
 * ==========================================================================================================
 */
template <typename T, typename... Ts>
struct TupleElementTypeHolder<0, Tuple<T, Ts...>> {
    typedef T type;         
};

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder
 * Description  : Determines the types of all elements in the Tuple but the first
 * Params       : i     : The index of the element in the Tuple to store the tye of 
 *              : T     : The type of the ith element in the Tuple
 *              : Ts    : The types of the rest of the elements in the Tuple
 * ==========================================================================================================
 */
template <size_t i, typename T, typename... Ts>
struct TupleElementTypeHolder<i, Tuple<T, Ts...>> {
    typedef typename TupleElementTypeHolder<i - 1, Tuple<Ts...>>::type type;  
};

/*
 * ==========================================================================================================
 * Function     : get
 * Description  : Gets the 0th element in a Tuple
 * Inputs       : tuple     :The Tuple to get the 0th element from
 * Params       : i         : The index of the element in the Tuple to get
 *              : Ts        : The types of the elements in the Tuple
 * ==========================================================================================================
 */
template <size_t i, typename... Ts>
typename std::enable_if<i == 0, typename TupleElementTypeHolder<0, Tuple<Ts...>>::type&>::type 
get(Tuple<Ts...>& tuple) 
{
    return tuple._back;
}

/*
 * ==========================================================================================================
 * Function     : get
 * Description  : Gets the element i of the tuple
 * Inputs       : tuple     : The Tuple to get the ith element from 
 * Params       : i         : The index of the element in the Tuple to get
 *              : T         : The type of the element i in the Tuple
 *              : Ts        : The test of the types of the elements in the Tuple
 * ==========================================================================================================
 */
template <size_t i, typename T, typename... Ts>
typename std::enable_if<i != 0, typename TupleElementTypeHolder<i, Tuple<T, Ts...>>::type&>::type
get(Tuple<T, Ts...>& tuple) 
{
    Tuple<Ts...>& tupleBase = tuple;
    return get<i - 1>(tupleBase);       
}

namespace tuple {
    
template <typename... Ts>
size_t size(Tuple<Ts...>& tuple) { return sizeof...(Ts); };

}

}       // End namespace frnn

#endif
