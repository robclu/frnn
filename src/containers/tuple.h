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
//! @tparam  Ts     The types of the elements to be stored in teh Tuple.
// ==========================================================================================================
template <typename... Ts> struct Tuple {};

// ==========================================================================================================
//! @struct  Tuple 
// ==========================================================================================================
template <typename T>
struct Tuple<T> {
public:
    T _back;            //!< Element of the Tuple
public:
    // ======================================================================================================
    //! @brief      Constuctor for a Tuple when there is only one element (base case of recursive 
    //!             construction).
    //! @param[in]  element     The element to add to the Tuple.
    //! @tparam     T           Type of the element to add to the Tuple.
    // ======================================================================================================
    Tuple(T element) : _back(element) {}
};

// ==========================================================================================================
//! @struct  Tuple 
// ==========================================================================================================
template <typename T, typename... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...> {
public: 
    T _back;            //!< Element of the Tuple
public:
     // =====================================================================================================
     //! @brief     Constructor for a Tuple when there is more than one element.
     //! @param[in] element     The element to add to the Tuple on this iteration of the recursive 
     //!            construction.
     //! @param[in] elements    The rest of the elements to add to the Tuple on the remaining iterations.
     //! @tparam    T           Type of the element to add to the Tuple. 
     //! @tparam    Ts          The types of the elements to add to the Tuple on the following iterations of 
     //!            the recursive construction.
     //  ====================================================================================================
    Tuple(T element, Ts... elements) : Tuple<Ts...>(elements...), _back(element) {}
};

// ==========================================================================================================
//! @struct     TupleElementTypeHolder
// ==========================================================================================================
template <size_t i, typename T> struct TupleElementTypeHolder;

// ==========================================================================================================
//! @struct     TupleElementTypeHolder
// ==========================================================================================================
template <typename T, typename... Ts>
struct TupleElementTypeHolder<0, Tuple<T, Ts...>> {
    typedef T type;         
};

// ==========================================================================================================
//! @struct     TupleElementTypeHolder
//! @brief      Defines the types of each of the elements in a Tuple.
//! @tparam i   Index of the element in the Tuple for which the type must be declared. 
//! @tparam T   Type of the element at position i in Tuple./
//! @tparam Ts  Types of the elements at positions != i in the Tuple. 
// ==========================================================================================================
template <size_t i, typename T, typename... Ts>
struct TupleElementTypeHolder<i, Tuple<T, Ts...>> {
    typedef typename TupleElementTypeHolder<i - 1, Tuple<Ts...>>::type type;  
};

// ==========================================================================================================
//! @brief      Gets the 0th element in a Tuple. Enabled if i is equal to 0.
//! @param[in]  tuple   The Tuple to get the element from.
//! @tparam     i       The index of the element to get.
//! @tparam     Ts      The types of all elements in the Tuple.
// ==========================================================================================================
template <size_t i, typename... Ts>
typename std::enable_if<i == 0, typename TupleElementTypeHolder<0, Tuple<Ts...>>::type&>::type 
get(Tuple<Ts...>& tuple) 
{
    return tuple._back;
}

// ==========================================================================================================
//! @brief      Gets the element at position i in the Tuple. Enabled if i is not equal to 0.
//! @param[in]  tuple   The Tuple to get the element from.
//! @tparam     i       The index of the element in the Tuple.
//! @tparam     T       The type of the element at position i.
//! @tparam     Ts      The types of all the elements in the Tuple.
// ==========================================================================================================
template <size_t i, typename T, typename... Ts>
typename std::enable_if<i != 0, typename TupleElementTypeHolder<i, Tuple<T, Ts...>>::type&>::type
get(Tuple<T, Ts...>& tuple) 
{
    Tuple<Ts...>& tupleBase = tuple;
    return get<i - 1>(tupleBase);       
}

namespace tuple {
    
// ==========================================================================================================
//! @brief      Gets the size of a Tuple
//! @param[in]  tuple   The Tuple to get the size of.
//! @tparam     Ts      The types of the elements in the Tuple.
//! @return     The size of the Tuple tuple.
// ========================================================================================================== 
template <typename... Ts>
size_t size(Tuple<Ts...>& tuple) { return sizeof...(Ts); }

}

}       // End namespace frnn

#endif
