/*
 *  Header file for fastRNN tuple container.
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

/*
 *  For an article on this see : http://eli.thegreenplace.net/2014/variadic-templates-in-c/
 */

#ifndef _FRNN_CONTAINERS_TUPLE_
#define _FRNN_CONTAINERS_TUPLE_

namespace frnn {

/*
 * ==========================================================================================================
 * Struct       : Tuple 
 * 
 * Description  : Tuple class which can be created using any types and any number of elements
 * 
 * Params       : Ts        : The types of the elements in the Tuple
 * ==========================================================================================================
 */
template <typename... Ts> struct Tuple {};

template <typename T>
struct Tuple<T> 
{
    /*
     * ======================================================================================================
     * Function         : Tuple 
     * 
     * Description      : Constuctor for a Tuple when there is only one element (base case of recursive
     *                    construction
     * 
     * Inputs           : element   : The element to add to the tuple
     * ======================================================================================================
     */
    Tuple(T element) : back_(element) {}
    
    T back_;        // The last element of the Tuple
};

template <typename T, typename... Ts>
struct Tuple<T, Ts...> : Tuple<Ts...> 
{
    /*
     * ======================================================================================================
     * Function         : Tuple 
     * 
     * Description      : Constructor for a Tuple when there is more than one element
     * 
     * Inputs           : element   : The element to add to the tuple on this iteration of the construction
     *                  : elements  : The rest of the elements to add to the tuple on the remaining iterations 
     *                                of the construction
     *
     * Params           : T         : The type of the element to add
     *                  : Ts        : The types of the rest of the elemnts to add
     * ======================================================================================================
     */
    Tuple(T element, Ts... elements) : Tuple<Ts...>(elements...), back_(element) {}
    
    T back_;        // The last element at the current iteration of the construction
};

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder 
 * 
 * Description  : Struct which holds the types of all the elements in a Tuple
 * 
 * Params       : i     : The index of the element for which the type must be stored
 *              : T     : The type of the element
 * ==========================================================================================================
 */
template <size_t i, typename T> struct TupleElementTypeHolder;

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder
 * 
 * Description  : Determines the type of the 0 element in the tuple
 * 
 * Params       : T     : The type of the 0 element in the Tuple
 *              : Ts    : The types of the rest of the elements in the Tuple
 * ==========================================================================================================
 */
template <typename T, typename... Ts>
struct TupleElementTypeHolder<0, Tuple<T, Ts...>>
{ 
    typedef T type;
};

/*
 * ==========================================================================================================
 * Struct       : TupleElementTypeHolder
 * 
 * Description  : Determines the types of all elements in the Tuple but the first
 * 
 * Params       : i     : The index of the element in the Tuple to store the tye of 
 *              : T     : The type of the ith element in the Tuple
 *              : Ts    : The types of the rest of the elements in the Tuple
 * ==========================================================================================================
 */
template <size_t i, typename T, typename... Ts>
struct TupleElementTypeHolder<i, Tuple<T, Ts...>> 
{
    typedef typename TupleElementTypeHolder<i - 1, Tuple<Ts...>>::type type;  
};

/*
 * ==========================================================================================================
 * Function     : get
 * 
 * Description  : Gets the 0th element in a Tuple
 * 
 * Inputs       : tuple     :The Tuple to get the 0th element from
 * 
 * Params       : i         : The index of the element in the Tuple to get
 *              : Ts        : The types of the elements in the Tuple
 * ==========================================================================================================
 */
template <size_t i, typename... Ts>
typename std::enable_if<i == 0, typename TupleElementTypeHolder<0, Tuple<Ts...>>::type&>::type 
get(Tuple<Ts...>& tuple) 
{
    return tuple.back_;
}

/*
 * ==========================================================================================================
 * Function     : get
 *  
 * Description  : Gets the element i of the tuple
 * 
 * Inputs       : tuple     : The Tuple to get the ith element from 
 * 
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

}       // End namespace frnn

#endif
