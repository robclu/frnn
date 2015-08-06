// ==========================================================================================================
//! @file variadic_map.h
//!       Header file for the fastRNN VariadicMap class to create an unordered map from template variadic 
//!       template parameters.
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

#ifndef _FRNN_CONTAINERS_VARIADIC_MAP_
#define _FRNN_CONTAINERS_VARIADIC_MAP_

#include <unordered_map>
#include <iostream>

namespace frnn {

template <typename T>    
struct VariadicMap {
public:
    /* ======================================== Typedefs ================================================== */
    typedef std::unordered_map<T, size_t>   map;
    typedef typename map::size_type         size_type;
    typedef typename map::iterator          iterator;
    typedef typename map::const_iterator    const_iterator;
    /* ==================================================================================================== */ 
private:
    std::unordered_map<T, size_type>   _elements;    
public:
    template <typename... Ts>
    VariadicMap(T element, Ts... elements) 
    {
        createMap<0>(element, elements...);
    }    
        
    template <size_type iter, typename E>
    void createMap(E element)
    {
        _elements.insert(std::make_pair<T, size_t>(static_cast<T>(element), iter));
    }
    
    template <size_type iter = 0, typename E, typename... Ts>
    void createMap(E element, Ts... elements)
    {
        _elements.insert(std::make_pair<T, size_type>(static_cast<T>(element), iter));
        createMap<iter + 1>(elements...);
    }
   
    size_type size() const { return _elements.size(); }
    
    const_iterator begin() const { return _elements.begin(); }
    iterator begin() { return _elements.begin(); }
    const_iterator end() const  { return _elements.end(); }
    iterator end()   { return _elements.end(); }    
};

}       // End namespace frnn

#endif
