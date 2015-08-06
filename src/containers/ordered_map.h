// ==========================================================================================================
//! @file ordered_map.h
//!       Header file for the fastRNN OrderedMap class to create a hastable where the keys are specified and 
//!       each key has a value which is equal to number the element was inserted into the map.
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

#ifndef _FRNN_CONTAINERS_ORDERED_MAP_
#define _FRNN_CONTAINERS_ORDERED_MAP_

#include <unordered_map>
#include <typeinfo>

namespace frnn {

// ==========================================================================================================
//! @struct     OrderedMap
//! @brief      Creates an OrderedMap  where each of the arguments are the keys and the values are the 
//!             indeices of the arguments. Essentially this is just a list of elements that can be searched
//!             in constant time while still being able to determine their order.
//! @tparam     K       The type of the keys for the map.
// ==========================================================================================================
template <typename K>    
struct OrderedMap {
public:
    /* ======================================== Typedefs ================================================== */
    typedef std::unordered_map<K, size_t>   map;
    typedef typename map::size_type         size_type;
    typedef typename map::iterator          iterator;
    typedef typename map::const_iterator    const_iterator;
    /* ==================================================================================================== */ 
private:
    std::unordered_map<K, size_type>    _elements;                              //!< Key-value pair elements   
public:
    // ======================================================================================================
    //! @brief      Default constructor.
    // ======================================================================================================
    OrderedMap() {}
    
    // ======================================================================================================
    //! @brief      Adds all arguments as keys in the map, which the value being the element's argument index.
    //! @param[in]  element     The first element (key) to add to the map.
    //! @param[in]  elements    The other elements (keys) to add to the map.
    //! @tparam     Ks          The types of the other elements (keys) to add to the map.
    // ======================================================================================================
    template <typename... Ks>
    OrderedMap(K element, Ks... elements) 
    {
        createMap<0>(element, elements...);
    }    
        
    // ======================================================================================================
    //! @brief      Creates an unordered_map - terminating case.
    //! @param[in]  element     The element to add to the map.
    //! @tparam     iter        The iteration of the createMap function.
    //! @tparam     E           The type of the element to add to the mao.
    // ======================================================================================================
    template <size_type iter, typename E>
    void createMap(E element)
    {
        _elements.insert(std::make_pair<K, size_type>(static_cast<K>(element), iter));
    }
    
    // ======================================================================================================
    //! @brief  Creates an unordered_map - case for all but the terminating case.
    //! @param[in]  element     The element to add to the map.
    //! @param[in]  elements    The other elements still to be added to the map.
    //! @tparam     iter        The iteration number of the createMapfunction.
    //! @tparam     E           The type of the element to add to the map.
    //! @tparam     Es          The types of the other elements to add to the map.
    // ======================================================================================================
    template <size_type iter = 0, typename E, typename... Es>
    void createMap(E element, Es... elements)
    {
        _elements.insert(std::make_pair<K, size_type>(static_cast<K>(element), iter));
        createMap<iter + 1>(elements...);
    }

    // ======================================================================================================
    //! @brief      Inserts an element into the map.
    //! @param[in]  key     The key of the element to add.
    // ======================================================================================================
    void insert(K&& key) 
    {
        _elements.insert(std::make_pair<K, size_type>(static_cast<K>(key), _elements.size()));
    }
   
    // ======================================================================================================
    //! @brief      Moves an element from one map to this map.
    //! @param      it      The iterator to move to this map.
    // ======================================================================================================
    void insert(iterator it) 
    {
        _elements.insert(std::move(*it));
    }
   
    // ======================================================================================================
    //! @brief      Inserts an element into the map.
    //! @param[in]  key     The key of the element to add.
    // ======================================================================================================
    void insert(K& key) 
    {
        _elements.insert(std::make_pair<K, size_type>(std::move(key), _elements.size()));
    }
    // ======================================================================================================
    //! @brief  Gets the size of the map.
    //! @return The size of the map.
    // ======================================================================================================
    size_type size() const { return _elements.size(); }
    
    // ======================================================================================================
    //! @brief  Gets an iterator to the start of the map.
    //! @return A constant iterator which points to the beginning of the map.
    // ======================================================================================================
    const_iterator begin() const { return _elements.begin(); }
    
    // ======================================================================================================
    //! @brief  Gets an iterator to the start of the map.
    //! @return An iterator which points to the beginning of the map.
    // ======================================================================================================
    iterator begin() { return _elements.begin(); }
   
    // ======================================================================================================
    //! @brief  Gets an iterator to the end of the map.
    //! @return A constant iterator which points to the end of the map.
    // ======================================================================================================
    const_iterator end() const  { return _elements.end(); }

    // ======================================================================================================
    //! @brief  Gets an iterator to the end of the map.
    //! @return An iterator which points to the end of the map.
    // ======================================================================================================
    iterator end() { return _elements.end(); }    
   
    // ======================================================================================================
    //! @brief  Searches for an element in the map, and if found, returns an iterator to the element,
    //!         otherwise returns an iterator to the end of the map.
    //! @param  key     The key to search for in the map.
    //! @return An iterator which points to the element with a key key if key is a valid key for the map,
    //!         otherwise an iterator to the end of the map.
    // ======================================================================================================
    iterator find(const K& key) { return _elements.find(key); }

    // ======================================================================================================
    //! @brief  Searches for an element in the map, and if found, returns an iterator to the element,
    //!         otherwise returns an iterator to the end of the map.
    //! @param  key     The key to search for in the map.
    //! @return A constant iterator which points to the element with a key key if key is a valid key for the 
    //!         map, otherwise an iterator to the end of the map.
    // ======================================================================================================
    const_iterator find(const K& key) const { return _elements.find(key); }
};

}       // End namespace frnn

#endif
