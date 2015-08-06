/*
 *  Test file for fastRNN container classes.
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

#include <gtest/gtest.h>
#include <iostream>

#include "tuple.h"
#include "ordered_map.h"
#include <string>

TEST( frnnTuple, CanCreateTupleWithMultipleTypes )
{
    frnn::Tuple<int, char> tuple(4, 'c');
    EXPECT_EQ( frnn::tuple::size(tuple), 2 );
}

TEST( frnnTuple, CanCreateTupleWithAnyNumberOfElements )
{
    frnn::Tuple<int, int, float, std::string> tuple(4, 3, 3.7, "string");
    EXPECT_EQ( frnn::tuple::size(tuple), 4 );
}

TEST( frnnTuple, CanGetTupleElement )
{
    frnn::Tuple<int, int, float, std::string> tuple(4, 3, 3.7, "string");
    
    int element = frnn::get<1>(tuple);
    
    EXPECT_EQ( element, 3 );
}

TEST( frnnTuple, CanSetTupleElement )
{
    frnn::Tuple<int, int, float, std::string> tuple(4, 3, 3.7, "string");
    
    frnn::get<2>(tuple) = 4.5;
    
    EXPECT_EQ( 4.5, frnn::get<2>(tuple) );
}

TEST( frnnOrderedMap, CanCreateVariadicMapAndGetSize ) 
{
    frnn::OrderedMap<int> omap(4, 2, 1);
    EXPECT_EQ( 3, omap.size() );
}

TEST( frnnOrderedMap, CanIterateOverMap ) 
{
    frnn::OrderedMap<int> omap(4, 2, 1, 5, 6);
    
    int sum = 0;            // Sum of elements
    
    for (auto& element : omap ) sum += element.first;
            
    EXPECT_EQ( sum, 18 );
}

TEST( frnnOrderedMap, CanFindElementsInMap ) 
{
    frnn::OrderedMap<int> omap(4, 2, 1, 5, 6);
    
    auto element = omap.find(2);
    
    EXPECT_EQ( element->first , 2 );
    EXPECT_EQ( element->second, 1 );
}

TEST( frnnOrderedMap, CanInsertElementIntoMap )
{
    frnn::OrderedMap<int> omap(4, 2, 1, 5, 6);
    int x =10;    
    
    omap.insert(9);         // Check r-value ref works
    omap.insert(x);         // Check l-value ref works
    
    auto element = omap.find(9);
    
    EXPECT_EQ( element->first , 9 );
    EXPECT_EQ( element->second, 5 );
}   

TEST( frnnOrderedMap, CanInsertIteratorIntoMap )
{
    frnn::OrderedMap<int> omap_1(4, 2, 1, 5, 6);
    frnn::OrderedMap<int> omap_2;
    
    // Insert into map 2 from map 1
    omap_2.insert(omap_1.find(2));
    
    EXPECT_EQ( omap_2.size(), 1 );
}    
