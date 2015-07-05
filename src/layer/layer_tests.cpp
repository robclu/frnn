/*
 *  Test file for cuRNN layer class.
 *
 *  Copyright (C) 2015 Rob Clucas robclu1818@gmail.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT AN_size.y WARRANTY; without even the implied warranty of
 *  MERCHANTABILIT_size.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  _size.you should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "layer.hpp"
#include "softmaxPolicy.hpp"

const size_t INPUTS  = 3;
const size_t NODES   = 4;
const size_t DEPTH   = 4;

typedef curnn::layer<float, NODES, INPUTS, DEPTH, curnn::policies::softmaxPolicy> curnnLayerSmaxf;

TEST( curnnTensor, CanCreateSoftmaxLayerCorrectly ) {
    curnnLayerSmaxf softmaxLayer;

    EXPECT_EQ( softmaxLayer.numNodes    , NODES  );
    EXPECT_EQ( softmaxLayer.numInputs   , INPUTS );
    EXPECT_EQ( softmaxLayer.depth       , DEPTH  );
}

TEST( curnnTensor, CanForwardPassOnSoftmaxLayer ) {
    curnnLayerSmaxf softmaxLayer;

    std::vector<float> ins, outs;

    for ( int i = 0; i < INPUTS; i++ ) {
        ins.push_back( 0.5f );
    }

    softmaxLayer.forward( ins, outs );
    
    EXPECT_EQ( outs[ 0 ], 0.2f );
}
