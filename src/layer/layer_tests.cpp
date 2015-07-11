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
#include "types/softmax_policy.hpp"
#include "../curnn/curnn.h"

const size_t    INPUTS      = 200;
const size_t    NODES       = 80;
const size_t    DEPTH       = 4;
const float     TOLERANCE   = 1e-5;

typedef curnn::Layer<float,                             // Data type
                     curnn::device::GPU,                // Device type
                     NODES, INPUTS, DEPTH,              // Size
                     curnn::ltype::SoftmaxPolicy>  curnnLayerSmaxf;        

TEST( curnnLayer, CanCreateSoftmaxLayerCorrectly ) {
    curnnLayerSmaxf softmaxLayer;

    EXPECT_EQ( softmaxLayer.num_nodes    , NODES  );
    EXPECT_EQ( softmaxLayer.num_inputs   , INPUTS );
    EXPECT_EQ( softmaxLayer.depth        , DEPTH  );
}

TEST( curnnLayer, InitializesWeightsBiasesAndActivationsToZero ) {
    curnnLayerSmaxf softmaxLayer;
   
   const curnn::Tensor4<float> layerWghtsBiasActs = softmaxLayer.getWBA();
    
   // Just checking first depth level
   for ( uint i = 0; i < INPUTS + 2; i++ ) {
       for ( uint n = 0; n < NODES; n++ ) {
            EXPECT_NEAR( layerWghtsBiasActs( n, i, 0, 0 ), 0.0f, TOLERANCE );
       }
   }
}

TEST( curnnLayer, CanInitializeAndReadWeights ) {
    curnnLayerSmaxf softmaxLayer;
    float lo = 0.0f; float high = 1.0f;
    
    softmaxLayer.initializeWeights( lo, high );
   
   const curnn::Tensor4<float> layerWghtsBiasActs = softmaxLayer.getWBA();
    
   // Just checking first depth level
   for ( uint i = 0; i < INPUTS; i++ ) {
       for ( uint n = 0; n < NODES; n++ ) {
            EXPECT_GE( layerWghtsBiasActs( n, i, 0, 0 ), lo );
            EXPECT_LE( layerWghtsBiasActs( n, i, 0, 0 ), high );
       }
   }
}

TEST( curnnLayer, CanForwardPassOnSoftmaxLayer ) {
    curnnLayerSmaxf softmaxLayer;

    std::vector<float> ins, outs;

    for ( int i = 0; i < INPUTS; i++ ) {
        ins.push_back( 0.5f );
    }

    softmaxLayer.forward( ins, outs );

    // Sum of outputs must be 1 for correct distribution
    float sum = 0.0f;
    for ( uint i = 0; i < outs.size(); i++ ) sum += outs[ i ]; 
    
    // Account for floating point variablity
    EXPECT_NEAR( sum, 1.0f, TOLERANCE );
}

TEST( curnnLayer, SoftmaxLayerCanBackpropCorrectly ) {
    curnnLayerSmaxf softmaxLayer;

    std::vector<float> targets, outs;

    for ( int i = 0; i < NODES; i++ ) {
        outs.push_back( float( i ) );
        targets.push_back( float( i - 1 ) );
    }

    softmaxLayer.backward( outs, targets );
    const float* errs = softmaxLayer.getErrors();
    
    for ( uint i = 0; i < outs.size(); i++ ) {
        EXPECT_EQ( outs[ i ] - targets[ i ], errs[ i ] );
    }
}
