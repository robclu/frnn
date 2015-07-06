/*
 *  Test file for cuRNN math functions.
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

#include "math.hpp"             // Includes kernels 
#include "math_general.h"

using std::vector;

// General tesing parameres 
// Note : Do not make this so big that the GPU will run out of memory,
//        which is only really a problem for the double precision functions
const size_t NUM_ELEMENTS = 3e6;
const float  TOLERANCE    = 1e-4;     // For difference between GPU and CPU math functions

/* =========================================== NOTES ========================================================
 *
 * 1. Sum does not work with doubles since the kernels use the shfl operations which can only handle ints and
 *    floats. (conversion from other params to these will be provided later on).
 *
 * 2. Softmax works with ints but the testing is not done as it is useless, since softmax returns a
 *    'probablility' for each element in the vector on the range [0, 1], using ints will give the result 
 *    of 0 for each element, which is not worth the increased time for test execution
 *
 * ==========================================================================================================
 */

TEST( curnnMathGpu, CanGenerateRandomNumbersUniformDistribution ) {
    float lo = 0.0f; float hi = 0.1f;
    
    EXPECT_GE( curnn::math::rand( lo, hi ), lo );
    EXPECT_LT( curnn::math::rand( lo, hi ), hi );
}

TEST( curnnMathGpu, AxpyOperationComputesCorrectlyWithFloats ) {
    curnn::curnnError error;
    const float A = 2.0f;

    vector<float> x;
    vector<float> y;

    // Fill vectors with data
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( float( i ) ); 
        y.push_back( float( i ) );
    }

    // axpy with floats
    curnn::math::axpy( error, A, x, y );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( y[i], A * i + i );
    }
}

TEST( curnnMathGpu, AxpyOperationComputesCorrectlyWithDoubles ) {
    curnn::curnnError error;
    const double A = 2.0f;
    
    vector<double> x;
    vector<double> y;

    // Fill vectors with data
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( double( i ) ); 
        y.push_back( double( i ) );
    }

    // Performs axpy with doubles
    curnn::math::axpy( error, A, x, y );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( y[i], A * i + i );
    }
}

TEST( curnnMathGpu, ReductionSumComputesCorrectlyWithFloats ) {
    curnn::curnnError error;
    vector<float> x;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }

    EXPECT_EQ( NUM_ELEMENTS, curnn::math::sum( error, x ) );
}

TEST( curnnMathGpu, ReductionSumComputesCorrectlyWithInts ) {
    curnn::curnnError error;
    vector<int> x;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( int( 1 ) );
    }

    EXPECT_EQ( NUM_ELEMENTS, curnn::math::sum( error, x ) );
}

TEST( curnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithFloatsAndEmptyResultsVector ) {
    curnn::curnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }

    // Get the results of the sum into the results vector
    curnn::math::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( curnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithFloatsAndFullResultsVector ) {
    curnn::curnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
        results.push_back( 0.f );
    }

    // Get the results of the sum into the results vector
    curnn::math::sumVectorized( error, x, results );
        
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( curnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithIntsAndEmptyResultsVector ) {
    curnn::curnnError error;
    vector<int> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1 );
    }

    // Get the results of the sum into the results vector
    curnn::math::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( curnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithIntsAndFullResultsVector ) {
    curnn::curnnError error;
    vector<int> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1 );
        results.push_back( 0.f );
    }

    // Get the results of the sum into the results vector
    curnn::math::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( curnnMathGpu, SoftmaxComputesCorrectlyForFloats ) {
    curnn::curnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }

    // Get the results of the sum into the results vector
    curnn::math::softmax( error, x, results );

    for ( size_t i = 1; i < NUM_ELEMENTS; i++ ) {
        float softmax_i = exp( 1.f ) / ( exp( 1.f ) * (float)NUM_ELEMENTS );
        // Account for difference in GPU exp and CPU exp
        EXPECT_NEAR( softmax_i, results[ i ], TOLERANCE );
    }
}

TEST( curnnMathCpu, CanPerformXminusYWithVectorizedCpuKernelWithFloats ) {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> out;
    
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( float( i ) );
        y.push_back( float( i + 1 ) );
        out.push_back( 0.0f );
    }
    
    // Perform CPU X - Y
    xmyCpu( x, y, out );
    
    for ( size_t i = 0; i < out.size(); i++ ) {
        EXPECT_EQ( out[ i ], -1.f );
    }
}

TEST( curnnMathCpu, CanPerformXminusYWithVectorizedCpuKernelWithDoubles ) {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> out;
    
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( double( i ) );
        y.push_back( double( i + 1 ) );
        out.push_back( 0.0 );
    }
    
    // Perform CPU X - Y
    xmyCpu( x, y, out );
    
    for ( size_t i = 0; i < out.size(); i++ ) {
        EXPECT_EQ( out[ i ], -1.0 );
    }
}

