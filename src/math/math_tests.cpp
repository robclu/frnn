/*
 *  Test file for fastRNN math functions.
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

#include "../frnn/types.h"
#include "math.hpp"             // Math functions for both CPU and GPU

using std::vector;
using frnn::device;

// General tesing parameres 
// Note : Do not make this so big that the GPU will run out of memory,
//        which is only really a problem for the double precision functions
const size_t NUM_ELEMENTS      = 3e6;
const size_t NUM_ELEMENTS_RAND = 1e5;
const size_t NUM_ELEMENTS_CPU  = 3e6;
const float  TOLERANCE         = 1e-4;     // For difference between GPU and CPU math functions

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

TEST( frnnMathGpu, CanGenerateNRandomNumbersUniformDistribution ) {
    float lo = -2.0f; float hi = 10.f;
    float random_numbers[ NUM_ELEMENTS_RAND ];
    
    // Generate 10 randon numbers on the GPU
    frnn::math<float, frnn::device::GPU>::rand( random_numbers, NUM_ELEMENTS_RAND, lo, hi );    
    
    for ( size_t i = 0; i < NUM_ELEMENTS_RAND; i++ ) {
        EXPECT_GE( random_numbers[ i ], lo );
        EXPECT_LT( random_numbers[ i ], hi );
    }
}

TEST( frnnMathGpu, AxpyOperationComputesCorrectlyWithFloats ) {
    frnn::frnnError error;
    const float A = 2.0f;

    vector<float> x;
    vector<float> y;

    // Fill vectors with data
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( float( i ) ); 
        y.push_back( float( i ) );
    }

    // axpy with floats
    frnn::math<float, frnn::device::GPU>::axpy( error, A, x, y );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( y[i], A * i + i );
    }
}

TEST( frnnMathGpu, AxpyOperationComputesCorrectlyWithDoubles ) {
    frnn::frnnError error;
    const double A = 2.0f;
    
    vector<double> x;
    vector<double> y;

    // Fill vectors with data
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( double( i ) ); 
        y.push_back( double( i ) );
    }

    // Performs axpy with doubles
    frnn::math<double, frnn::device::GPU>::axpy( error, A, x, y );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( y[i], A * i + i );
    }
}

TEST( frnnMathGpu, ReductionSumComputesCorrectlyWithFloats ) {
    frnn::frnnError error;
    vector<float> x;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }
    
    float sum_of_elements = frnn::math<float, frnn::device::GPU>::sum( error, x );
    EXPECT_EQ( NUM_ELEMENTS, sum_of_elements );
}

TEST( frnnMathGpu, ReductionSumComputesCorrectlyWithInts ) {
    frnn::frnnError error;
    vector<int> x;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( int( 1 ) );
    }
    
    int sum_of_elements = frnn::math<int, frnn::device::GPU>::sum( error, x );
    EXPECT_EQ( NUM_ELEMENTS, sum_of_elements );
}

TEST( frnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithFloatsAndEmptyResultsVector ) {
    frnn::frnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }

    // Get the results of the sum into the results vector
    frnn::math<float, frnn::device::GPU>::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( frnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithFloatsAndFullResultsVector ) {
    frnn::frnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
        results.push_back( 0.f );
    }

    // Get the results of the sum into the results vector
    frnn::math<float, frnn::device::GPU>::sumVectorized( error, x, results );
        
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( frnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithIntsAndEmptyResultsVector ) {
    frnn::frnnError error;
    vector<int> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1 );
    }

    // Get the results of the sum into the results vector
    frnn::math<int, frnn::device::GPU>::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( frnnMathGpu, ReductionSumVectorizedComputesCorrectlyWithIntsAndFullResultsVector ) {
    frnn::frnnError error;
    vector<int> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1 );
        results.push_back( 0.f );
    }

    // Get the results of the sum into the results vector
    frnn::math<int, frnn::device::GPU>::sumVectorized( error, x, results );

    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        EXPECT_EQ( NUM_ELEMENTS, results[ i ]  );
    }
}

TEST( frnnMathGpu, SoftmaxComputesCorrectlyForFloats ) {
    frnn::frnnError error;
    vector<float> x, results;

    // Fill x with data 
    for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
        x.push_back( 1.f );
    }

    // Get the results of the sum into the results vector
    frnn::math<float, frnn::device::GPU>::softmax( error, x, results );

    for ( size_t i = 1; i < NUM_ELEMENTS; i++ ) {
        float softmax_i = exp( 1.f ) / ( exp( 1.f ) * (float)NUM_ELEMENTS );
        // Account for difference in GPU exp and CPU exp
        EXPECT_NEAR( softmax_i, results[ i ], TOLERANCE );
    }
}

TEST( frnnMathCpu, CanGenerateNRandomNumbersUniformDistribution ) {
    float lo = 2.0f; float hi = 13.1f;
    float random_numbers[ NUM_ELEMENTS_RAND ];
    
    // Generate 10 randon numbers on the CPU
    frnn::math<float, frnn::device::CPU>::rand( random_numbers, NUM_ELEMENTS_RAND, lo, hi );    
    
    for ( size_t i = 0; i < NUM_ELEMENTS_RAND; i++ ) {
        EXPECT_GE( random_numbers[ i ], lo );
        EXPECT_LT( random_numbers[ i ], hi );
    }
}

TEST( frnnMathCpu, CanPerformXminusYWithVectorizedCpuKernelWithFloats ) {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> out;
    
    for ( size_t i = 0; i < NUM_ELEMENTS_CPU; i++ ) {
        x.push_back( float( i ) );
        y.push_back( float( i + 1 ) );
        out.push_back( 0.0f );
    }
    
    // Perform CPU X - Y
    frnn::math<float, frnn::device::CPU>::xmy( x, y, out );
    
    for ( size_t i = 0; i < out.size(); i++ ) {
        EXPECT_EQ( out[ i ], -1.f );
    }
}

TEST( frnnMathCpu, CanPerformXminusYWithVectorizedCpuKernelWithDoubles ) {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> out;
    
    for ( size_t i = 0; i < NUM_ELEMENTS_CPU; i++ ) {
        x.push_back( double( i ) );
        y.push_back( double( i + 1 ) );
        out.push_back( 0.0 );
    }
    
    // Perform CPU X - Y
    frnn::math<double, frnn::device::CPU>::xmy( x, y, out );
    
    for ( size_t i = 0; i < out.size(); i++ ) {
        EXPECT_EQ( out[ i ], -1.0 );
    }
}

