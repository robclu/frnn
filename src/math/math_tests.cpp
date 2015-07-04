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
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "math.hpp"
#include "math.cuh"

using std::vector;

// General tesing parameres 
// Note : Do not make this so big that the GPU will run out of memory,
//        which is only really a problem for the double precision functions
const size_t NUM_ELEMENTS = 3e5;
const float  TOLERANCE	  = 1e-4;     // For difference between GPU and CPU math functions

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

TEST( curnnMath, AxpyOperationComputesCorrectlyWithFloats ) {
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

TEST( curnnMath, AxpyOperationComputesCorrectlyWithDoubles ) {
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

TEST( curnnMath, ReductionSumComputesCorrectlyWithFloats ) {
	curnn::curnnError error;
	vector<float> x;

	// Fill x with data 
	for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
		x.push_back( 1.f );
	}

	EXPECT_EQ( NUM_ELEMENTS, curnn::math::sum( error, x ) );
}

TEST( curnnMath, ReductionSumComputesCorrectlyWithInts ) {
	curnn::curnnError error;
	vector<int> x;

	// Fill x with data 
	for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
		x.push_back( int( 1 ) );
	}

	EXPECT_EQ( NUM_ELEMENTS, curnn::math::sum( error, x ) );
}

TEST( curnnMath, ReductionSumVectorizedComputesCorrectlyWithFloatsAndEmptyResultsVector ) {
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

TEST( curnnMath, ReductionSumVectorizedComputesCorrectlyWithFloatsAndFullResultsVector ) {
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

TEST( curnnMath, ReductionSumVectorizedComputesCorrectlyWithIntsAndEmptyResultsVector ) {
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

TEST( curnnMath, ReductionSumVectorizedComputesCorrectlyWithIntsAndFullResultsVector ) {
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

TEST( curnnMath, SoftmaxComputesCorrectlyForFloats ) {
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

TEST( curnnMath, SoftmaxComputesCorrectlyOnTensors ) {
	curnn::curnnError error;
	
	// Create tensor that holds 2 pages, and each page has 
	// and M x N weight matrix, 1 x N bias vector, and 1 x N 
	// results vector so the tensor has dimension:
	// ( M + 1 + 1 ) X N X 2 X 0
	uint N = 2;				// Num nodes in layer
	uint I = 4;				// Num inputs 
	uint M = I + 2;			// I inputs, a bias, and activation
	uint D = 2;				// depth of 2
	curnn::tensor4<float> tensor( N, M, D, 1 );

	// Fill tensor with data 
	for ( int i = 0; i < tensor.z; i++ ) {
		for ( int j = 0; j < tensor.y; j++ ) {
			for ( int k = 0; k < tensor.x; k++ ) {
				tensor( k, j, i, 0 ) = k + 1;
				std::cout << tensor( k, j, i, 0 ) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl;
	}

	// Simple inputs for testing ( needs to be same dimension as N
	std::vector<float> x = { 2.f, 2.f, 2.f, 3.f };
	std::vector<float> y;						// Outputs

	// Execute softmax function on tensor using x and storing the results in y
	curnn::math::softmax( error, x, tensor, I, y );
	
	// Check results
	EXPECT_EQ( y[ 0 ], 20.f );
	EXPECT_EQ( y[ 1 ], 40.f );
}

