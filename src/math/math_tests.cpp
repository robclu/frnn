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

// Constants for faster (but less thorough) testing
#define FAST_TEST 1

// General tesing parameres 
// Note : Do not make this so big that the GPU will run out of memory,
//        which is only really a problem for the double precision functions
const size_t NUM_ELEMENTS = 3e6;

TEST( curnnMath, AxpyOperationComputesCorrectlyWithFloats ) {
	// Create curnn error status
	curnn::curnnError error;
	const float A = 2.0f;

	// Create data vectors
	vector<float> x;
	vector<float> y;

	// Fill vectors with data
	for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
		x.push_back( float( i ) ); 
		y.push_back( float( i ) );
	}

	// Execute axpy with floats
	curnn::math::axpy( error, A, x, y );

	if ( FAST_TEST ) {											// FAST testing
		if ( NUM_ELEMENTS < 10 ) {
			for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
				EXPECT_EQ( y[i], A * i + i );
			}
		} else {
			for ( size_t i = NUM_ELEMENTS - 10; i < NUM_ELEMENTS; i++ ) {
				EXPECT_EQ( y[i], A * i + i );
			}
		}	
	} else {													// THOROUGH testing
		for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
			EXPECT_EQ( y[i], A * i + i );
		}
	}
}

TEST( curnnMath, AxpyOperationComputesCorrectlyWithDoubles ) {
	// Create curnn error status
	curnn::curnnError error;
	const double A = 2.0f;

	// Create data vectors
	vector<double> x;
	vector<double> y;

	// Fill vectors with data
	for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
		x.push_back( double( i ) ); 
		y.push_back( double( i ) );
	}

	// Performs axpy with doubles
	curnn::math::axpy( error, A, x, y );

	if ( FAST_TEST ) {											// FAST testing
		if ( NUM_ELEMENTS < 10 ) {
			for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
				EXPECT_EQ( y[i], A * i + i );
			}
		} else {
			for ( size_t i = NUM_ELEMENTS - 10; i < NUM_ELEMENTS; i++ ) {
				EXPECT_EQ( y[i], A * i + i );
			}
		}	
	} else {													// THOROUGH testing
		for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
			EXPECT_EQ( y[i], A * i + i );
		}
	}
}

TEST( curnnMath, ReductionSumComputesCorrectlyWithFloats ) {
	// Create curnn error status
	curnn::curnnError error;

	// Create data vector 
	vector<float> x;

	// Fill x with data 
	for ( size_t i = 0; i < NUM_ELEMENTS; i++ ) {
		x.push_back( 1.f );
	}

	EXPECT_EQ( NUM_ELEMENTS, curnn::math::sum( error, x ) );
}
