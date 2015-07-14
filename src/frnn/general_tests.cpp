/*
 *  Test file for fastRNN util functions.
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
#include <typeinfo>

#include "types.h"

/* 
 * =========================================== NOTES ========================================================
 *
 * 1. When using the vectorized instances in device code with templated functions, 
 *    you will need to first use a typedef, so for example if the template prameter 
 *    is dType then
 *
 *       template <typename dType> 
 *       void example_function( ... ) {
 *			
 *			typedef typename frnn::VectorizedTypeGpu<dType, 2> vec2;
 *
 *			vec2 vectorized_dtype;
 *		}			
 *
 * ==========================================================================================================
 */

TEST( frnnTypesGpu, CanDetermineVectorizedDoublesFromDouble ) {
	double1 cudaDouble1;
	double2 cudaDouble2;

	// Declare a frnn vectorized doubles using type traits
	frnn::VectorizedTypeGpu<double, 1>::vect_type frnnDouble1;
	frnn::VectorizedTypeGpu<double, 2>::vect_type frnnDouble2;
	
	EXPECT_EQ( typeid( cudaDouble1 ).name(), typeid( frnnDouble1 ).name() );
	EXPECT_EQ( typeid( cudaDouble2 ).name(), typeid( frnnDouble2 ).name() );
}

TEST( frnnTypesGpu, CanDetermineVectorizedFloatFromFloat ) {
	float1 cudaFloat1;
	float2 cudaFloat2;
	float4 cudaFloat4;

	// Declare frnn vectorized floats using type traits
	frnn::VectorizedTypeGpu<float, 1>::vect_type frnnFloat1;
	frnn::VectorizedTypeGpu<float, 2>::vect_type frnnFloat2;
	frnn::VectorizedTypeGpu<float, 4>::vect_type frnnFloat4;
	
	EXPECT_EQ( typeid( cudaFloat1 ).name(), typeid( frnnFloat1 ).name() );
	EXPECT_EQ( typeid( cudaFloat2 ).name(), typeid( frnnFloat2 ).name() );
	EXPECT_EQ( typeid( cudaFloat4 ).name(), typeid( frnnFloat4 ).name() );
}

TEST( frnnTypesGpu, CanDetermineVectorizedIntFromInt ) {
	int1 cudaInt1;
	int2 cudaInt2;
	int4 cudaInt4;

	// Declare frnn vectorized ints using type traits
	frnn::VectorizedTypeGpu<int, 1>::vect_type frnnInt1;
	frnn::VectorizedTypeGpu<int, 2>::vect_type frnnInt2;
	frnn::VectorizedTypeGpu<int, 4>::vect_type frnnInt4;
	
	EXPECT_EQ( typeid( cudaInt1 ).name(), typeid( frnnInt1 ).name() );
	EXPECT_EQ( typeid( cudaInt2 ).name(), typeid( frnnInt2 ).name() );
	EXPECT_EQ( typeid( cudaInt4 ).name(), typeid( frnnInt4 ).name() );
}

TEST( frnnTypesGpu, CanDetermineVectorizedUIntFromUInt ) {
	uint1 cudaUint1;
	uint2 cudaUint2;
	uint4 cudaUint4;

	// Declare frnn vectorized unsigned ints using type traits
	frnn::VectorizedTypeGpu<uint, 1>::vect_type frnnUint1;
	frnn::VectorizedTypeGpu<uint, 2>::vect_type frnnUint2;
	frnn::VectorizedTypeGpu<uint, 4>::vect_type frnnUint4;
	
	EXPECT_EQ( typeid( cudaUint1 ).name(), typeid( frnnUint1 ).name() );
	EXPECT_EQ( typeid( cudaUint2 ).name(), typeid( frnnUint2 ).name() );
	EXPECT_EQ( typeid( cudaUint4 ).name(), typeid( frnnUint4 ).name() );
}

TEST( frnnTypesGpu, CanDetermineVectorizedCharFromChar ) {
	char1 cudaChar1;
	char2 cudaChar2;
	char4 cudaChar4;

	// Declare frnn vectorized unsigned ints using type traits
	frnn::VectorizedTypeGpu<char, 1>::vect_type frnnChar1;
	frnn::VectorizedTypeGpu<char, 2>::vect_type frnnChar2;
	frnn::VectorizedTypeGpu<char, 4>::vect_type frnnChar4;
	
	EXPECT_EQ( typeid( cudaChar1 ).name(), typeid( frnnChar1 ).name() );
	EXPECT_EQ( typeid( cudaChar2 ).name(), typeid( frnnChar2 ).name() );
	EXPECT_EQ( typeid( cudaChar4 ).name(), typeid( frnnChar4 ).name() );
}

TEST( frnnTypesCpu, CanDetermineVectorizedFloatFromFloat ) {
    frnn::VectorizedTypeCpu<float>::vect_type frnnVectorizedCpu;
    __m128 sseVectorizedCpu;
    
    EXPECT_EQ( typeid( frnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( frnnTypesCpu, CanDetermineVectorizedDoubleFromDouble ) {
    frnn::VectorizedTypeCpu<double>::vect_type frnnVectorizedCpu;
    __m128d sseVectorizedCpu;
    
    EXPECT_EQ( typeid( frnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( frnnTypesCpu, CanDetermineVectorizedIntFromInt ) {
    frnn::VectorizedTypeCpu<int>::vect_type frnnVectorizedCpu;
    __m128i sseVectorizedCpu;
    
    EXPECT_EQ( typeid( frnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}

TEST( frnnTypesCpu, CanDetermineVectorizedCharFromChar ) {
    frnn::VectorizedTypeCpu<char>::vect_type frnnVectorizedCpu;
    __m128i sseVectorizedCpu;
    
    EXPECT_EQ( typeid( frnnVectorizedCpu ).name(), typeid( sseVectorizedCpu ).name() );
}
