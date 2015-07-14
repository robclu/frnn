/*
 *  Header file for fastRNN CPU vectorized types and isntructions The provide 
 *  a wrapped to the SIMD instructions and types required for using the SIMD
 *  instructions. The reason for this is that the instructions are sepcific to 
 *  the type (float, double ...) but the frnn functions are all tyoe templates
 *  so the wrapper allows the correct functiosn and types to be called for any 
 *  templated function.
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
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation,
 *	Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _FRNN_VECTORIZED_TYPES_CPU_
#define _FRNN_VECTORIZED_TYPES_CPU_

#include <emmintrin.h>          // SSE vectorized types

namespace frnn {
    
/*
 * ==========================================================================================================
 * Struct       : VectorizedTypeCpu
 * 
 * Description  : Gets a vectorized (v4) version of a type for sse instructions.
 * 
 * Params       : dType     : The type of data (float, double, int )
 * 
 * Example      : Calling frnn::vectorizedTypeCpu<float>::vectType will then use the __m128 type
 * ==========================================================================================================
 */
template <typename dType> struct VectorizedTypeCpu;

template <> struct VectorizedTypeCpu<int>    { typedef __m128i vect_type; };
template <> struct VectorizedTypeCpu<char>   { typedef __m128i vect_type; };
template <> struct VectorizedTypeCpu<float>  { typedef __m128  vect_type; };
template <> struct VectorizedTypeCpu<double> { typedef __m128d vect_type; };
template <> struct VectorizedTypeCpu<float*> { typedef __m128* vect_type; };

/*
 * ==========================================================================================================
 * Struct       : VectorizedInstructions 
 * 
 * Description  : Provides general names for the SIMD instructions for any type instance of the class. The
 *                function pointers allow the functions to be called without an instance of the struct
 *                
 * Params       : dType     : The type of data 
 * ==========================================================================================================
 */

template <typename dType> struct VectorizedInstructionsCpu;

// Size functions for float and double
constexpr size_t sizeFloatVectorized()  { return 4; }
constexpr size_t sizeDoubleVectorized() { return 2; }

// Float specification
template <> struct VectorizedInstructionsCpu<float> {
    
    static constexpr auto typeSize = &sizeFloatVectorized;
    
    typedef __m128 (*load)( const float* );
    static constexpr load mm_load_u = &_mm_loadu_ps;
    
    typedef __m128 (*sub)( __m128, __m128 );
    static constexpr sub mm_sub_p = &_mm_sub_ps;
    
    typedef void (*store)( float*, __m128 );
    static constexpr store mm_store_p = &_mm_store_ps;
};

// Double specification
template <> struct VectorizedInstructionsCpu<double> {
   
    static constexpr auto typeSize = &sizeDoubleVectorized;

    typedef __m128d (*load)( const double* );
    static constexpr load mm_load_u = &_mm_loadu_pd;
    
    typedef __m128d (*sub)( __m128d, __m128d );
    static constexpr sub mm_sub_p = &_mm_sub_pd;
    
    typedef void (*store)( double*, __m128d );
    static constexpr store mm_store_p = &_mm_store_pd;
};

}   // Namespace frnn


#endif
