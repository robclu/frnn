/*
 *  Header file for cuRNN COU vectorized types and isntructions The provide 
 *  a wrapped to the SIMD instructions and types required for using the SIMD
 *  instructions. The reason for this is that the instructions are sepcific to 
 *  the type (float, double ...) but the curnn functions are all tyoe templates
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

#ifndef _CURNN_VECTORIZED_TYPES_CPU_
#define _CURNN_VECTORIZED_TYPES_CPU_

#include <emmintrin.h>          // SSE vectorized types

namespace curnn {
    
/*
 * ==========================================================================================================
 * Struct       : vectorizedTypeCpu
 * 
 * Description  : Gets a vectorized (v4) version of a type for sse instructions.
 * 
 * Params       : dType     : The type of data (float, double, int )
 * 
 * Example      : Calling curnn::vectorizedTypeCpu<float>::vectType will then use the __m128 type
 * ==========================================================================================================
 */
template <typename dType> struct vectorizedTypeCpu;

template <> struct vectorizedTypeCpu<int>    { typedef __m128i vectType; };
template <> struct vectorizedTypeCpu<char>   { typedef __m128i vectType; };
template <> struct vectorizedTypeCpu<float>  { typedef __m128  vectType; };
template <> struct vectorizedTypeCpu<double> { typedef __m128d vectType; };
template <> struct vectorizedTypeCpu<float*>  { typedef __m128*  vectType; };

/*
 * ==========================================================================================================
 * Struct       : vectInstructions 
 * 
 * Description  : Provides general names for the SIMD instructions for any type instance of the class. The
 *                function pointers allow the functions to be called without an instance of the struct
 *                
 * Params       : dType     : The type of data 
 * ==========================================================================================================
 */

template <typename dType> struct vectInstructions;

// Size functions for float and double
constexpr size_t sizeFloatVect()  { return 4; }
constexpr size_t sizeDoubleVect() { return 2; }

// Float specification
template <> struct vectInstructions<float> {
    
    static constexpr auto typeSize = &sizeFloatVect;
    
    typedef __m128 (*load)( const float* );
    static constexpr load mm_load_u = &_mm_loadu_ps;
    
    typedef __m128 (*sub)( __m128, __m128 );
    static constexpr sub mm_sub_p = &_mm_sub_ps;
    
    typedef void (*store)( float*, __m128 );
    static constexpr store mm_store_p = &_mm_store_ps;
};

// Double specification
template <> struct vectInstructions<double> {
   
    static constexpr auto typeSize = &sizeDoubleVect;

    typedef __m128d (*load)( const double* );
    static constexpr load mm_load_u = &_mm_loadu_pd;
    
    typedef __m128d (*sub)( __m128d, __m128d );
    static constexpr sub mm_sub_p = &_mm_sub_pd;
    
    typedef void (*store)( double*, __m128d );
    static constexpr store mm_store_p = &_mm_store_pd;
};

}   // Namespace curnn


#endif
