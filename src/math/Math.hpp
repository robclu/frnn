#ifndef CURRN_MATH_INCLUDED
#define CURNN_MATH_INCLUDED

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../util/error.hpp"

// To convert variable names to strings
#define stringify( name ) varname( #name )
char* varname( char* name ) { return name; }

namespace currn {
	namespace math {
		/*
		 * =========================================================================     
		 * Function		: axpy
		 *
		 * Description	: performs a*X + Y
		 *
		 * Inputs		: a		: Constant for multiplication 
		 *              : x     : Vector to multiply with a
		 *              : y     : Vector to add to a*x
		 * 
		 * Outputs		:
		 * =========================================================================
		 */
		template<class dType>
		void saxpy( const dType a, vector<dType> x, vector<dType> y ) {
			dType* da, *db, *dc;			// Device variables

			// Allocate memory for device variables			
			if ( cudaMalloc( (void**)&da, x.size() * sizeof( dType ) ) != cudaSuccess ) {
					curnn::util::err::allocError( stringify( da ) );
			}
				

		}
	}
}
#endif

