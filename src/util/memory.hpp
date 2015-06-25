#ifndef CURNN_UTIL_MEMORY_INCLUDED
#define CURNN_UTIL_MEMORY_INCLUDED

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdlib.h>
#include <iostream>

// To convert variable names to strings
#define stringify( name ) varname( #name )
inline char* varname( char* name ) { return name; }

namespace curnn {
	namespace util {
		
		namespace err {
			/*
			 * ==============================================================================================
			 * Function		: allocError
			 *
			 * Description	: Prints an error message of allocation failed
			 *
			 * Inputs		: varname	: The name of the variable that could not be allocated
			 * ==============================================================================================
			 */
			inline void allocError( char * varname ) {
				std::cerr << "Error allocating memory for " << varname << "\n";
			}

			/*
			 * ==============================================================================================
			 * Function		: copyError
			 *
			 * Description	: Prints an error message if host to device, or device to host copy failed
			 *
			 * Inputs		: varname	: The name of the variable that could not be copied to/from
			 * ==============================================================================================
			 */
			inline void copyError( char* varname ) {
				std::cerr << "Error copying to/from variable " << varname << "\n";
			}
		}

		namespace mem {
			/*
			 * ==============================================================================================
			 * Function		: allocVector
			 *
			 * Description	: Allocates space for a device vector and copies data to it, and chceks for 
			 *                errors
			 *
			 * Inputs		: status		: CUBLAS status handle
			 *				: hostVector	: The pointer to the filled host vector
			 *				: numElements	: The number of elements in the vector
			 *				: devVector		: The pointer to the empty device vector
			 * ==============================================================================================
			 */
			template <class dType>
			void allocVector( cublasStatus_t& status, const dType* hostVector,  size_t numElements, dType* devVector ) {
				// Allocate memory on the device
				if ( cudaMalloc( (void**)&devVector, numElements * sizeof( dType ) ) != cudaSuccess ) {
					err::allocError( stringify( devVector ) );
					exit( EXIT_FAILURE );
				}
				// Set memory on device
				status = cublasSetVector( numElements, sizeof( dType ), hostVector, 1, devVector, 1 );
				if ( status != CUBLAS_STATUS_SUCCESS ) {
					err::copyError( stringify( devVector ) );
					exit( EXIT_FAILURE );
				} 
			}

			/*
			 * ==============================================================================================
			 * Function		: getVector
			 *
			 * Description	: Gets a vector from the device into a host vector, and checks for errors.
			 *
			 * Inputs		: status		: CUBLAS status handle
			 *				: hostVector	: The pointer to the filled host vector
			 *				: numElements	: The number of elements in the vector
			 *				: devVector		: The pointer to the empty device vector
			 * ==============================================================================================
			 */
			template <class dType>
			void getVector( cublasStatus_t& status, const dType* devVector,  size_t numElements, dType* hostVector ) {
				// Get data from device
				status = cublasGetVector( numElements, sizeof( dType ), devVector, 1, hostVector, 1 );
				if ( status != CUBLAS_STATUS_SUCCESS ) {
					err::copyError( stringify( hostVector ) );
					exit( EXIT_FAILURE );
				} 
			}
		}
	}
}
#endif
