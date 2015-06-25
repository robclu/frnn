#include "math.hpp"
#include <iostream>
#include <stdlib.h>

namespace curnn {
void saxpy( const float a, const std::vector<float>& x, std::vector<float>& y ) {

	cublasStatus_t status;			
	cublasHandle_t handle;
	float* da = 0, *dx = 0, *dy = 0;

	// Initialize handle
	status = cublasCreate( &handle );
	cublasSetPointerMode( handle, CUBLAS_POINTER_MODE_DEVICE );

	// Allocate and fill device vectors with host vector data (checks for errors)
	curnn::util::mem::allocVector( status, &a, 1, da );	
	curnn::util::mem::allocVector( status, &x[0], x.size(), dx );
	curnn::util::mem::allocVector( status, &y[0], y.size(), dy );

	// Perform CUBLAS saxpy
	status = cublasSaxpy( handle, x.size(), da, dx, 1, dy, 1 );

	// Get the result (checks for errors)
	curnn::util::mem::getVector( status, dy, y.size(), &y[0] );	

	// Destroy cublas handle
	status = cublasDestroy( handle );

	cudaFree( da );
	cudaFree( dx );
	cudaFree( dy );
}
}