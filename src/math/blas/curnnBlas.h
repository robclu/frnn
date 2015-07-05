/*
 *  Header file for cuRNN blas functions, whicha are simply structs
 *  with function pointers that call cublas functions, but can determine
 *  if the float or double version of the cublas functions should be called.
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
 *  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CURNN_BLAS_
#define _CURNN_BLAS_

#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
 * ========================================= NOTES ==========================================================
 * 1. The functions in this namespace are simply 'wrappers' that allow the cublas functions to be called in
 *    templated functions in other part of the cuRNN library, they are not my own implementation of these
 *    functions
 * ==========================================================================================================
 */

namespace curnn {
namespace blas  {
/*
 * ==========================================================================================================
 * Struct       : functions 
 *
 * Description  : Struct that holds function pointer to cublas functions for different data types
 * 
 * Params       : dType     : The type of data the function must use
 * ==========================================================================================================
 */
template <typename dType> struct functions;

// Partial specification for single precision cublas functions 
// (See cublas API reference for details)
template <> struct functions<float> {
    
    // Matix vector multiplication
    typedef cublasStatus_t (*fpgemv)( cublasHandle_t  , cublasOperation_t  , int     , int           ,   
                                      const float*    , const float*       , int     , const float*  ,
                                      int             , const float*       , float*  , int           );
    static constexpr fpgemv gemv = &cublasSgemv;
    
    // A*X plus Y
    typedef cublasStatus_t (*fpaxpy)( cublasHandle_t  , int     , const float*  , const float*  ,
                                      int             , float*  , int                           );
    static constexpr fpaxpy axpy = &cublasSaxpy;

};

// Partial specification for double precision cublas functions
template <> struct functions<double> {

    // Matix vector multiplication
    typedef cublasStatus_t (*fpgemv)( cublasHandle_t  , cublasOperation_t  , int     , int           ,   
                                      const double*   , const double*      , int     , const double* ,
                                      int             , const double*      , double* , int           );
    static constexpr fpgemv gemv = &cublasDgemv;
    
    // A*X plus Y
    typedef cublasStatus_t (*fpaxpy)( cublasHandle_t  , int     , const double*  , const double*    ,
                                      int             , double* , int                               );
    static constexpr fpaxpy axpy = &cublasDaxpy;
};

}
}


#endif
