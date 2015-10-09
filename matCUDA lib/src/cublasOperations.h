#ifndef CUBLASOPERATIONS_H
#define CUBLASOPERATIONS_H

#include <cublas.h>
#include <cublas_v2.h>
#include <cublasXt.h>

#include "matCUDA.h"

#ifndef BATCHSIZE
#define BATCHSIZE 1
#endif

#ifndef EIG_MAX_ITER
#define EIG_MAX_ITER 500
#endif

namespace matCUDA
{
	template <typename TElement>
	class cublasOperations
	{	
		template <typename TElement> friend class Array;

	public:

		// C = A x B
		cublasStatus_t multiply( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C );
		cublasStatus_t multiply_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C ); // TODO

		// C = A + B
		cublasStatus_t add( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string );
		cublasStatus_t add_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string );

		// transpose
		cublasStatus_t transpose( Array<TElement> *A, Array<TElement> *C );
		cublasStatus_t transpose_zerocopy( Array<TElement> *A, Array<TElement> *C );

		// conjugate
		cublasStatus_t conjugate( Array<TElement> *A );
		cublasStatus_t conjugate_zerocopy( Array<TElement> *A ); // TODO

		// hermitian
		cublasStatus_t hermitian( Array<TElement> *A, Array<TElement> *C );
		cublasStatus_t hermitian_zerocopy( Array<TElement> *A, Array<TElement> *C );

		// determinant
		cublasStatus_t determinant( Array<TElement> *A );

		// invert
		cublasStatus_t invert( Array<TElement> *result, Array<TElement> *data );
		cublasStatus_t invert_zerocopy( Array<TElement> *result, Array<TElement> *data );

		// LU decomposition
		cublasStatus_t LU( Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot );
		cublasStatus_t LU( Array<TElement> *A, Array<TElement> *LU );

		// least square solution
		cublasStatus_t LS( Array<TElement> *A, Array<TElement> *x, Array<TElement> *C );
		cublasStatus_t LS_zerocopy( Array<TElement> *A, Array<TElement> *x, Array<TElement> *C );

		// QR decomposition
		cublasStatus_t QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R );

		// eigenvector-eigenvalue
		cublasStatus_t eig( Array<TElement> *A, Array<TElement> *eigenvectors );

	private:

		cublasStatus_t cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const TElement *alpha, const TElement *A, int lda, const TElement *beta, const TElement *B, int ldb, TElement *C, int ldc);
		cublasStatus_t cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const TElement *alpha, const TElement *A, int lda, const TElement *x, int incx, const TElement *beta, TElement *y, int incy );
		cublasStatus_t cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const TElement *alpha, const TElement *A, int lda, const TElement *B, int ldb, const TElement *beta, TElement *C, int ldc);
		cublasStatus_t cublasTger( cublasHandle_t handle, int m, int n, const TElement *alpha, const TElement *x, int incx, const TElement *y, int incy, TElement *A, int lda );
		cublasStatus_t cublasTgetrfBatched( cublasHandle_t handle, int n, TElement *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize );
		cublasStatus_t cublasTgetriBatched( cublasHandle_t handle, int n, const TElement *Aarray[], int lda, int *PivotArray, TElement *Carray[], int ldc, int *infoArray, int batchSize );
		cublasStatus_t cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, TElement *Aarray[], int lda, TElement *Carray[], int ldc, int *info, int *devInfoArray, int batchSize );
		cublasStatus_t cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, TElement *Aarray[], int lda, TElement *TauArray[], int *info, int batchSize );
		cublasStatus_t cublasTdot( cublasHandle_t handle, int n, const TElement *x, int incx, const TElement *y, int incy, TElement *result );
		cublasStatus_t cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const TElement *alpha, const TElement *A, int lda, const TElement *B, int ldb, const TElement *beta, TElement *C, int ldc);
		
		// matrix x matrix = matrix (some experiments)
		cublasStatus_t multiply_MatrixMatrix_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_stream( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_Xt( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		
		// vector(transp) x vector = scalar
		cublasStatus_t multiply_VectorTranspVector_Scalar( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		
		// vector x vector(transp) = matrix
		cublasStatus_t multiply_VectorVectorTransp_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		
		// matrix x vector = vector 
		cublasStatus_t multiply_MatrixVector_Vector( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		
		// vector(transp) x matrix = vector(transp) 
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		
		// conjugate
		cublasStatus_t conjugate_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size );
		cublasStatus_t conjugate_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size );

		// define operation parameters
		void define_sum_subtract_transpose_hermitian_operation( TElement &alpha, TElement &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID );
	
		// build permutation matrix from cublas permutation vector
		void from_permutation_vector_to_permutation_matrix( Array<TElement> *pivotMatrix, Array<int> *pivotVector );
	};
}

#endif