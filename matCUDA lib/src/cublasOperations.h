#ifndef CUBLASOPERATIONS_H
#define CUBLASOPERATIONS_H

#include <cublas.h>
#include <cublas_v2.h>

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

		// C = A + B
		cublasStatus_t add( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C );
		cublasStatus_t add2( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string );

		// C = A - B
		cublasStatus_t minus( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C );

		// transpose
		cublasStatus_t transpose( Array<TElement> *A );

		// conjugate
		cublasStatus_t conjugate( Array<TElement> *A );

		// hermitian
		cublasStatus_t hermitian( Array<TElement> *A );

		// determinant
		cublasStatus_t determinant( Array<TElement> *A );

		// invert
		cublasStatus_t invert( Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot );

		// LU decomposition
		cublasStatus_t LU( Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot );

		// least square solution
		cublasStatus_t LS( Array<TElement> A, Array<TElement> *x, Array<TElement> C );

		// QR decomposition
		cublasStatus_t QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R );

		// eigenvector-eigenvalue
		cublasStatus_t eig( Array<TElement> *A, Array<TElement> *eigenvectors );

	private:

		cublasStatus_t sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const int *alpha, const int *A, const int *beta, const int *B, int *C, _matrixSize matrix_size);
		cublasStatus_t sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const float *alpha, const float *A, const float *beta, const float *B, float *C, _matrixSize matrix_size);
		cublasStatus_t sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const double *alpha, const double *A, const double *beta, const double *B, double *C, _matrixSize matrix_size);
		cublasStatus_t sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const ComplexFloat *alpha, const ComplexFloat *A, const ComplexFloat *beta, const ComplexFloat *B, ComplexFloat *C, _matrixSize matrix_size);
		cublasStatus_t sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const ComplexDouble *alpha, const ComplexDouble *A, const ComplexDouble *beta, const ComplexDouble *B, ComplexDouble *C, _matrixSize matrix_size);
	
		// matrix x matrix = matrix
		cublasStatus_t multiply_MatrixMatrix_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );

		// vector(transp) x vector = scalar
		cublasStatus_t multiply_VectorTranspVector_Scalar( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );

		// vector x vector(transp) = matrix
		cublasStatus_t multiply_VectorVectorTransp_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );

		// matrix x vector = vector OR vector(transp) x matrix = vector(transp)
		cublasStatus_t multiply_MatrixVector_Vector( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );

		// vector(transp) x matrix = vector(transp)
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );
		cublasStatus_t multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size );

		// sum and subtract of arrays
		cublasStatus_t sum_subtract_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size );
		cublasStatus_t sum_subtract_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size );
		cublasStatus_t sum_subtract_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size );
		cublasStatus_t sum_subtract_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size );
		cublasStatus_t sum_subtract_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size );

		// transpose and hermitian of arrays
		cublasStatus_t transpose_hermitian_int( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size );
		cublasStatus_t transpose_hermitian_float( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size );
		cublasStatus_t transpose_hermitian_double( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size );
		cublasStatus_t transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size );
		cublasStatus_t transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size );

		// conjugate
		cublasStatus_t conjugate_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size );
		cublasStatus_t conjugate_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size );

		// LU decomposition
		cublasStatus_t LU_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size );
		cublasStatus_t LU_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size );
		cublasStatus_t LU_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size );
		cublasStatus_t LU_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size );
		cublasStatus_t LU_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size );

		// inversion
		cublasStatus_t invert_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size );
		cublasStatus_t invert_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size );
		cublasStatus_t invert_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size );
		cublasStatus_t invert_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size );
		cublasStatus_t invert_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size );

		// LS solution
		cublasStatus_t LS_int( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size );
		cublasStatus_t LS_float( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size );
		cublasStatus_t LS_double( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size );
		cublasStatus_t LS_ComplexFloat( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> , _matrixSize matrix_size );
		cublasStatus_t LS_ComplexDouble( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size );

		// QR decomposition
		cublasStatus_t QR_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size );
		cublasStatus_t QR_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size );
		cublasStatus_t QR_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size );
		cublasStatus_t QR_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size );
		cublasStatus_t QR_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size );

		// eigenvalues/eigenvectors calculation
		cublasStatus_t eig_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size );
		cublasStatus_t eig_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size );
		cublasStatus_t eig_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size );
		cublasStatus_t eig_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size );
		cublasStatus_t eig_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size );
		
		// define operation parameters
		void define_sum_subtract_transpose_hermitian_operation( TElement &alpha, TElement &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID );
	
		// build permutation matrix from cublas permutation vector
		void from_permutation_vector_to_permutation_matrix( Array<TElement> *pivotMatrix, Array<int> *pivotVector );
	};
}

#endif