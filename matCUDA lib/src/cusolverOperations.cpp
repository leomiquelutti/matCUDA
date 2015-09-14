#include <boost\math\special_functions.hpp>

#include "common.h"

#include "cusolverOperations.h"

namespace matCUDA
{
	template<> cusolverStatus_t cusolverOperations<ComplexFloat>::dpss( Array<ComplexFloat> *eigenvector, index_t N, double NW, index_t degree )
	{
		return CUSOLVER_STATUS_NOT_INITIALIZED;
	}

	template<> cusolverStatus_t cusolverOperations<ComplexDouble>::dpss( Array<ComplexDouble> *eigenvector, index_t N, double NW, index_t degree )
	{
		return CUSOLVER_STATUS_NOT_INITIALIZED;
	}

	template <typename TElement>
	cusolverStatus_t cusolverOperations<TElement>::dpss( Array<TElement> *eigenvector, index_t N, double NW, index_t degree )
	{
		// define matrix T (NxN) 
		TElement** T = new TElement*[ N ];
		for(int i = 0; i < N; ++i)
			T[ i ] = new TElement[ N ];

		// fill in T as function of ( N, W ) 
		// T is a tridiagonal matrix, i. e., it has diagonal, subdiagonal and superdiagonal
		// the others elements are 0
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				if( j == i - 1 ) // subdiagonal
					T[ i ][ j ] = ( (TElement)N - i )*i/2;
				else if( j == i ) // diagonal
					T[ i ][ j ] = pow( (TElement)(N-1)/2 - i, 2 )*boost::math::cos_pi( 2*NW/(TElement)N/boost::math::constants::pi<TElement>() );
				else if( j == i + 1 ) // superdiagonal
					T[ i ][ j ] = ( i + 1 )*( (TElement)N - 1 - i )/2*( j == i + 1 );
				else // others elements
					T[ i ][ j ] = 0;
			}
		}
	
		// declarations needed
		cusolverStatus_t statCusolver = CUSOLVER_STATUS_SUCCESS;
		cusolverSpHandle_t handleCusolver = NULL;
		cusparseHandle_t handleCusparse = NULL;
		cusparseMatDescr_t descrA = NULL;
		int *h_cooRowIndex = NULL, *h_cooColIndex = NULL;
		TElement *h_cooVal = NULL; 
		int *d_cooRowIndex = NULL, *d_cooColIndex = NULL, *d_csrRowPtr = NULL; 
		TElement *d_cooVal = NULL; 
		int nnz; 
		TElement *h_eigenvector0 = NULL, *d_eigenvector0 = NULL, *d_eigenvector = NULL;
		int maxite = 1e6; // number of maximum iteration
		TElement tol = 1; // tolerance
		TElement mu, *d_mu;
		TElement max_lambda;

		// define interval of eigenvalues of T
		// interval is [-max_lambda,max_lambda]
		max_lambda = ( N - 1 )*( N + 2 ) + N*( N + 1 )/8 + 0.25;
	
		// amount of nonzero elements of T
		nnz = 3*N - 2;

		// allocate host memory
		h_cooRowIndex = new int[ nnz*sizeof( int ) ];
		h_cooColIndex = new int[ nnz*sizeof( int ) ];
		h_cooVal = new TElement[ nnz*sizeof( TElement ) ];
		h_eigenvector0 = new TElement[ N*sizeof( TElement ) ];

		// fill in vectors that describe T as a sparse matrix
		int counter = 0;
		for (int i = 0; i < N; i++ ) {
			for( int j = 0; j < N; j++ ) {
				if( T[ i ][ j ] != 0 ) {
					h_cooRowIndex[counter] = i;
					h_cooColIndex[counter] = j;
					h_cooVal[counter++] = T[ i ][ j ];
				}
			}
		}
	
		// fill in initial eigenvector guess  
		for( int i = 0; i < N; i++ )
			h_eigenvector0[ i ] =  1/( abs( i - N/2 ) + 1 );

		// allocate device memory
		CUDA_CALL( cudaMalloc((void**)&d_cooRowIndex,nnz*sizeof( int )) ); 
		CUDA_CALL( cudaMalloc((void**)&d_cooColIndex,nnz*sizeof( int )) ); 
		CUDA_CALL( cudaMalloc((void**)&d_cooVal, nnz*sizeof( TElement )) );
		CUDA_CALL( cudaMalloc((void**)&d_csrRowPtr, (N+1)*sizeof( int )) );
		CUDA_CALL( cudaMalloc((void**)&d_eigenvector0, N*sizeof( TElement )) );
		CUDA_CALL( cudaMalloc((void**)&d_eigenvector, N*sizeof( TElement )) );
		CUDA_CALL( cudaMalloc( &d_mu, sizeof( TElement ) ) );
		CUDA_CALL( cudaMemset( d_mu, -max_lambda, sizeof( TElement ) ) );

		// copy data to device
		CUDA_CALL( cudaMemcpy( d_cooRowIndex, h_cooRowIndex, (size_t)(nnz*sizeof( int )), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_cooColIndex, h_cooColIndex, (size_t)(nnz*sizeof( int )), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_cooVal, h_cooVal, (size_t)(nnz*sizeof( TElement )), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_eigenvector0, h_eigenvector0, (size_t)(N*sizeof( TElement )), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( &mu, d_mu, sizeof( TElement ), cudaMemcpyDeviceToHost ) );
	
		// initialize cusparse and cusolver
		CUSOLVER_CALL( cusolverSpCreate( &handleCusolver ) );
		CUSPARSE_CALL( cusparseCreate( &handleCusparse ) );

		// create and define cusparse matrix descriptor
		CUSPARSE_CALL( cusparseCreateMatDescr(&descrA) );
		CUSPARSE_CALL( cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL ) );
		CUSPARSE_CALL( cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO ) );

		// transform from coordinates (COO) values to compressed row pointers (CSR) values
		CUSPARSE_CALL( cusparseXcoo2csr( handleCusparse, d_cooRowIndex, nnz, N, d_csrRowPtr, CUSPARSE_INDEX_BASE_ZERO ) );
	
		// call cusolverSp<type>csreigvsi
		CUSOLVER_CALL( dpss_type( &handleCusolver, N, nnz, &descrA, d_cooVal, d_csrRowPtr, d_cooColIndex, max_lambda, d_eigenvector0, maxite, tol, d_mu, d_eigenvector ) );

		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from device to host
		CUDA_CALL( cudaMemcpy( &mu, d_mu, (size_t)sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( eigenvector->m_data.GetElements(), d_eigenvector, (size_t)(N*sizeof( TElement )), cudaMemcpyDeviceToHost ) );
	
		// destroy and free stuff
		CUSPARSE_CALL( cusparseDestroyMatDescr( descrA ) );
		CUSPARSE_CALL( cusparseDestroy( handleCusparse ) );
		CUSOLVER_CALL( cusolverSpDestroy( handleCusolver ) );
		CUDA_CALL( cudaFree( d_cooRowIndex ) );
		CUDA_CALL( cudaFree( d_cooColIndex ) );
		CUDA_CALL( cudaFree( d_cooVal ) );
		CUDA_CALL( cudaFree( d_csrRowPtr ) );
		CUDA_CALL( cudaFree( d_eigenvector0 ) );
		CUDA_CALL( cudaFree( d_eigenvector ) );
		CUDA_CALL( cudaFree( d_mu ) );
		delete[] h_eigenvector0;
		delete[] h_cooRowIndex;
		delete[] h_cooColIndex;
		delete[] h_cooVal;

		return CUSOLVER_STATUS_SUCCESS;
	}

	template cusolverStatus_t cusolverOperations<float>::dpss( Array<float> *eigenvector, index_t N, double NW, index_t degree );
	template cusolverStatus_t cusolverOperations<double>::dpss( Array<double> *eigenvector, index_t N, double NW, index_t degree );

	cusolverStatus_t cusolverOperations<float>::dpss_type( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float mu0, const float *d_eigenvector0, int maxite, float tol, float *d_mu, float *d_eigenvector )
	{
		return cusolverSpScsreigvsi( *handle, m ,nnz, *descrA, csrValA, csrRowPtrA, csrColIndA, mu0, d_eigenvector0, maxite, tol, d_mu, d_eigenvector );;
	}

	cusolverStatus_t cusolverOperations<double>::dpss_type( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double mu0, const double *d_eigenvector0, int maxite, double tol, double *d_mu, double *d_eigenvector )
	{
		return cusolverSpDcsreigvsi( *handle, m ,nnz, *descrA, csrValA, csrRowPtrA, csrColIndA, mu0, d_eigenvector0, maxite, tol, d_mu, d_eigenvector );
	}
}