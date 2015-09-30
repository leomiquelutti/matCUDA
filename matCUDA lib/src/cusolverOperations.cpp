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
		CUSOLVER_CALL( cusolverSpTcsreigvsi( &handleCusolver, N, nnz, &descrA, d_cooVal, d_csrRowPtr, d_cooColIndex, max_lambda, d_eigenvector0, maxite, tol, d_mu, d_eigenvector ) );

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

	template <typename TElement>
	cusolverStatus_t cusolverOperations<TElement>::QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R )
	{
		cusolverDnHandle_t solver_handle;
		CUSOLVER_CALL( cusolverDnCreate(&solver_handle) );

		int M = A->getDim(0);
		int N = A->getDim(1);
		int minMN = std::min(M,N);

		TElement *d_A, *h_A, *TAU, *Workspace, *d_Q;
		h_A = A->m_data.GetElements();
		CUDA_CALL( cudaMalloc(&d_A, M * N * sizeof(TElement)) );
		CUDA_CALL( cudaMemcpy(d_A, h_A, M * N * sizeof(TElement), cudaMemcpyHostToDevice) );

		int work_size = 0;
		CUSOLVER_CALL( cusolverDnTgeqrf_bufferSize(&solver_handle, M, N, d_A, M, &work_size) );
		CUDA_CALL( cudaMalloc(&TAU, minMN * sizeof(TElement)) );
		CUDA_CALL(cudaMalloc(&Workspace, work_size * sizeof(TElement)));

		int *devInfo;
		CUDA_CALL( cudaMalloc(&devInfo, sizeof(int)) );
		CUDA_CALL( cudaMemset( (void*)devInfo, 0, 1 ) );

		CUSOLVER_CALL( cusolverDnTgeqrf(&solver_handle, M, N, d_A, M, TAU, Workspace, work_size, devInfo) );
		
		int devInfo_h = 0;  
		CUDA_CALL( cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
		if (devInfo_h != 0)
			return CUSOLVER_STATUS_INTERNAL_ERROR;

		// CALL CUDA FUNCTION
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		for(int j = 0; j < M; j++)
			for(int i = j + 1; i < N; i++)
				(*R)(i,j) = 0;

		// --- Initializing the output Q matrix (Of course, this step could be done by a kernel function directly on the device)
		//*Q = eye<TElement> ( std::min(Q->getDim(0),Q->getDim(1)) );
		CUDA_CALL( cudaMalloc(&d_Q, M*M*sizeof(TElement)) );
		cudaEye<TElement>( d_Q, std::min(Q->getDim(0),Q->getDim(1)) );

		// --- CUDA QR execution
		CUSOLVER_CALL( cusolverDnTormqr(&solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, M, N, std::min(M, N), d_A, M, TAU, d_Q, M, Workspace, work_size, devInfo) );

		// --- At this point, d_Q contains the elements of Q. Showing this.
		CUDA_CALL( cudaMemcpy(Q->m_data.GetElements(), d_Q, M*M*sizeof(TElement), cudaMemcpyDeviceToHost) );
		CUDA_CALL( cudaDeviceSynchronize() );
  
		CUSOLVER_CALL( cusolverDnDestroy(solver_handle) );

		return CUSOLVER_STATUS_SUCCESS;
	}
	
	template cusolverStatus_t cusolverOperations<int>::QR( Array<int> *A, Array<int> *Q, Array<int> *R );
	template cusolverStatus_t cusolverOperations<float>::QR( Array<float> *A, Array<float> *Q, Array<float> *R );
	template cusolverStatus_t cusolverOperations<double>::QR( Array<double> *A, Array<double> *Q, Array<double> *R );
	template cusolverStatus_t cusolverOperations<ComplexFloat>::QR( Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R );
	template cusolverStatus_t cusolverOperations<ComplexDouble>::QR( Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R );
	
	template <typename TElement>
	cusolverStatus_t cusolverOperations<TElement>::QR_zerocopy( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R )
	{
		cusolverDnHandle_t solver_handle;
		CUSOLVER_CALL( cusolverDnCreate(&solver_handle) );

		int M = A->getDim(0);
		int N = A->getDim(1);
		int minMN = std::min(M,N);

		TElement *d_A, *h_A, *TAU, *Workspace, *d_Q, *d_R;
		h_A = A->m_data.GetElements();
		CUDA_CALL( cudaMalloc(&d_A, M * N * sizeof(TElement)) );
		CUDA_CALL( cudaMemcpy(d_A, h_A, M * N * sizeof(TElement), cudaMemcpyHostToDevice) );

		int work_size = 0;
		CUSOLVER_CALL( cusolverDnTgeqrf_bufferSize(&solver_handle, M, N, d_A, M, &work_size) );
		CUDA_CALL( cudaMalloc(&TAU, minMN * sizeof(TElement)) );
		CUDA_CALL(cudaMalloc(&Workspace, work_size * sizeof(TElement)));

		int *devInfo;
		CUDA_CALL( cudaMalloc(&devInfo, sizeof(int)) );
		CUDA_CALL( cudaMemset( (void*)devInfo, 0, 1 ) );

		CUSOLVER_CALL( cusolverDnTgeqrf(&solver_handle, M, N, d_A, M, TAU, Workspace, work_size, devInfo) );
		
		int devInfo_h = 0;  
		CUDA_CALL( cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
		if (devInfo_h != 0)
			return CUSOLVER_STATUS_INTERNAL_ERROR;

		// CALL CUDA FUNCTION
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_R, R->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_Q, Q->m_data.GetElements(), 0 ) );

		zeros_under_diag<TElement>( d_R, std::min(R->getDim(0),R->getDim(1)) );
		//for(int j = 0; j < M; j++)
		//	for(int i = j + 1; i < N; i++)
		//		(*R)(i,j) = 0;

		// --- Initializing the output Q matrix (Of course, this step could be done by a kernel function directly on the device)
		//*Q = eye<TElement> ( std::min(Q->getDim(0),Q->getDim(1)) );
		cudaEye<TElement>( d_Q, std::min(Q->getDim(0),Q->getDim(1)) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// --- CUDA QR_zerocopy execution
		CUSOLVER_CALL( cusolverDnTormqr(&solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, M, N, std::min(M, N), d_A, M, TAU, d_Q, M, Workspace, work_size, devInfo) );
		CUDA_CALL( cudaDeviceSynchronize() );
  
		CUSOLVER_CALL( cusolverDnDestroy(solver_handle) );

		return CUSOLVER_STATUS_SUCCESS;
	}
	
	template cusolverStatus_t cusolverOperations<int>::QR_zerocopy( Array<int> *A, Array<int> *Q, Array<int> *R );
	template cusolverStatus_t cusolverOperations<float>::QR_zerocopy( Array<float> *A, Array<float> *Q, Array<float> *R );
	template cusolverStatus_t cusolverOperations<double>::QR_zerocopy( Array<double> *A, Array<double> *Q, Array<double> *R );
	template cusolverStatus_t cusolverOperations<ComplexFloat>::QR_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R );
	template cusolverStatus_t cusolverOperations<ComplexDouble>::QR_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R );
	
	cusolverStatus_t cusolverOperations<float>::cusolverSpTcsreigvsi( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float mu0, const float *x0, int maxite, float tol, float *mu, float *x )
	{
		return cusolverSpScsreigvsi( *handle, m ,nnz, *descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x );;
	}

	cusolverStatus_t cusolverOperations<double>::cusolverSpTcsreigvsi( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double mu0, const double *x0, int maxite, double tol, double *mu, double *x )
	{
		return cusolverSpDcsreigvsi( *handle, m ,nnz, *descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0, maxite, tol, mu, x );
	}
	
	cusolverStatus_t cusolverOperations<ComplexFloat>::cusolverSpTcsreigvsi( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const ComplexFloat *csrValA, const int *csrRowPtrA, const int *csrColIndA, ComplexFloat mu0, const ComplexFloat *x0, int maxite, ComplexFloat tol, ComplexFloat *mu, ComplexFloat *x )
	{
		cuFloatComplex mu02 = make_cuFloatComplex( mu0.real(), mu0.imag() );
		return cusolverSpCcsreigvsi( *handle, m ,nnz, *descrA, (const cuFloatComplex*)csrValA, csrRowPtrA, csrColIndA, mu02, (const cuFloatComplex*)x0, maxite, tol.real(), (cuFloatComplex*)mu, (cuFloatComplex*)x );
	}
	
	cusolverStatus_t cusolverOperations<ComplexDouble>::cusolverSpTcsreigvsi( cusolverSpHandle_t *handle, int m, int nnz, cusparseMatDescr_t *descrA, const ComplexDouble *csrValA, const int *csrRowPtrA, const int *csrColIndA, ComplexDouble mu0, const ComplexDouble *x0, int maxite, ComplexDouble tol, ComplexDouble *mu, ComplexDouble *x )
	{
		cuDoubleComplex mu02 = make_cuDoubleComplex( mu0.real(), mu0.imag() );
		return cusolverSpZcsreigvsi( *handle, m ,nnz, *descrA, (const cuDoubleComplex*)csrValA, csrRowPtrA, csrColIndA, mu02, (const cuDoubleComplex*)x0, maxite, tol.real(), (cuDoubleComplex*)mu, (cuDoubleComplex*)x );
	}

	cusolverStatus_t cusolverOperations<int>::cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, int *A, int lda, int *Lwork )
	{
		return CUSOLVER_STATUS_SUCCESS;
	}	

	cusolverStatus_t cusolverOperations<float>::cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, float *A, int lda, int *Lwork )
	{
		return cusolverDnSgeqrf_bufferSize( *handle, m, n, A, lda, Lwork );
	}	

	cusolverStatus_t cusolverOperations<double>::cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, double *A, int lda, int *Lwork )
	{
		return cusolverDnDgeqrf_bufferSize( *handle, m, n, A, lda, Lwork );
	}	

	cusolverStatus_t cusolverOperations<ComplexFloat>::cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, ComplexFloat *A, int lda, int *Lwork )
	{
		return cusolverDnCgeqrf_bufferSize( *handle, m, n, (cuFloatComplex*)A, lda, Lwork );
	}	

	cusolverStatus_t cusolverOperations<ComplexDouble>::cusolverDnTgeqrf_bufferSize( cusolverDnHandle_t *handle, int m, int n, ComplexDouble *A, int lda, int *Lwork )
	{
		return cusolverDnZgeqrf_bufferSize( *handle, m, n, (cuDoubleComplex*)A, lda, Lwork );
	}	

	cusolverStatus_t cusolverOperations<int>::cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, int *A, int lda, int *TAU, int *Workspace, int Lwork, int *devInfo )
	{
		return CUSOLVER_STATUS_SUCCESS;
	}	

	cusolverStatus_t cusolverOperations<float>::cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo )
	{
		return cusolverDnSgeqrf( *handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<double>::cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo )
	{
		return cusolverDnDgeqrf( *handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<ComplexFloat>::cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, ComplexFloat *A, int lda, ComplexFloat *TAU, ComplexFloat *Workspace, int Lwork, int *devInfo )
	{
		return cusolverDnCgeqrf( *handle, m, n, (cuFloatComplex*)A, lda, (cuFloatComplex*)TAU, (cuFloatComplex*)Workspace, Lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<ComplexDouble>::cusolverDnTgeqrf( cusolverDnHandle_t *handle, int m, int n, ComplexDouble *A, int lda, ComplexDouble *TAU, ComplexDouble *Workspace, int Lwork, int *devInfo )
	{
		return cusolverDnZgeqrf( *handle, m, n, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)TAU, (cuDoubleComplex*)Workspace, Lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<int>::cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const int *A, int lda, const int *tau, int *C, int ldc, int *work, int lwork, int *devInfo )
	{
		return CUSOLVER_STATUS_SUCCESS;
	}	

	cusolverStatus_t cusolverOperations<float>::cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc, float *work, int lwork, int *devInfo )
	{
		return cusolverDnSormqr( *handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<double>::cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *tau, double *C, int ldc, double *work, int lwork, int *devInfo )
	{
		return cusolverDnDormqr( *handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<ComplexFloat>::cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const ComplexFloat *A, int lda, const ComplexFloat *tau, ComplexFloat *C, int ldc, ComplexFloat *work, int lwork, int *devInfo )
	{
		return cusolverDnCunmqr( *handle, side, trans, m, n, k, (const cuFloatComplex*)A, lda, (const cuFloatComplex*)tau, (cuFloatComplex*)C, ldc, (cuFloatComplex*)work, lwork, devInfo );
	}	

	cusolverStatus_t cusolverOperations<ComplexDouble>::cusolverDnTormqr( cusolverDnHandle_t *handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const ComplexDouble *A, int lda, const ComplexDouble *tau, ComplexDouble *C, int ldc, ComplexDouble *work, int lwork, int *devInfo )
	{
		return cusolverDnZunmqr( *handle, side, trans, m, n, k, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)tau, (cuDoubleComplex*)C, ldc, (cuDoubleComplex*)work, lwork, devInfo );
	}	
}