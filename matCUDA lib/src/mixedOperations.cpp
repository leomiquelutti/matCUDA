#include "common.h"

#include "mixedOperations.h"

namespace matCUDA
{
	// eig decomposition

	template <typename TElement>
	cublasStatus_t mixedOperations<TElement>::eig( Array<TElement> *A, Array<TElement> *eigvec )
	{
		cusolverDnHandle_t handleCusolver;
		CUSOLVER_CALL( cusolverDnCreate(&handleCusolver) );

		cublasHandle_t handleCublas;
		CUBLAS_CALL( cublasCreate( &handleCublas ) );

		cusolverOperations<TElement> opCusolver;
		cublasOperations<TElement> opCublas;
		mixedOperations<TElement> test;

		int M = A->getDim(0);
		int N = A->getDim(1);
		int minMN = std::min(M,N);
		TElement alpha = 1, beta = 0;

		TElement *d_A, *TAU, *Workspace, *d_Q, *d_R;
		CUDA_CALL( cudaMalloc(&d_A, M * N * sizeof(TElement)) );
		CUDA_CALL( cudaMalloc(&d_Q, M * N * sizeof(TElement)) );
		CUDA_CALL( cudaMalloc(&d_R, M * N * sizeof(TElement)) );
		CUDA_CALL( cudaMemcpy(d_A, A->data(), M * N * sizeof(TElement), cudaMemcpyHostToDevice) );

		int Lwork = 0;
		CUSOLVER_CALL( opCusolver.cusolverDnTgeqrf_bufferSize(&handleCusolver, M, N, d_A, M, &Lwork) );
		CUDA_CALL( cudaMalloc(&TAU, minMN * sizeof(TElement)) );
		CUDA_CALL(cudaMalloc(&Workspace, Lwork * sizeof(TElement)));

		int *devInfo;
		CUDA_CALL( cudaMalloc(&devInfo, sizeof(int)) );
		CUDA_CALL( cudaMemset( (void*)devInfo, 0, 1 ) );

		Array<TElement> aux( M, N );
		aux = 0;
		aux.print();
		CUDA_CALL( cudaMemcpy(d_Q, aux.data(), M * N * sizeof(TElement), cudaMemcpyHostToDevice) );

		int devInfo_h = 0;  
		for( int i = 0; i < EIG_MAX_ITER; i++ ) {

			CUSOLVER_CALL( opCusolver.cusolverDnTgeqrf(&handleCusolver, M, N, d_A, M, TAU, Workspace, Lwork, devInfo) );
		
			CUDA_CALL( cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
			if (devInfo_h != 0)
				return CUBLAS_STATUS_INTERNAL_ERROR;

			// evaluate R
			CUDA_CALL( cudaMemcpy( d_R, d_A, M * N * sizeof(TElement), cudaMemcpyDeviceToDevice ) );
			zeros_under_diag( d_R, std::min( M, N ) );

			// evaluate Q
			cuda_eye<TElement>( d_Q, minMN );
			CUSOLVER_CALL( opCusolver.cusolverDnTormqr(&handleCusolver, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, M, N, std::min(M, N), d_A, M, TAU, d_Q, M, Workspace, Lwork, devInfo) );
		
			// checkout
			CUDA_CALL( cudaMemcpy( aux.data(), d_Q, M * N * sizeof(TElement), cudaMemcpyDeviceToHost ) );
			aux.print();
			
			// evaluate new A = R*Q
			CUBLAS_CALL( opCublas.cublasTgemm(handleCublas, CUBLAS_OP_N, CUBLAS_OP_N, minMN, minMN, minMN, &alpha, d_R, minMN, d_Q, minMN, &beta, d_A, minMN) );
		
			//// checkout
			//CUDA_CALL( cudaMemcpy( aux.data(), d_R, M * N * sizeof(TElement), cudaMemcpyDeviceToHost ) );
			//aux.print();

		}
		CUDA_CALL( cudaDeviceSynchronize() );

		CUDA_CALL( cudaMemcpy( A->data(), d_A, M * N * sizeof(TElement), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( eigvec->data(), d_Q, M * N * sizeof(TElement), cudaMemcpyDeviceToHost ) );

		CUDA_CALL( cudaFree( d_A ) );
		CUDA_CALL( cudaFree( d_Q ) );
		CUDA_CALL( cudaFree( d_R ) );
  
		CUSOLVER_CALL( cusolverDnDestroy(handleCusolver) );
		CUBLAS_CALL( cublasDestroy(handleCublas) );

		return CUBLAS_STATUS_SUCCESS;
	}
	
	template cublasStatus_t mixedOperations<int>::eig( Array<int> *A, Array<int> *eigenvectors );
	template cublasStatus_t mixedOperations<float>::eig( Array<float> *A, Array<float> *eigenvectors );
	template cublasStatus_t mixedOperations<double>::eig( Array<double> *A, Array<double> *eigenvectors );
	template cublasStatus_t mixedOperations<ComplexFloat>::eig( Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors );
	template cublasStatus_t mixedOperations<ComplexDouble>::eig( Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors );
}