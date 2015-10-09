#include <boost/exception/all.hpp>
#include <boost/pointer_cast.hpp>
#include <iostream>

#include "common.h"

#include "cublasOperations.h"

namespace matCUDA
{
	// eig decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig( Array<TElement> *A, Array<TElement> *eigenvectors )
	{
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( A->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_INITIALIZED;
		if( A->GetDescriptor().GetDim(0) != A->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_INITIALIZED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		TElement *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(TElement)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const TElement *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// define device memory
		int *PivotArray;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const TElement *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(TElement*), cudaMemcpyHostToDevice) );
	
		TElement alpha = 1, beta = 0;
		int INFOh = 2;

		for( int i = 0; i < EIG_MAX_ITER; i++ )
		{
			// CALL CUBLAS FUNCTION - QR decomposition
			CUBLAS_CALL( cublasTgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

			cudaDeviceSynchronize();
			CUDA_CALL( cudaGetLastError() );

			// info from GPU
			if( infoArray < 0 )
			{
				printf("Parameter invalid for cublas<type>geqrfBatched: %d\n", -infoArray);
				cublasShutdown();
				return CUBLAS_STATUS_EXECUTION_FAILED;
			}

			// CALL CUDA FUNCTION
			zeros_under_diag<TElement> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );

			// CALL CUBLAS FUNCTION
			CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
			CUDA_CALL( cudaDeviceSynchronize() );

			// copy from GPU
			CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

			if( INFOh == lda )
			{
				printf("Factorization Failed: Matrix is singular\n");
				cublasShutdown();
				return CUBLAS_STATUS_EXECUTION_FAILED;
			}

			CUBLAS_CALL( cublasTgetriBatched( handle, matrix_size.Arows, (const TElement**)Aarray, lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

			CUBLAS_CALL( cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( d_aux ) );
		CUBLAS_CALL( cublasFree( d_tau ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( tauArray ) );
		CUBLAS_CALL( cublasFree( auxArray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::eig( Array<int> *A, Array<int> *eigenvectors );
	template cublasStatus_t cublasOperations<float>::eig( Array<float> *A, Array<float> *eigenvectors );
	template cublasStatus_t cublasOperations<double>::eig( Array<double> *A, Array<double> *eigenvectors );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig( Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig( Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors );

	// QR decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R )
	{
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( A->GetDescriptor().GetNDim() != 2 ||
			Q->GetDescriptor().GetNDim() != 2 ||
			R->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_INITIALIZED;
		if( A->GetDescriptor().GetDim(0) < A->GetDescriptor().GetDim(1) ||
			Q->GetDescriptor().GetDim(0) < Q->GetDescriptor().GetDim(1) ||
			R->GetDescriptor().GetDim(0) < R->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_INITIALIZED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = Q->GetDescriptor().GetDim(1);
		matrix_size.Brows = Q->GetDescriptor().GetDim(0);
		matrix_size.Ccols = R->GetDescriptor().GetDim(1);
		matrix_size.Crows = R->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));

		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		TElement *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(TElement)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const TElement *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION - QR decomposition
		CUBLAS_CALL( cublasTgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// info from GPU
		if( infoArray < 0 )
		{
			printf("Parameter invalid for cublasTgeqrfBatched: %d\n", -infoArray);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		//Array<TElement> tau( size_tau );
		//CUDA_CALL( cudaMemcpy( tau.m_data.GetElements(), d_tau, tau.m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		//CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// CALL CUDA FUNCTION
		zeros_under_diag<TElement> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// define device memory
		int *PivotArray;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const TElement *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasTgetriBatched( handle, matrix_size.Arows, (const TElement**)Aarray, lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

		TElement alpha = 1, beta = 0;
		CUBLAS_CALL( cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );

		CUDA_CALL( cudaMemcpy( Q->m_data.GetElements(), d_A, Q->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( d_aux ) );
		CUBLAS_CALL( cublasFree( d_tau ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( tauArray ) );
		CUBLAS_CALL( cublasFree( auxArray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );


				// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::QR( Array<int> *A, Array<int> *Q, Array<int> *R );
	template cublasStatus_t cublasOperations<float>::QR( Array<float> *A, Array<float> *Q, Array<float> *R );
	template cublasStatus_t cublasOperations<double>::QR( Array<double> *A, Array<double> *Q, Array<double> *R );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR( Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR( Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R );

	// least square solution

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS( Array<TElement> *A, Array<TElement> *x, Array<TElement> *C )
	{	
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( A->GetDescriptor().GetNDim() != 2 ||
			C->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_INITIALIZED;
		if( A->GetDescriptor().GetDim(0) < A->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_INITIALIZED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));

		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		//save matrix address 
		const TElement *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), C->m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		//C->m_padded = false;
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = (*C)( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );
	
		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::LS( Array<int> *A, Array<int> *x, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::LS( Array<float> *A, Array<float> *x, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::LS( Array<double> *A, Array<double> *x, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS( Array<ComplexFloat> *A, Array<ComplexFloat> *x, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS( Array<ComplexDouble> *A, Array<ComplexDouble> *x, Array<ComplexDouble> *C );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_zerocopy( Array<TElement> *A, Array<TElement> *x, Array<TElement> *C )
	{	
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( A->GetDescriptor().GetNDim() != 2 ||
			C->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_INITIALIZED;
		if( A->GetDescriptor().GetDim(0) < A->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_INITIALIZED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));

		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		//CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		// pass host pointer to device
		//CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->data(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->data(), 0 ) );

		//save matrix address 
		const TElement *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );
		//CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), C->m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		//C->m_padded = false;
		//CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = (*C)( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		//CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );
	
		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::LS_zerocopy( Array<int> *A, Array<int> *x, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::LS_zerocopy( Array<float> *A, Array<float> *x, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::LS_zerocopy( Array<double> *A, Array<double> *x, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *x, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *x, Array<ComplexDouble> *C );

	// implementation on inversion

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert( Array<TElement> *result, Array<TElement> *data )
	{	
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( result->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_SUPPORTED;
		if( result->GetDescriptor().GetDim(0) != result->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_SUPPORTED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = data->GetDescriptor().GetDim(1);
		matrix_size.Arows = data->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		TElement *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		const TElement *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), data->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}
	
		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetriBatched( handle, matrix_size.Arows, (const TElement**)Aarray, lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		//result->m_padded = false;
		CUDA_CALL( cudaMemcpy( result->m_data.GetElements(), d_C, result->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
		
	template cublasStatus_t cublasOperations<int>::invert( Array<int> *result, Array<int> *data );
	template cublasStatus_t cublasOperations<float>::invert( Array<float> *result, Array<float> *data );
	template cublasStatus_t cublasOperations<double>::invert( Array<double> *result, Array<double> *data );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert( Array<ComplexFloat> *result, Array<ComplexFloat> *data );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert( Array<ComplexDouble> *result, Array<ComplexDouble> *data );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_zerocopy( Array<TElement> *result, Array<TElement> *data )
	{	
		cublasHandle_t handle;
		cublasCreate_v2(&handle);

		if( result->GetDescriptor().GetNDim() != 2 )
			return CUBLAS_STATUS_NOT_SUPPORTED;
		if( result->GetDescriptor().GetDim(0) != result->GetDescriptor().GetDim(1) )
			return CUBLAS_STATUS_NOT_SUPPORTED;
	
		_matrixSize matrix_size;
		matrix_size.Acols = data->GetDescriptor().GetDim(1);
		matrix_size.Arows = data->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		TElement *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, result->m_data.GetElements(), 0 ) );

		//save matrix address 
		const TElement *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), data->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}
	
		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetriBatched( handle, matrix_size.Arows, (const TElement**)Aarray, lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		
		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
		
	template cublasStatus_t cublasOperations<int>::invert_zerocopy( Array<int> *result, Array<int> *data );
	template cublasStatus_t cublasOperations<float>::invert_zerocopy( Array<float> *result, Array<float> *data );
	template cublasStatus_t cublasOperations<double>::invert_zerocopy( Array<double> *result, Array<double> *data );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert_zerocopy( Array<ComplexFloat> *result, Array<ComplexFloat> *data );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert_zerocopy( Array<ComplexDouble> *result, Array<ComplexDouble> *data );

	// implementation of LU decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU( Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		//cublasCreate_v2(&handle);
		cudaError_t error;

		if( A->GetDescriptor().GetNDim() != 2 )
			return stat;
		if( A->GetDescriptor().GetDim(0) != A->GetDescriptor().GetDim(1) )
			return stat;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = LU->GetDescriptor().GetDim(1);
		matrix_size.Brows = LU->GetDescriptor().GetDim(0);
		matrix_size.Ccols = Pivot->GetDescriptor().GetDim(1);
		matrix_size.Crows = Pivot->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		TElement *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		TElement *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}
	
		Array<int> pivotVector( size_pivot );
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( pivotVector.m_data.GetElements(), PivotArray, size_pivot*sizeof( int ), cudaMemcpyDeviceToHost ) );

		from_permutation_vector_to_permutation_matrix( Pivot, &pivotVector );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::LU( Array<int> *A, Array<int> *LU, Array<int> *Pivot);
	template cublasStatus_t cublasOperations<float>::LU( Array<float> *A, Array<float> *LU, Array<float> *Pivot );
	template cublasStatus_t cublasOperations<double>::LU( Array<double> *A, Array<double> *LU, Array<double> *Pivot );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU( Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU( Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU( Array<TElement> *A, Array<TElement> *LU )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		//cublasCreate_v2(&handle);
		cudaError_t error;

		if( A->GetDescriptor().GetNDim() != 2 )
			return stat;
		if( A->GetDescriptor().GetDim(0) != A->GetDescriptor().GetDim(1) )
			return stat;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = LU->GetDescriptor().GetDim(1);
		matrix_size.Brows = LU->GetDescriptor().GetDim(0);

		CUBLAS_CALL(cublasCreate(&handle));
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		TElement *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(TElement*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		TElement *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(TElement*), cudaMemcpyHostToDevice) );
		
		/////***** performance test *****/////
		CUDA_CALL( cudaDeviceSynchronize() );
		tic();
		for( int i = 0; i < 1; i++ ) {

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );

		}
		CUDA_CALL( cudaDeviceSynchronize() );
		toc();
		////***** end of performance test *****/////

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}
	
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );
		
		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::LU( Array<int> *A, Array<int> *LU );
	template cublasStatus_t cublasOperations<float>::LU( Array<float> *A, Array<float> *LU );
	template cublasStatus_t cublasOperations<double>::LU( Array<double> *A, Array<double> *LU );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU( Array<ComplexFloat> *A, Array<ComplexFloat> *LU );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU( Array<ComplexDouble> *A, Array<ComplexDouble> *LU );

	// implementation of conjugate

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::conjugate( Array<TElement> *A )
	{	
		cublasHandle_t handle;
		cublasCreate( &handle ); 
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		Array<TElement> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		TElement alpha2 = TElement( 1.0, 0.0 );
		TElement beta2 = TElement( 0.0, 0.0 );
		cublasOperation_t op1 = CUBLAS_OP_T, op2 = CUBLAS_OP_T;

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		TElement *d_A, *d_B, *d_C;
		d_B = &alpha2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = lda;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha2, d_A, lda, &beta2, d_A, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		op1 = op2 = CUBLAS_OP_C;
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Acols, &alpha2, d_C, lda, &beta2, d_C, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );
	
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_A, A->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<ComplexFloat>::conjugate( Array<ComplexFloat> *A );
	template cublasStatus_t cublasOperations<ComplexDouble>::conjugate( Array<ComplexDouble> *A );
	
	// implementation of hermitian

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::hermitian( Array<TElement> *A, Array<TElement> *C )
	{
		cublasHandle_t handle;
		cublasCreate( &handle ); 
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , "H");	

		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		TElement *d_A, *d_B, *d_C;
		d_B = NULL;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );
	
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<ComplexFloat>::hermitian( Array<ComplexFloat> *A, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::hermitian( Array<ComplexDouble> *A, Array<ComplexDouble> *C );

	cublasStatus_t cublasOperations<int>::hermitian( Array<int> *A, Array<int> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<float>::hermitian( Array<float> *A, Array<float> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<double>::hermitian( Array<double> *A, Array<double> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::hermitian_zerocopy( Array<TElement> *A, Array<TElement> *C )
	{	
		cublasHandle_t handle;
		cublasCreate( &handle ); 
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , "H");	

		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		TElement *d_A, *d_B, *d_C;
		d_B = NULL;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<ComplexFloat>::hermitian_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::hermitian_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *C );

	cublasStatus_t cublasOperations<int>::hermitian_zerocopy( Array<int> *A, Array<int> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<float>::hermitian_zerocopy( Array<float> *A, Array<float> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<double>::hermitian_zerocopy( Array<double> *A, Array<double> *C )
	{
		*C = A->transpose();
		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of transpose
		
	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose( Array<TElement> *A, Array<TElement> *C )
	{	
		cublasHandle_t handle;
		cublasCreate( &handle ); 
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , "T");	

		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		TElement *d_A, *d_B, *d_C;
		d_B = NULL;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );
		//CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );
	
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
	
	template cublasStatus_t cublasOperations<int>::transpose( Array<int> *A, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::transpose( Array<float> *A, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::transpose( Array<double> *A, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose( Array<ComplexFloat> *A, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose( Array<ComplexDouble> *A, Array<ComplexDouble> *C );
		
	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_zerocopy( Array<TElement> *A, Array<TElement> *C )
	{	
		cublasHandle_t handle;
		cublasCreate( &handle ); 
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , "T");	

		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		TElement *d_A, *d_B, *d_C;
		d_B = NULL;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		//std::cout << C->getDim(0) << " " << C->getDim(1) << std:: endl;
		//C->m_padded = false;
		//C->GetDescriptor().Swap();
		//std::cout << C->getDim(0) << " " << C->getDim(1) << std:: endl;
		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
	
	template cublasStatus_t cublasOperations<int>::transpose_zerocopy( Array<int> *A, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::transpose_zerocopy( Array<float> *A, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::transpose_zerocopy( Array<double> *A, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *C );

	// implementation of add

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::add( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID )
	{
		cublasHandle_t handle;
		cublasCreate( &handle ); 

		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		// define memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows * sizeof( TElement );
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows * sizeof( TElement );
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows * sizeof( TElement );

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memory on the device
		CUDA_CALL( cudaMalloc( &d_A, size_A ) );
		CUDA_CALL( cudaMalloc( &d_B, size_B ) );
		CUDA_CALL( cudaMalloc( &d_C, size_C ) );
		
		// copy vectors to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->data(), size_A, cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->data(), size_B, cudaMemcpyHostToDevice ) );
		
		/////***** performance test *****/////
		//CUDA_CALL( cudaDeviceSynchronize() );
		//tic();
		//for( int i = 0; i < 10; i++ ) {
		
		// CALL CUBLAS FUNCTION
		CUBLAS_CALL(cublasTgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ));
		
		//}
		//CUDA_CALL( cudaDeviceSynchronize() );
		//toc();
		////***** end of performance test *****/////
		
		CUDA_CALL( cudaMemcpy( C->data(), d_C, size_C, cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		CUBLAS_CALL( cublasDestroy(handle) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::add( Array<int> *A, Array<int> *B, Array<int> *C, std::string ID );
	template cublasStatus_t cublasOperations<float>::add( Array<float> *A, Array<float> *B, Array<float> *C, std::string ID );
	template cublasStatus_t cublasOperations<double>::add( Array<double> *A, Array<double> *B, Array<double> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexFloat>::add( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexDouble>::add( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::add_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID )
	{
		cublasHandle_t handle;
		cublasCreate( &handle ); 

		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		TElement alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		// define memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL(cublasTgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ));
	
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaGetLastError());

		CUBLAS_CALL( cublasDestroy(handle) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::add_zerocopy( Array<int> *A, Array<int> *B, Array<int> *C, std::string ID );
	template cublasStatus_t cublasOperations<float>::add_zerocopy( Array<float> *A, Array<float> *B, Array<float> *C, std::string ID );
	template cublasStatus_t cublasOperations<double>::add_zerocopy( Array<double> *A, Array<double> *B, Array<double> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexFloat>::add_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexDouble>::add_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID );

	// define sum, subtract, transpose or hermitian operation

	template<> void cublasOperations<ComplexFloat>::define_sum_subtract_transpose_hermitian_operation( ComplexFloat &alpha, ComplexFloat &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID )
	{
		if( ID.compare( "+" ) == 0)
		{
			alpha._Val[0] = 1;
			beta._Val[0] = 1;
		}
		else if( ID.compare( "-" ) == 0)
		{
			alpha._Val[0] = 1;
			beta._Val[0] = -1;
		}
		else if( ID.compare( "T" ) == 0)
		{
			alpha._Val[0] = 1;
			op1 = CUBLAS_OP_T;
			op2 = CUBLAS_OP_T;
		}
		else if( ID.compare( "H" ) == 0)
		{
			alpha._Val[0] = 1;
			op1 = CUBLAS_OP_C;
			op2 = CUBLAS_OP_C;
		}
	};

	template<> void cublasOperations<ComplexDouble>::define_sum_subtract_transpose_hermitian_operation( ComplexDouble &alpha, ComplexDouble &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID )
	{
		if( ID.compare( "+" ) == 0)
		{
			alpha._Val[0] = 1;
			beta._Val[0] = 1;
		}
		else if( ID.compare( "-" ) == 0)
		{
			alpha._Val[0] = 1;
			beta._Val[0] = -1;
		}
		else if( ID.compare( "T" ) == 0)
		{
			alpha._Val[0] = 1;
			op1 = CUBLAS_OP_T;
			op2 = CUBLAS_OP_T;
		}
		else if( ID.compare( "H" ) == 0)
		{
			alpha._Val[0] = 1;
			op1 = CUBLAS_OP_C;
			op2 = CUBLAS_OP_C;
		}
	};

	template <typename TElement>
	void cublasOperations<TElement>::define_sum_subtract_transpose_hermitian_operation( TElement &alpha, TElement &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID )
	{
		if( ID.compare( "+" ) == 0)
		{
			alpha = 1;
			beta = 1;
		}
		else if( ID.compare( "-" ) == 0)
		{
			alpha = 1;
			beta = -1;
		}
		else if( ID.compare( "T" ) == 0)
		{
			alpha = 1;
			op1 = CUBLAS_OP_T;
			op2 = CUBLAS_OP_T;
		}
		else if( ID.compare( "H" ) == 0 )
		{
			alpha = 1;
			op1 = CUBLAS_OP_C;
			op2 = CUBLAS_OP_C;
		}
	}

	template void cublasOperations<int>::define_sum_subtract_transpose_hermitian_operation( int &alpha, int &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID );
	template void cublasOperations<float>::define_sum_subtract_transpose_hermitian_operation( float &alpha, float &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID );
	template void cublasOperations<double>::define_sum_subtract_transpose_hermitian_operation( double &alpha, double &beta, cublasOperation_t &op1, cublasOperation_t &op2, std::string ID );

	// implementation of multiply

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C )
	{
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		if(matrix_size.Arows == 1 && matrix_size.Bcols == 1 )
		{
			// implement "vector(Transp) x vector = scalar" (dot function)
			CUBLAS_CALL( multiply_VectorTranspVector_Scalar( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows > 1 && matrix_size.Bcols == 1 && (matrix_size.Acols > 1 || matrix_size.Brows > 1))
		{
			// implement "matrix x vector = vector"
			CUBLAS_CALL( multiply_MatrixVector_Vector( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Acols == 1 && matrix_size.Bcols > 1)
		{
			// implement "vector(Transp) x vector = matrix"
			CUBLAS_CALL( multiply_VectorVectorTransp_Matrix( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows > 1 && matrix_size.Bcols > 1)
		{
			// implement "matrix x matrix = matrix"
			CUBLAS_CALL( multiply_MatrixMatrix_Matrix_Xt( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows == 1 && matrix_size.Bcols > 1)
		{
			// implement "vector(Transp) x matrix = vector(Transp)"
			CUBLAS_CALL( multiply_VectorTranspMatrix_VectorTransp( A, B, C, matrix_size ) );
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::multiply( Array<int> *A, Array<int> *B, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::multiply( Array<float> *A, Array<float> *B, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::multiply( Array<double> *A, Array<double> *B, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C )
	{
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		if(matrix_size.Arows == 1 && matrix_size.Bcols == 1 )
		{
			// implement "vector(Transp) x vector = scalar" (dot function)
			CUBLAS_CALL( multiply_VectorTranspVector_Scalar_zerocopy( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows > 1 && matrix_size.Bcols == 1 && (matrix_size.Acols > 1 || matrix_size.Brows > 1))
		{
			// implement "matrix x vector = vector"
			CUBLAS_CALL( multiply_MatrixVector_Vector_zerocopy( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Acols == 1 && matrix_size.Bcols > 1)
		{
			// implement "vector(Transp) x vector = matrix"
			CUBLAS_CALL( multiply_VectorVectorTransp_Matrix_zerocopy( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows > 1 && matrix_size.Bcols > 1)
		{
			// implement "matrix x matrix = matrix"
			CUBLAS_CALL( multiply_MatrixMatrix_Matrix_Xt( A, B, C, matrix_size ) );
		}
		else if(matrix_size.Arows == 1 && matrix_size.Bcols > 1)
		{
			// implement "vector(Transp) x matrix = vector(Transp)"
			CUBLAS_CALL( multiply_VectorTranspMatrix_VectorTransp_zerocopy( A, B, C, matrix_size ) );
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::multiply_zerocopy( Array<int> *A, Array<int> *B, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::multiply_zerocopy( Array<float> *A, Array<float> *B, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::multiply_zerocopy( Array<double> *A, Array<double> *B, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_zerocopy( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_zerocopy( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C );

	// implementation of Matrix x Matrix = Matrix

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;
		
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(TElement), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(TElement), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		
		CUBLAS_CALL( cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );	
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		CUBLAS_CALL( cublasShutdown() );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_stream( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );
		
		// create stream
		cudaStream_t stream;
		cudaStreamCreate( &stream );

		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;
		
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(TElement), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrixAsync( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows, stream ) );
		CUBLAS_CALL( cublasSetMatrixAsync( matrix_size.Brows, matrix_size.Bcols, sizeof(TElement), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows, stream ) );
		
		//tic();
		//for( int i = 0; i < 10; i++ )
		CUBLAS_CALL( cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );	
		//CUDA_CALL( cudaStreamSynchronize( stream ) );
		//toc();

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrixAsync( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), d_C, lda, C->m_data.GetElements(), ldb, stream ) );
		CUDA_CALL( cudaStreamSynchronize( stream ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		// Destroy the stream
		CUDA_CALL(cudaStreamDestroy(stream));

		CUBLAS_CALL( cublasShutdown() );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;
	
		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );	
		
		CUBLAS_CALL( cublasTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );	
		CUDA_CALL( cudaDeviceSynchronize() );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_Xt( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasXtHandle_t handle;
		cublasXtCreate(&handle);

		int devices[1] = { 0 };  // add this line
		CUBLAS_CALL( cublasXtDeviceSelect(handle, 1, devices) );

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		d_A = A->m_data.GetElements();
		d_B = B->m_data.GetElements();
		d_C = C->m_data.GetElements();

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasXtTgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );	
		CUDA_CALL( cudaDeviceSynchronize() );

		// Destroy the handle
		CUBLAS_CALL(cublasXtDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of Vector(transposed) x Vector = scalar (dot product)

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(TElement) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(TElement) * size_B;

		// define device memory
		TElement *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memory
		CUDA_CALL( cudaMalloc( &d_A, mem_size_A ) );
		CUDA_CALL( cudaMalloc( &d_B, mem_size_B ) );
	
		// copy to device
		CUDA_CALL( cudaMemcpy( d_A, A->data(), mem_size_A, cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->data(), mem_size_B, cudaMemcpyHostToDevice ) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTdot( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );	
		CUDA_CALL( cudaDeviceSynchronize()) ;
	
		*C->m_data.m_data = d_C;

		CUDA_CALL( cudaFree( d_A ) );
		CUDA_CALL( cudaFree( d_B ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(TElement) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(TElement) * size_B;

		// define device memory
		TElement *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// set device pointer to host memory
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTdot( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );	
		CUDA_CALL( cudaDeviceSynchronize()) ;
	
		*C->m_data.m_data = d_C;

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	// Implementation of vector x vector(transp) = matrix

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );
	
		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
	
		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		const TElement alpha = 1.0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(TElement), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(TElement), A->data(), incx, d_A, incy ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(TElement), B->data(), incx, d_B, incy ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), C->data(), matrix_size.Crows, d_C, matrix_size.Crows ) );
	
		CUDA_CALL( cudaDeviceSynchronize() );
		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTger( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );	
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), d_C, lda, C->data(), ldb ) );
		//CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(TElement), d_C, lda, C->data(), ldb ) );
		//CUDA_CALL( cudaMemcpy( C->data(), d_C, size_C, cudaMemcpyDeviceToHost ) );

		CUDA_CALL( cudaFree( d_A ) );
		CUDA_CALL( cudaFree( d_B ) );
		CUDA_CALL( cudaFree( d_C ) );
	
		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));
	
		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );
	
		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
	
		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// set device pointer to host memory
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );

		const TElement alpha = 1.0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;
	
		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTger( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );	
		CUDA_CALL( cudaDeviceSynchronize() );
	
		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));
	
		return CUBLAS_STATUS_SUCCESS;
	}

	// Implementation of matrix x vector = vector

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));		
		CUBLAS_CALL( cublasInit() );

		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			//*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(TElement), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(TElement), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(TElement), B->m_data.GetElements(), 1, d_B, 1 ) );

		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(TElement), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
		
	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));		
		CUBLAS_CALL( cublasInit() );

		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			//*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );
	
		const TElement alpha = 1.0;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}
		
	// implementation of vector(transp) x matrix = vector(transp)

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );

		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows * sizeof(TElement);
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows * sizeof(TElement);
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows * sizeof(TElement);

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memory
		CUDA_CALL( cudaMalloc( &d_A, size_A ) );
		CUDA_CALL( cudaMalloc( &d_B, size_B ) );
		CUDA_CALL( cudaMalloc( &d_C, size_C ) );

		// pass host pointer to device
		CUDA_CALL( cudaMemcpy( d_A, A->data(), size_A, cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->data(), size_B, cudaMemcpyHostToDevice ) );

		const TElement alpha = 1.0f;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
		CUDA_CALL( cudaDeviceSynchronize() );
	
		CUDA_CALL( cudaMemcpy( C->data(), d_C, size_C, cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_zerocopy( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasHandle_t handle;
		CUBLAS_CALL(cublasCreate(&handle));	
		CUBLAS_CALL( cublasInit() );

		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &d_A, A->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_B, B->m_data.GetElements(), 0 ) );
		CUDA_CALL( cudaHostGetDevicePointer( &d_C, C->m_data.GetElements(), 0 ) );

		const TElement alpha = 1.0f;
		const TElement beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasTgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
		CUDA_CALL( cudaDeviceSynchronize() );

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return CUBLAS_STATUS_SUCCESS;
	}

	// transform permutation vector into permutation matrix
	// TODO (or redo) - too slow!!!

	template <typename TElement>
	void cublasOperations<TElement>::from_permutation_vector_to_permutation_matrix( Array<TElement> *pivotMatrix, Array<int> *pivotVector )
	{
		//pivotVector->print();
		//*pivotMatrix = eye<TElement>(pivotVector->getDim(0));
		//index_t idx1, idx2;
		//for( int i = 0; i < pivotVector->GetDescriptor().GetDim(0); i++ ) {
		//	if( i + 1 == (*pivotVector)(i) )
		//		continue;
		//	else
		//	{
		//		idx1 = i;
		//		idx2 = (*pivotVector)(i)-1;
		//		(*pivotMatrix)( idx1, idx1 ) = 0;
		//		(*pivotMatrix)( idx2, idx2 ) = 0;
		//		(*pivotMatrix)( idx1, idx2 ) = 1;
		//		(*pivotMatrix)( idx2, idx1 ) = 1;
		//	}
		//	pivotMatrix->print();
		//}
		//pivotMatrix->print();
		//
		//*pivotMatrix = eye<TElement>(pivotVector->getDim(0));


		//pivotVector->print();
		//eye<double>(pivotVector->getDim(0)).print();
		Array<TElement> pivotAux = eye<TElement>(pivotVector->GetDescriptor().GetDim(0));
		index_t idx1, idx2;
		for( int i = 0; i < pivotVector->GetDescriptor().GetDim(0); i++ ) {
			idx1 = i;
			idx2 = (*pivotVector)(i)-1;
			pivotAux( idx1, idx1 ) = 0;
			pivotAux( idx2, idx2 ) = 0;
			pivotAux( idx1, idx2 ) = 1;
			pivotAux( idx2, idx1 ) = 1;

			(*pivotMatrix) = pivotAux*(*pivotMatrix);
			pivotAux = eye<TElement>(pivotVector->GetDescriptor().GetDim(0));
			//pivotMatrix->print();
		}
		//pivotMatrix->print();
	}

	// cublas wrappers

	cublasStatus_t cublasOperations<int>::cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const int *alpha, const int *A, int lda, const int *beta, const int *B, int ldb, int *C, int ldc)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	cublasStatus_t cublasOperations<float>::cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc)
	{
		return cublasSgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
	}

	cublasStatus_t cublasOperations<double>::cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc)
	{
		return cublasDgeam( handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
	}	
	
	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const ComplexFloat *alpha, const ComplexFloat *A, int lda, const ComplexFloat *beta, const ComplexFloat *B, int ldb, ComplexFloat *C, int ldc)
	{
		const cuFloatComplex alpha2 = make_cuFloatComplex( alpha->real(), alpha->imag() );
		const cuFloatComplex beta2 = make_cuFloatComplex( beta->real(), beta->imag() );
		return cublasCgeam( handle, transa, transb, m, n, &alpha2, (cuFloatComplex*)A, lda, &beta2, (cuFloatComplex*)B, ldb, (cuFloatComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgeam( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const ComplexDouble *alpha, const ComplexDouble *A, int lda, const ComplexDouble *beta, const ComplexDouble *B, int ldb, ComplexDouble *C, int ldc)
	{
		const cuDoubleComplex alpha2 = make_cuDoubleComplex( alpha->real(), alpha->imag() );
		const cuDoubleComplex beta2 = make_cuDoubleComplex( beta->real(), beta->imag() );
		return cublasZgeam( handle, transa, transb, m, n, &alpha2, (cuDoubleComplex*)A, lda, &beta2, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<int>::cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const int *alpha, const int *A, int lda, const int *B, int ldb, const int *beta, int *C, int ldc)
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<float>::cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
	{
		return cublasSgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
	}

	cublasStatus_t cublasOperations<double>::cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
	{
		return cublasDgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const ComplexFloat *alpha, const ComplexFloat *A, int lda, const ComplexFloat *B, int ldb, const ComplexFloat *beta, ComplexFloat *C, int ldc)
	{
		cuFloatComplex alpha2, beta2;
		alpha2 = make_cuFloatComplex( alpha->real(), alpha->imag() );
		beta2 = make_cuFloatComplex( beta->real(), beta->imag() );
		return cublasCgemm( handle, transa, transb, m, n, k, (const cuFloatComplex*)alpha, (const cuFloatComplex*)A, lda, (const cuFloatComplex*)B, ldb, (const cuFloatComplex*)beta, (cuFloatComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const ComplexDouble *alpha, const ComplexDouble *A, int lda, const ComplexDouble *B, int ldb, const ComplexDouble *beta, ComplexDouble *C, int ldc)
	{
		cuDoubleComplex alpha2, beta2;
		alpha2 = make_cuDoubleComplex( alpha->real(), alpha->imag() );
		beta2 = make_cuDoubleComplex( beta->real(), beta->imag() );
		return cublasZgemm( handle, transa, transb, m, n, k, &alpha2, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)B, ldb, &beta2, (cuDoubleComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<int>::cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const int *alpha, const int *A, int lda, const int *B, int ldb, const int *beta, int *C, int ldc)
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<float>::cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
	{
		return cublasXtSgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
	}

	cublasStatus_t cublasOperations<double>::cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
	{
		return cublasXtDgemm( handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const ComplexFloat *alpha, const ComplexFloat *A, int lda, const ComplexFloat *B, int ldb, const ComplexFloat *beta, ComplexFloat *C, int ldc)
	{
		return cublasXtCgemm( handle, transa, transb, m, n, k, (const cuFloatComplex*)alpha, (const cuFloatComplex*)A, lda, (const cuFloatComplex*)B, ldb, (const cuFloatComplex*)beta, (cuFloatComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasXtTgemm( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const ComplexDouble *alpha, const ComplexDouble *A, int lda, const ComplexDouble *B, int ldb, const ComplexDouble *beta, ComplexDouble *C, int ldc)
	{
		return cublasXtZgemm( handle, transa, transb, m, n, k, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)B, ldb, (const cuDoubleComplex*)beta, (cuDoubleComplex*)C, ldc );
	}

	cublasStatus_t cublasOperations<int>::cublasTdot( cublasHandle_t handle, int n, const int *x, int incx, const int *y, int incy, int *result )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	cublasStatus_t cublasOperations<float>::cublasTdot( cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result )
	{
		return cublasSdot( handle, n, x, incx, y, incy, result );
	}

	cublasStatus_t cublasOperations<double>::cublasTdot( cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result )
	{
		return cublasDdot( handle, n, x, incx, y, incy, result );
	}	
	
	cublasStatus_t cublasOperations<ComplexFloat>::cublasTdot( cublasHandle_t handle, int n, const ComplexFloat *x, int incx, const ComplexFloat *y, int incy, ComplexFloat *result )
	{
		cuFloatComplex aux;
		CUBLAS_CALL( cublasCdotu( handle, n, (const cuFloatComplex*)x, incx, (const cuFloatComplex*)y, incy, &aux ) );
		*result = ComplexFloat( aux.x, aux.y );
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTdot( cublasHandle_t handle, int n, const ComplexDouble *x, int incx, const ComplexDouble *y, int incy, ComplexDouble *result )
	{
		cuDoubleComplex aux;
		CUBLAS_CALL( cublasZdotu( handle, n, (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy, &aux ) );
		*result = ComplexDouble( aux.x, aux.y );
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<int>::cublasTger( cublasHandle_t handle, int m, int n, const int *alpha, const int *x, int incx, const int *y, int incy, int *A, int lda )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}
	
	cublasStatus_t cublasOperations<float>::cublasTger( cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda )
	{
		return cublasSger( handle, m, n, alpha, x, incx, y, incy, A, lda );
	}
	
	cublasStatus_t cublasOperations<double>::cublasTger( cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda )
	{
		return cublasDger( handle, m, n, alpha, x, incx, y, incy, A, lda );
	}
	
	cublasStatus_t cublasOperations<ComplexFloat>::cublasTger( cublasHandle_t handle, int m, int n, const ComplexFloat *alpha, const ComplexFloat *x, int incx, const ComplexFloat *y, int incy, ComplexFloat *A, int lda )
	{
		const cuFloatComplex alpha2 = make_cuFloatComplex( alpha->real(), alpha->imag() );
		return cublasCgeru( handle, m, n, &alpha2, (const cuFloatComplex*)x, incx, (const cuFloatComplex*)y, incy, (cuFloatComplex*)A, lda );
	}
	
	cublasStatus_t cublasOperations<ComplexDouble>::cublasTger( cublasHandle_t handle, int m, int n, const ComplexDouble *alpha, const ComplexDouble *x, int incx, const ComplexDouble *y, int incy, ComplexDouble *A, int lda )
	{
		const cuDoubleComplex alpha2 = make_cuDoubleComplex( alpha->real(), alpha->imag() );
		return cublasZgeru( handle, m, n, &alpha2, (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)y, incy, (cuDoubleComplex*)A, lda );
	}

	cublasStatus_t cublasOperations<int>::cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const int *alpha, const int *A, int lda, const int *x, int incx, const int *beta, int *y, int incy )
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t cublasOperations<float>::cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy )
	{
		return cublasSgemv( handle, op, m, n, alpha, A, lda, x, incx, beta, y, incy );
	}

	cublasStatus_t cublasOperations<double>::cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy )
	{
		return cublasDgemv( handle, op, m, n, alpha, A, lda, x, incx, beta, y, incy );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const ComplexFloat *alpha, const ComplexFloat *A, int lda, const ComplexFloat *x, int incx, const ComplexFloat *beta, ComplexFloat *y, int incy )
	{
		return cublasCgemv( handle, op, m, n, (const cuFloatComplex*)alpha, (const cuFloatComplex*)A, lda, (const cuFloatComplex*)x, incx, (const cuFloatComplex*)beta, (cuFloatComplex*)y, incy );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgemv( cublasHandle_t handle, cublasOperation_t op, int m, int n, const ComplexDouble *alpha, const ComplexDouble *A, int lda, const ComplexDouble *x, int incx, const ComplexDouble *beta, ComplexDouble *y, int incy )
	{
		return cublasZgemv( handle, op, m, n, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A, lda, (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy );
	}

	cublasStatus_t cublasOperations<int>::cublasTgetrfBatched( cublasHandle_t handle, int n, int *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<float>::cublasTgetrfBatched( cublasHandle_t handle, int n, float *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize )
	{
		return cublasSgetrfBatched( handle, n, Aarray, lda, PivotArray, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<double>::cublasTgetrfBatched( cublasHandle_t handle, int n, double *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize )
	{
		return cublasDgetrfBatched( handle, n, Aarray, lda, PivotArray, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgetrfBatched( cublasHandle_t handle, int n, ComplexFloat *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize )
	{
		return cublasCgetrfBatched( handle, n, (cuFloatComplex**)Aarray, lda, PivotArray, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgetrfBatched( cublasHandle_t handle, int n, ComplexDouble *Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize )
	{
		return cublasZgetrfBatched( handle, n, (cuDoubleComplex**)Aarray, lda, PivotArray, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<int>::cublasTgetriBatched( cublasHandle_t handle, int n, const int *Aarray[], int lda, int *PivotArray, int *Carray[], int ldc, int *infoArray, int batchSize )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<float>::cublasTgetriBatched( cublasHandle_t handle, int n, const float *Aarray[], int lda, int *PivotArray, float *Carray[], int ldc, int *infoArray, int batchSize )
	{
		return cublasSgetriBatched( handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<double>::cublasTgetriBatched( cublasHandle_t handle, int n, const double *Aarray[], int lda, int *PivotArray, double *Carray[], int ldc, int *infoArray, int batchSize )
	{
		return cublasDgetriBatched( handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgetriBatched( cublasHandle_t handle, int n, const ComplexFloat *Aarray[], int lda, int *PivotArray, ComplexFloat *Carray[], int ldc, int *infoArray, int batchSize )
	{
		return cublasCgetriBatched( handle, n, (const cuFloatComplex**)Aarray, lda, PivotArray, (cuFloatComplex**)Carray, ldc, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgetriBatched( cublasHandle_t handle, int n, const ComplexDouble *Aarray[], int lda, int *PivotArray, ComplexDouble *Carray[], int ldc, int *infoArray, int batchSize )
	{
		return cublasZgetriBatched( handle, n, (const cuDoubleComplex**)Aarray, lda, PivotArray, (cuDoubleComplex**)Carray, ldc, infoArray, BATCHSIZE );
	}

	cublasStatus_t cublasOperations<int>::cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, int *Aarray[], int lda, int *Carray[], int ldc, int *info, int *devInfoArray, int batchSize )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<float>::cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs,  float *Aarray[], int lda, float *Carray[], int ldc, int *info, int *devInfoArray, int batchSize )
	{
		return cublasSgelsBatched( handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize );
	}

	cublasStatus_t cublasOperations<double>::cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double *Aarray[], int lda, double *Carray[], int ldc, int *info, int *devInfoArray, int batchSize )
	{
		return cublasDgelsBatched( handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, ComplexFloat *Aarray[], int lda, ComplexFloat *Carray[], int ldc, int *info, int *devInfoArray, int batchSize )
	{
		return cublasCgelsBatched( handle, trans, m, n, nrhs, (cuFloatComplex**)Aarray, lda, (cuFloatComplex**)Carray, ldc, info, devInfoArray, batchSize );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgelsBatched( cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, ComplexDouble *Aarray[], int lda, ComplexDouble *Carray[], int ldc, int *info, int *devInfoArray, int batchSize )
	{
		return cublasZgelsBatched( handle, trans, m, n, nrhs, (cuDoubleComplex**)Aarray, lda, (cuDoubleComplex**)Carray, ldc, info, devInfoArray, batchSize );
	}

	cublasStatus_t cublasOperations<int>::cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, int *Aarray[], int lda, int *TauArray[], int *info, int batchSize )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<float>::cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, float *Aarray[], int lda, float *TauArray[], int *info, int batchSize )
	{
		return cublasSgeqrfBatched( handle, m, n, Aarray, lda, TauArray, info, batchSize );
	}

	cublasStatus_t cublasOperations<double>::cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, double *Aarray[], int lda, double *TauArray[], int *info, int batchSize )
	{
		return cublasDgeqrfBatched( handle, m, n, Aarray, lda, TauArray, info, batchSize );
	}

	cublasStatus_t cublasOperations<ComplexFloat>::cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, ComplexFloat *Aarray[], int lda, ComplexFloat *TauArray[], int *info, int batchSize )
	{
		return cublasCgeqrfBatched( handle, m, n, (cuFloatComplex**)Aarray, lda, (cuFloatComplex**)TauArray, info, batchSize );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::cublasTgeqrfBatched( cublasHandle_t handle, int m, int n, ComplexDouble *Aarray[], int lda, ComplexDouble *TauArray[], int *info, int batchSize )
	{
		return cublasZgeqrfBatched( handle, m, n, (cuDoubleComplex**)Aarray, lda, (cuDoubleComplex**)TauArray, info, batchSize );
	}
}