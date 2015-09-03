#include <boost/exception/all.hpp>
#include <boost/pointer_cast.hpp>

#include "cublasOperations.h"
#include "array.cuh"

namespace matCUDA
{

	// eig decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig( Array<TElement> *A, Array<TElement> *eigenvectors )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cudaError_t error;

		if( A->GetDescriptor().GetNDim() != 2 )
			return stat;
		if( A->GetDescriptor().GetDim(0) != A->GetDescriptor().GetDim(1) )
			return stat;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = eig_int( handle, A, eigenvectors, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = eig_float( handle, A, eigenvectors, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = eig_double( handle, A, eigenvectors, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = eig_ComplexFloat( handle, A, eigenvectors, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = eig_ComplexDouble( handle, A, eigenvectors, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::eig( Array<int> *A, Array<int> *eigenvectors );
	template cublasStatus_t cublasOperations<float>::eig( Array<float> *A, Array<float> *eigenvectors );
	template cublasStatus_t cublasOperations<double>::eig( Array<double> *A, Array<double> *eigenvectors );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig( Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig( Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::eig_int( cublasHandle_t handle, Array<int> *A, Array<int> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::eig_int( cublasHandle_t handle, Array<float> *A, Array<float> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::eig_int( cublasHandle_t handle, Array<double> *A, Array<double> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size )
	{	
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::eig_float( cublasHandle_t handle, Array<int> *A, Array<int> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::eig_float( cublasHandle_t handle, Array<double> *A, Array<double> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::eig_float( cublasHandle_t handle, Array<float> *A, Array<float> *eigenvectors, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		float *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(float*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(float)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const float *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(float*), cudaMemcpyHostToDevice) );

		// define device memory
		int *PivotArray;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const float *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(float*), cudaMemcpyHostToDevice) );
	
		float alpha = 1, beta = 0;
		int INFOh = 2;

		for( int i = 0; i < EIG_MAX_ITER; i++ )
		{
			// CALL CUBLAS FUNCTION - QR decomposition
			CUBLAS_CALL( cublasSgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

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
			zeros_under_diag<float> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );

			// CALL CUBLAS FUNCTION
			CUBLAS_CALL( cublasSgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
	
			cudaDeviceSynchronize();
			CUDA_CALL( cudaGetLastError() );

			// copy from GPU
			CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

			if( INFOh == lda )
			{
				printf("Factorization Failed: Matrix is singular\n");
				cublasShutdown();
				return CUBLAS_STATUS_EXECUTION_FAILED;
			}

			CUBLAS_CALL( cublasSgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const float*>(Aarray), lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

			CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );
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

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::eig_double( cublasHandle_t handle, Array<int> *A, Array<int> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::eig_double( cublasHandle_t handle, Array<float> *A, Array<float> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::eig_double( cublasHandle_t handle, Array<double> *A, Array<double> *eigenvectors, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::eig_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::eig_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::eig_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::eig_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::eig_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::eig_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *eigenvectors, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::eig_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::eig_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::eig_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *eigenvectors, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::eig_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *eigenvectors, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::eig_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *eigenvectors, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_SUCCESS;
	}

	// QR decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR( Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cudaError_t error;

		if( A->GetDescriptor().GetNDim() != 2 ||
			Q->GetDescriptor().GetNDim() != 2 ||
			R->GetDescriptor().GetNDim() != 2 )
			return stat;
		if( A->GetDescriptor().GetDim(0) < A->GetDescriptor().GetDim(1) ||
			Q->GetDescriptor().GetDim(0) < Q->GetDescriptor().GetDim(1) ||
			R->GetDescriptor().GetDim(0) < R->GetDescriptor().GetDim(1) )
			return stat;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = Q->GetDescriptor().GetDim(1);
		matrix_size.Brows = Q->GetDescriptor().GetDim(0);
		matrix_size.Ccols = R->GetDescriptor().GetDim(1);
		matrix_size.Crows = R->GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = QR_int( handle, A, Q, R, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = QR_float( handle, A, Q, R, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = QR_double( handle, A, Q, R, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = QR_ComplexFloat( handle, A, Q, R, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = QR_ComplexDouble( handle, A, Q, R, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::QR( Array<int> *A, Array<int> *Q, Array<int> *R );
	template cublasStatus_t cublasOperations<float>::QR( Array<float> *A, Array<float> *Q, Array<float> *R );
	template cublasStatus_t cublasOperations<double>::QR( Array<double> *A, Array<double> *Q, Array<double> *R );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR( Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR( Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::QR_int( cublasHandle_t handle, Array<int> *A, Array<int> *Q, Array<int> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::QR_int( cublasHandle_t handle, Array<float> *A, Array<float> *Q, Array<float> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::QR_int( cublasHandle_t handle, Array<double> *A, Array<double> *Q, Array<double> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size )
	{	
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::QR_float( cublasHandle_t handle, Array<int> *A, Array<int> *Q, Array<int> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::QR_float( cublasHandle_t handle, Array<double> *A, Array<double> *Q, Array<double> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::QR_float( cublasHandle_t handle, Array<float> *A, Array<float> *Q, Array<float> *R, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		float *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(float*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(float)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const float *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(float*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION - QR decomposition
		CUBLAS_CALL( cublasSgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// info from GPU
		if( infoArray < 0 )
		{
			printf("Parameter invalid for cublas<type>geqrfBatched: %d\n", -infoArray);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		Array<float> tau( size_tau );
		CUDA_CALL( cudaMemcpy( tau.m_data.GetElements(), d_tau, tau.m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );

		////// TODO
		// define device memory
		//float *d_H, *d_Q, *d_v, *d_eye;
		// kernel for eye;
		// fill in d_eye and d_Q
		//for( int i = 0; i < size_tau; i++ )
		//{
			// kernel for fill in d_v with R
			// kernel for d_H = tau[j]*v*v' -> cublas
			// kernel for d_H = I - d_H -> cublas
			// kernel for d_Q = d_Q*d_H
		//}
		////// END OF TODO

		// CALL CUDA FUNCTION
		zeros_under_diag<float> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );

		// define device memory
		int *PivotArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const float *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(float*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasSgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const float*>(Aarray), lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

		float alpha = 1, beta = 0;
		CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );

		CUDA_CALL( cudaMemcpy( Q->m_data.GetElements(), d_A, Q->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );

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

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::QR_double( cublasHandle_t handle, Array<int> *A, Array<int> *Q, Array<int> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::QR_double( cublasHandle_t handle, Array<float> *A, Array<float> *Q, Array<float> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::QR_double( cublasHandle_t handle, Array<double> *A, Array<double> *Q, Array<double> *R, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		double *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(double*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(double*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(double*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(double*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(double)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const double *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION - QR decomposition
		CUBLAS_CALL( cublasDgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

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
		zeros_under_diag<double> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );

		// define device memory
		int *PivotArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const double *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasDgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const double*>(Aarray), lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

		double alpha = 1, beta = 0;
		CUBLAS_CALL( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );

		CUDA_CALL( cudaMemcpy( Q->m_data.GetElements(), d_A, Q->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );

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

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::QR_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *Q, Array<int> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::QR_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *Q, Array<float> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::QR_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *Q, Array<double> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::QR_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::QR_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		cuComplex *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(cuComplex)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const cuComplex *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuComplex), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuComplex), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(cuComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION - QR decomposition
		CUBLAS_CALL( cublasCgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// info from GPU
		if( infoArray < 0 )
		{
			printf("Parameter invalid for cublas<type>geqrfBatched: %d\n", -infoArray);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );
		R->print();

		// CALL CUDA FUNCTION
		zeros_under_diag<cuComplex> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );

		// define device memory
		int *PivotArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const cuComplex *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(cuComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasCgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const cuComplex*>(Aarray), lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

		cuComplex alpha, beta;
		alpha = make_cuComplex( 1, 0 );
		beta = make_cuComplex( 0, 0 );
		CUBLAS_CALL( cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );

		CUDA_CALL( cudaMemcpy( Q->m_data.GetElements(), d_A, Q->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );

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

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::QR_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *Q, Array<TElement> *R, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::QR_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *Q, Array<int> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::QR_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *Q, Array<float> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::QR_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *Q, Array<double> *R, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::QR_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *Q, Array<ComplexFloat> *R, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::QR_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *Q, Array<ComplexDouble> *R, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Acols * matrix_size.Arows;
		int size_tau = std::min( matrix_size.Arows, matrix_size.Acols);
		size_tau = (unsigned)std::max( 1, size_tau );

		// define device memory
		cuDoubleComplex *d_A, *d_tau, *d_C, *d_aux, **Aarray = NULL, **tauArray = NULL, **Carray = NULL, **auxArray = NULL;
		int infoArray, *devInfoArray, *infoArray2;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&tauArray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&auxArray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_A ) ); 
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_aux ) ); 
		CUBLAS_CALL( cublasAlloc( size_tau, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_tau ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );
		
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		//save matrix address 
		const cuDoubleComplex *ptr_to_A[1], *ptr_to_tau[1], *ptr_to_C[1], *ptr_to_aux[1];
		ptr_to_A[0] = d_A;
		ptr_to_tau[0] = d_tau;
		ptr_to_C[0] = d_C;
		ptr_to_aux[0] = d_aux;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A->m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), A->m_data.GetElements(), ldc, d_aux, ldc ) );
	
		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(tauArray,ptr_to_tau, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(auxArray,ptr_to_aux, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION - QR decomposition
		CUBLAS_CALL( cublasZgeqrfBatched( handle, matrix_size.Arows, matrix_size.Acols, Aarray, lda, tauArray, &infoArray, BATCHSIZE ) );// CALL CUBLAS FUNCTION

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
		zeros_under_diag<cuDoubleComplex> ( d_A, std::max(1,(int)std::min(matrix_size.Arows,matrix_size.Acols) ) );
		CUDA_CALL( cudaMemcpy( R->m_data.GetElements(), d_A, R->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );

		// define device memory
		int *PivotArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;

		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray2 ) );

		//save matrix address 
		const cuDoubleComplex *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray2, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray2, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasZgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const cuDoubleComplex*>(Aarray), lda, PivotArray, Carray, lda, infoArray2, BATCHSIZE ) );

		cuDoubleComplex alpha, beta;
		alpha = make_cuDoubleComplex( 1, 0 );
		beta = make_cuDoubleComplex( 0, 0 );
		CUBLAS_CALL( cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Ccols, matrix_size.Acols, &alpha, d_aux, matrix_size.Arows, d_C, matrix_size.Brows, &beta, d_A, matrix_size.Crows) );

		CUDA_CALL( cudaMemcpy( Q->m_data.GetElements(), d_A, Q->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );

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

		return CUBLAS_STATUS_SUCCESS;
	}

	// least square solution

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS( Array<TElement> A, Array<TElement> *x, Array<TElement> C )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
		cudaError_t error;

		if( A.GetDescriptor().GetNDim() != 2 ||
			C.GetDescriptor().GetNDim() != 2 )
			return stat;
		if( A.GetDescriptor().GetDim(0) < A.GetDescriptor().GetDim(1) )
			return stat;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A.GetDescriptor().GetDim(1);
		matrix_size.Arows = A.GetDescriptor().GetDim(0);
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = LS_int( handle, A, x, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = LS_float( handle, A, x, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = LS_double( handle, A, x, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = LS_ComplexFloat( handle, A, x, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = LS_ComplexDouble( handle, A, x, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::LS( Array<int> A, Array<int> *x, Array<int> C );
	template cublasStatus_t cublasOperations<float>::LS( Array<float> A, Array<float> *x, Array<float> C );
	template cublasStatus_t cublasOperations<double>::LS( Array<double> A, Array<double> *x, Array<double> C );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS( Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS( Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_int( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LS_int( cublasHandle_t handle, Array<int> A, Array<int> *x, Array<int> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LS_int( cublasHandle_t handle, Array<float> A, Array<float> *x, Array<float> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LS_int( cublasHandle_t handle, Array<double> A, Array<double> *x, Array<double> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS_int( cublasHandle_t handle, Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS_int( cublasHandle_t handle, Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_float( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size )
	{	
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LS_float( cublasHandle_t handle, Array<int> A, Array<int> *x, Array<int> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LS_float( cublasHandle_t handle, Array<double> A, Array<double> *x, Array<double> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS_float( cublasHandle_t handle, Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS_float( cublasHandle_t handle, Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::LS_float( cublasHandle_t handle, Array<float> A, Array<float> *x, Array<float> C, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(float*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		//save matrix address 
		const float *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A.m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), C.m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(float*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		C.m_padded = false;
		CUDA_CALL( cudaMemcpy( C.m_data.GetElements(), d_C, C.m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = C( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_double( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LS_double( cublasHandle_t handle, Array<int> A, Array<int> *x, Array<int> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LS_double( cublasHandle_t handle, Array<float> A, Array<float> *x, Array<float> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS_double( cublasHandle_t handle, Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS_double( cublasHandle_t handle, Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::LS_double( cublasHandle_t handle, Array<double> A, Array<double> *x, Array<double> C, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(double*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(double*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		//save matrix address 
		const double *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A.m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), C.m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		C.m_padded = false;
		CUDA_CALL( cudaMemcpy( C.m_data.GetElements(), d_C, C.m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = C( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_ComplexFloat( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LS_ComplexFloat( cublasHandle_t handle, Array<int> A, Array<int> *x, Array<int> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LS_ComplexFloat( cublasHandle_t handle, Array<float> A, Array<float> *x, Array<float> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LS_ComplexFloat( cublasHandle_t handle, Array<double> A, Array<double> *x, Array<double> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LS_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::LS_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuComplex *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		//save matrix address 
		const cuComplex *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuComplex), A.m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuComplex), C.m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(cuComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		C.m_padded = false;
		CUDA_CALL( cudaMemcpy( C.m_data.GetElements(), d_C, C.m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = C( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LS_ComplexDouble( cublasHandle_t handle, Array<TElement> A, Array<TElement> *x, Array<TElement> C, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LS_ComplexDouble( cublasHandle_t handle, Array<int> A, Array<int> *x, Array<int> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LS_ComplexDouble( cublasHandle_t handle, Array<float> A, Array<float> *x, Array<float> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LS_ComplexDouble( cublasHandle_t handle, Array<double> A, Array<double> *x, Array<double> C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LS_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> A, Array<ComplexFloat> *x, Array<ComplexFloat> C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::LS_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> A, Array<ComplexDouble> *x, Array<ComplexDouble> C, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int infoArray, *devInfoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( BATCHSIZE, sizeof(int)*BATCHSIZE, (void **) &devInfoArray ) );

		//save matrix address 
		const cuDoubleComplex *ptr_to_A[1], *ptr_to_C[1];
		ptr_to_A[0] = d_A;
		ptr_to_C[0] = d_C;
	
		const int lda = matrix_size.Arows;
		const int ldc = matrix_size.Crows;
		unsigned int nhrs = matrix_size.Ccols;

		// copy to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A.m_data.GetElements(), lda, d_A, lda ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), C.m_data.GetElements(), ldc, d_C, ldc ) );

		//copy ptr_to_A references to device
		CUDA_CALL( cudaMemcpy(Aarray,ptr_to_A, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,ptr_to_C, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgelsBatched( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, matrix_size.Ccols, Aarray, lda, Carray, ldc, &infoArray, devInfoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh;
		CUDA_CALL( cudaMemcpy( &INFOh, devInfoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( infoArray != 0 )
		{
			printf("Parameter invalid for cublas<type>gelsBatched: %d\n", -INFOh);
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		C.m_padded = false;
		CUDA_CALL( cudaMemcpy( C.m_data.GetElements(), d_C, C.m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );
		for( int i = 0; i< x->GetDescriptor().GetDim( 0 ); i++ ) {
			for( int j = 0; j < x->GetDescriptor().GetDim( 1 ); j++ )
				(*x)( i, j ) = C( i, j );
		}

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );
		CUBLAS_CALL( cublasFree( devInfoArray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation on inversion

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert( Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
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

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = invert_int( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = invert_float( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = invert_double( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = invert_ComplexFloat( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = invert_ComplexDouble( handle, A, LU, Pivot, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::invert( Array<int> *A, Array<int> *LU, Array<int> *Pivot );
	template cublasStatus_t cublasOperations<float>::invert( Array<float> *A, Array<float> *LU, Array<int> *Pivot );
	template cublasStatus_t cublasOperations<double>::invert( Array<double> *A, Array<double> *LU, Array<int> *Pivot );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert( Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert( Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::invert_int( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::invert_int( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::invert_int( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{	
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::invert_float( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::invert_float( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::invert_float( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		float *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(float*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(float*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		const float *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(float*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(float*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		// copy from GPU
		int INFOh = 2;
		CUDA_CALL( cudaMemcpy( &INFOh, infoArray, sizeof( int ), cudaMemcpyDeviceToHost ) );

		if( INFOh == lda )
		{
			printf("Factorization Failed: Matrix is singular\n");
			cublasShutdown();
			return CUBLAS_STATUS_EXECUTION_FAILED;
		}

		CUBLAS_CALL( cublasSgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const float*>(Aarray), lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( Pivot->m_data.GetElements(), PivotArray, Pivot->m_data.m_numElements*sizeof( int ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::invert_double( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::invert_double( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::invert_double( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		double *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(double*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(double*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		const double *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUBLAS_CALL( cublasDgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const double*>(Aarray), lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( Pivot->m_data.GetElements(), PivotArray, Pivot->m_data.m_numElements*sizeof( int ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::invert_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::invert_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::invert_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::invert_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::invert_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		cuComplex *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		const cuComplex *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuComplex), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(cuComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(cuComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUBLAS_CALL( cublasCgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const cuComplex*>(Aarray), lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( Pivot->m_data.GetElements(), PivotArray, Pivot->m_data.m_numElements*sizeof( int ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::invert_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::invert_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::invert_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::invert_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::invert_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<int> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::invert_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<int> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;

		// define device memory
		cuDoubleComplex *d_A, *d_C, **Aarray = NULL, **Carray = NULL;
		int *PivotArray, *infoArray;

		//PivotArray = new int[matrix_size.Arows];
		//for (int i = 0; i < matrix_size.Arows; i++)
		//	PivotArray[i] = i;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUDA_CALL( cudaMalloc(&Carray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_C ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		const cuDoubleComplex *matrices[1], *inverses[1];
		matrices[0] = d_A;
		inverses[0] = d_C;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );
		CUDA_CALL( cudaMemcpy(Carray,inverses, sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUBLAS_CALL( cublasZgetriBatched( handle, matrix_size.Arows, boost::const_pointer_cast<const cuDoubleComplex*>(Aarray), lda, PivotArray, Carray, lda, infoArray, BATCHSIZE ) );
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( Pivot->m_data.GetElements(), PivotArray, Pivot->m_data.m_numElements*sizeof( int ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );
		CUBLAS_CALL( cublasFree( Carray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of LU decomposition

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU( Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cublasCreate_v2(&handle);
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

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = LU_int( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = LU_float( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = LU_double( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = LU_ComplexFloat( handle, A, LU, Pivot, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = LU_ComplexDouble( handle, A, LU, Pivot, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::LU( Array<int> *A, Array<int> *LU, Array<int> *Pivot);
	template cublasStatus_t cublasOperations<float>::LU( Array<float> *A, Array<float> *LU, Array<float> *Pivot );
	template cublasStatus_t cublasOperations<double>::LU( Array<double> *A, Array<double> *LU, Array<double> *Pivot );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU( Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU( Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LU_int( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LU_int( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<float> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LU_int( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<double> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LU_float( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LU_float( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<double> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::LU_float( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<float> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		float *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(float*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		float *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( pivotVector.m_data.GetElements(), PivotArray, size_pivot*sizeof( int ), cudaMemcpyDeviceToHost ) );

		from_permutation_vector_to_permutation_matrix( Pivot, &pivotVector );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LU_double( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LU_double( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<float> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::LU_double( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<double> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		double *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(double*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		double *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( pivotVector.m_data.GetElements(), PivotArray, size_pivot*sizeof( int ), cudaMemcpyDeviceToHost ) );

		from_permutation_vector_to_permutation_matrix( Pivot, &pivotVector );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LU_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LU_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<float> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LU_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<double> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::LU_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::LU_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		cuComplex *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		cuComplex *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuComplex), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( pivotVector.m_data.GetElements(), PivotArray, size_pivot*sizeof( int ), cudaMemcpyDeviceToHost ) );

		from_permutation_vector_to_permutation_matrix( Pivot, &pivotVector );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::LU_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *LU, Array<TElement> *Pivot, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::LU_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *LU, Array<int> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::LU_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *LU, Array<float> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::LU_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *LU, Array<double> *Pivot, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::LU_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *LU, Array<ComplexFloat> *Pivot, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::LU_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *LU, Array<ComplexDouble> *Pivot, _matrixSize matrix_size )
	{
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_pivot = std::min( matrix_size.Acols, matrix_size.Arows );

		// define device memory
		cuDoubleComplex *d_A, **Aarray = NULL;
		int *PivotArray, *infoArray;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));

		// allocate memmory
		CUDA_CALL( cudaMalloc(&Aarray,  sizeof(cuDoubleComplex*) * BATCHSIZE) );
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex)*BATCHSIZE, (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_pivot, sizeof(int)*BATCHSIZE, (void **) &PivotArray ) );
		CUBLAS_CALL( cublasAlloc( 1, sizeof(int)*BATCHSIZE, (void **) &infoArray ) );

		//save matrix address 
		cuDoubleComplex *matrices[1];
		matrices[0] = d_A;

		// copy to GPU
		const int lda = matrix_size.Arows;
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A->m_data.GetElements(), lda, d_A, lda ) );

		//copy matrices references to device
		CUDA_CALL( cudaMemcpy(Aarray,matrices, sizeof(double*), cudaMemcpyHostToDevice) );

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgetrfBatched( handle, matrix_size.Arows, Aarray, lda, PivotArray, infoArray, BATCHSIZE ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

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
		CUDA_CALL( cudaMemcpy( LU->m_data.GetElements(), d_A, LU->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );
		CUDA_CALL( cudaMemcpy( pivotVector.m_data.GetElements(), PivotArray, size_pivot*sizeof( int ), cudaMemcpyDeviceToHost ) );

		from_permutation_vector_to_permutation_matrix( Pivot, &pivotVector );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( PivotArray ) );
		CUBLAS_CALL( cublasFree( infoArray ) );
		CUBLAS_CALL( cublasFree( Aarray ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of conjugate

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::conjugate( Array<TElement> *A )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		Array<float> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = CUBLAS_STATUS_SUCCESS;
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = CUBLAS_STATUS_SUCCESS;
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = CUBLAS_STATUS_SUCCESS;
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = conjugate_ComplexFloat( handle, A, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = conjugate_ComplexDouble( handle, A, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		//A->print();
		//C->print();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::conjugate( Array<int> *A );
	template cublasStatus_t cublasOperations<float>::conjugate( Array<float> *A );
	template cublasStatus_t cublasOperations<double>::conjugate( Array<double> *A );
	template cublasStatus_t cublasOperations<ComplexFloat>::conjugate( Array<ComplexFloat> *A );
	template cublasStatus_t cublasOperations<ComplexDouble>::conjugate( Array<ComplexDouble> *A );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::conjugate_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexFloat alpha2 = ComplexFloat( 1.0, 0.0 );
		ComplexFloat beta2 = ComplexFloat( 0.0, 0.0 );
		cublasOperation_t op1 = CUBLAS_OP_T, op2 = CUBLAS_OP_T;

		//define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuComplex alpha = make_cuComplex( (int)alpha2.real(), (int)alpha2.imag() );
		cuComplex beta = make_cuComplex( (int)beta2.real(), (int)beta2.imag() );

		Array<ComplexFloat> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		cuComplex *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = lda;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasCgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		op1 = op2 = CUBLAS_OP_C;
		CUBLAS_CALL( cublasCgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Acols, &alpha, d_C, lda, &beta, d_C, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_A, A->m_data.m_numElements*sizeof( cuComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::conjugate_ComplexFloat( cublasHandle_t handle, Array<int> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::conjugate_ComplexFloat( cublasHandle_t handle, Array<float> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::conjugate_ComplexFloat( cublasHandle_t handle, Array<double> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::conjugate_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::conjugate_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::conjugate_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexDouble alpha2 = ComplexDouble( 1.0, 0.0 );
		ComplexDouble beta2 = ComplexDouble( 0.0, 0.0 );
		cublasOperation_t op1 = CUBLAS_OP_T, op2 = CUBLAS_OP_T;

		//define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuDoubleComplex alpha = make_cuDoubleComplex( (int)alpha2.real(), (int)alpha2.imag() );
		cuDoubleComplex beta = make_cuDoubleComplex( (int)beta2.real(), (int)beta2.imag() );

		Array<ComplexDouble> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = lda;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasZgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		CUDA_CALL( cudaGetLastError() );

		op1 = op2 = CUBLAS_OP_C;
		CUBLAS_CALL( cublasZgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Acols, &alpha, d_C, lda, &beta, d_C, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_A, A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::conjugate_ComplexDouble( cublasHandle_t handle, Array<int> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::conjugate_ComplexDouble( cublasHandle_t handle, Array<float> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::conjugate_ComplexDouble( cublasHandle_t handle, Array<double> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::conjugate_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::conjugate_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, _matrixSize matrix_size );

	// implementation of hermitian

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::hermitian( Array<TElement> *A )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		Array<float> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = transpose_hermitian_int( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = transpose_hermitian_float( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = transpose_hermitian_double( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = transpose_hermitian_ComplexFloat( handle, A, "H", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = transpose_hermitian_ComplexDouble( handle, A, "H", matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		//A->print();
		//C->print();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::hermitian( Array<int> *A );
	template cublasStatus_t cublasOperations<float>::hermitian( Array<float> *A );
	template cublasStatus_t cublasOperations<double>::hermitian( Array<double> *A );
	template cublasStatus_t cublasOperations<ComplexFloat>::hermitian( Array<ComplexFloat> *A );
	template cublasStatus_t cublasOperations<ComplexDouble>::hermitian( Array<ComplexDouble> *A );

	// implementation of transpose

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose( Array<TElement> *A )
	{	
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;
	
		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);

		//Array<float> C( matrix_size.Acols, matrix_size.Arows );
		//matrix_size.Ccols = C.GetDescriptor().GetDim(1);
	 //   matrix_size.Crows = C.GetDescriptor().GetDim(0);
		//matrix_size.Ccols = C->GetDescriptor().GetDim(1);
	 //   matrix_size.Crows = C->GetDescriptor().GetDim(0);
		//matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		//matrix_size.Brows = B->GetDescriptor().GetDim(0);
		//matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		//matrix_size.Crows = C->GetDescriptor().GetDim(0);

		//A->print();
		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = transpose_hermitian_int( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = transpose_hermitian_float( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = transpose_hermitian_double( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = transpose_hermitian_ComplexFloat( handle, A, "T", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = transpose_hermitian_ComplexDouble( handle, A, "T", matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::transpose( Array<int> *A );
	template cublasStatus_t cublasOperations<float>::transpose( Array<float> *A );
	template cublasStatus_t cublasOperations<double>::transpose( Array<double> *A );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose( Array<ComplexFloat> *A );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose( Array<ComplexDouble> *A );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_hermitian_int( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::transpose_hermitian_int( cublasHandle_t handle, Array<int> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::transpose_hermitian_int( cublasHandle_t handle, Array<float> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::transpose_hermitian_int( cublasHandle_t handle, Array<double> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_int( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose_hermitian_int( cublasHandle_t handle, Array<ComplexDouble> *A, std::string ID, _matrixSize matrix_size );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_hermitian_float( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::transpose_hermitian_float( cublasHandle_t handle, Array<int> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::transpose_hermitian_float( cublasHandle_t handle, Array<double> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_float( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose_hermitian_float( cublasHandle_t handle, Array<ComplexDouble> *A, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::transpose_hermitian_float( cublasHandle_t handle, Array<float> *A, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		float alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		Array<float> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		float *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( float ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasSgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->GetDescriptor().Swap();
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_hermitian_double( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::transpose_hermitian_double( cublasHandle_t handle, Array<int> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::transpose_hermitian_double( cublasHandle_t handle, Array<float> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_double( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::transpose_hermitian_double( cublasHandle_t handle, Array<ComplexDouble> *A, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::transpose_hermitian_double( cublasHandle_t handle, Array<double> *A, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		double alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		Array<double> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		double *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( double ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( double ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasDgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->GetDescriptor().Swap();
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<int> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<float> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<double> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexFloat alpha2 = ComplexFloat( 0, 0 );
		ComplexFloat beta2 = ComplexFloat( 0, 0 );
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuFloatComplex alpha = make_cuFloatComplex( alpha2.real(), alpha2.imag() );
		cuFloatComplex beta = make_cuFloatComplex( beta2.real(), beta2.imag() );

		Array<ComplexDouble> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasCgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->GetDescriptor().Swap();
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, std::string ID, _matrixSize matrix_size )
	{
		// TODO
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	template cublasStatus_t cublasOperations<int>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<int> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<float> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<double> *A, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::transpose_hermitian_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexDouble alpha2 = ComplexDouble( 0, 0 );
		ComplexDouble beta2 = ComplexDouble( 0, 0 );
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuDoubleComplex alpha = make_cuDoubleComplex( alpha2.real(), alpha2.imag() );
		cuDoubleComplex beta = make_cuDoubleComplex( beta2.real(), beta2.imag() );

		Array<ComplexDouble> C( matrix_size.Acols, matrix_size.Arows );
		matrix_size.Ccols = C.GetDescriptor().GetDim(1);
		matrix_size.Crows = C.GetDescriptor().GetDim(0);

		// define host memory size for matrices A and C
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
		unsigned int size_B = size_C;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
		d_B = &alpha;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C.m_data.GetElements(), C.m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;
		int ldc = matrix_size.Crows;
		CUBLAS_CALL( cublasZgeam( handle, op1, op2, matrix_size.Acols, matrix_size.Arows, &alpha, d_A, lda, &beta, d_A, ldb, d_C, ldc ) );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();
	
		A->GetDescriptor().Swap();
		A->m_padded = false;
		CUDA_CALL( cudaMemcpy( A->m_data.GetElements(), d_C, A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of minus

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::minus( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = sum_subtract_int( handle, A, B, C, "-", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = sum_subtract_float( handle, A, B, C, "-", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = sum_subtract_double( handle, A, B, C, "-", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = sum_subtract_ComplexFloat( handle, A, B, C, "-", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = sum_subtract_ComplexDouble( handle, A, B, C, "-", matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::minus( Array<int> *A, Array<int> *B, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::minus( Array<float> *A, Array<float> *B, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::minus( Array<double> *A, Array<double> *B, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::minus( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::minus( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C );

	// implementation of add2

	//template<> cublasStatus_t cublasOperations<ComplexFloat>::add2( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, const char ID )
	//{
	//	cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
	//	cublasHandle_t handle;
	//	cublasCreate( &handle ); 
	//
	//	_matrixSize matrix_size;
	//	matrix_size.Acols = A->GetDescriptor().GetDim(1);
	//    matrix_size.Arows = A->GetDescriptor().GetDim(0);
	//    matrix_size.Bcols = B->GetDescriptor().GetDim(1);
	//    matrix_size.Brows = B->GetDescriptor().GetDim(0);
	//    matrix_size.Ccols = C->GetDescriptor().GetDim(1);
	//    matrix_size.Crows = C->GetDescriptor().GetDim(0);
	//
	//	ComplexFloat alpha2 = ComplexFloat( 0, 0 );
	//	ComplexFloat beta2 = ComplexFloat( 0, 0 );
	//	cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;
	//
	//	define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
	//	cuCmplx alpha = cuCmplx( alpha2.real(), alpha2.imag() );
	//	cuCmplx beta = cuCmplx( beta2.real(), beta2.imag() );
	//
	//	// define host memory size for matrices A and B
	//    unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
	//    unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
	//    unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
	//
	//    // define device memory
	//    cuCmplx *d_A, *d_B, *d_C;
	//	
	//	cudaDeviceProp deviceProp;
	//	int devID = 0;
	//    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	//	
	//	// allocate memmory
	//	CUBLAS_CALL(cublasAlloc( size_A, sizeof(cuCmplx), (void **) &d_A ));
	//	CUBLAS_CALL(cublasAlloc( size_B, sizeof(cuCmplx), (void **) &d_B ));
	//	CUBLAS_CALL(cublasAlloc( size_C, sizeof(cuCmplx), (void **) &d_C ));
	//
	//	// copy to GPU
	//	CUDA_CALL(cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//	CUDA_CALL(cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//	CUDA_CALL(cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//
	//	// CALL CUBLAS FUNCTION
	//	//CUBLAS_CALL(sum_subtract_complex( handle, op1, op2, &alpha, d_A, &beta, d_B, d_C, matrix_size ));
	//	
	//	CUDA_CALL(cudaDeviceSynchronize());
	//	CUDA_CALL(cudaGetLastError());
	//
	//	// copy result from device to host
	//	CUDA_CALL(cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyDeviceToHost ));
	//
	//	// free memory
	//	CUBLAS_CALL(cublasFree( d_A ));
	//	CUBLAS_CALL(cublasFree( d_B ));
	//	CUBLAS_CALL(cublasFree( d_C ));
	//
	//	return CUBLAS_STATUS_SUCCESS;
	//}
	//
	//template<> cublasStatus_t cublasOperations<ComplexDouble>::add2( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, const char ID )
	//{
	//	cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
	//	cublasHandle_t handle;
	//	cublasCreate( &handle ); 
	//
	//	_matrixSize matrix_size;
	//	matrix_size.Acols = A->GetDescriptor().GetDim(1);
	//    matrix_size.Arows = A->GetDescriptor().GetDim(0);
	//    matrix_size.Bcols = B->GetDescriptor().GetDim(1);
	//    matrix_size.Brows = B->GetDescriptor().GetDim(0);
	//    matrix_size.Ccols = C->GetDescriptor().GetDim(1);
	//    matrix_size.Crows = C->GetDescriptor().GetDim(0);
	//
	//	ComplexDouble alpha2 = ComplexDouble( 0, 0 );
	//	ComplexDouble beta2 = ComplexDouble( 0, 0 );
	//	cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;
	//
	//	define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
	//	cuCmplx alpha = cuCmplx( alpha2.real(), alpha2.imag() );
	//	cuCmplx beta = cuCmplx( beta2.real(), beta2.imag() );
	//
	//	// define host memory size for matrices A and B
	//    unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
	//    unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
	//    unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;
	//
	//    // define device memory
	//    cuCmplx *d_A, *d_B, *d_C;
	//	
	//	cudaDeviceProp deviceProp;
	//	int devID = 0;
	//    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	//	
	//	// allocate memmory
	//	CUBLAS_CALL(cublasAlloc( size_A, sizeof(cuCmplx), (void **) &d_A ));
	//	CUBLAS_CALL(cublasAlloc( size_B, sizeof(cuCmplx), (void **) &d_B ));
	//	CUBLAS_CALL(cublasAlloc( size_C, sizeof(cuCmplx), (void **) &d_C ));
	//
	//	// copy to GPU
	//	CUDA_CALL(cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//	CUDA_CALL(cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//	CUDA_CALL(cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyHostToDevice ));
	//
	//	// CALL CUBLAS FUNCTION
	//	//CUBLAS_CALL(sum_subtract_complex( handle, op1, op2, &alpha, d_A, &beta, d_B, d_C, matrix_size ));
	//	//CUBLAS_CALL(sum_subtract( handle, op1, op2, &alpha, d_A, &beta, d_B, d_C, matrix_size ));
	//	
	//	CUDA_CALL(cudaDeviceSynchronize());
	//	CUDA_CALL(cudaGetLastError());
	//
	//	// copy result from device to host
	//	CUDA_CALL(cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( cuCmplx ), cudaMemcpyDeviceToHost ));
	//
	//	// free memory
	//	CUBLAS_CALL(cublasFree( d_A ));
	//	CUBLAS_CALL(cublasFree( d_B ));
	//	CUBLAS_CALL(cublasFree( d_C ));
	//
	//	return CUBLAS_STATUS_SUCCESS;
	//}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::add2( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
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

		// TO_REAL/COMPLEX_FUNCTION
		// define device memory
		TElement *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL(cublasAlloc( size_A, sizeof(TElement), (void **) &d_A ));
		CUBLAS_CALL(cublasAlloc( size_B, sizeof(TElement), (void **) &d_B ));
		CUBLAS_CALL(cublasAlloc( size_C, sizeof(TElement), (void **) &d_C ));

		// copy to GPU
		CUDA_CALL(cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ));
		CUDA_CALL(cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyHostToDevice ));

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL(sum_subtract( handle, op1, op2, &alpha, d_A, &beta, d_B, d_C, matrix_size ));
	
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaGetLastError());

		// copy result from device to host
		CUDA_CALL(cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( TElement ), cudaMemcpyDeviceToHost ));

		// free memory
		CUBLAS_CALL(cublasFree( d_A ));
		CUBLAS_CALL(cublasFree( d_B ));
		CUBLAS_CALL(cublasFree( d_C ));

		return CUBLAS_STATUS_SUCCESS;
	}

	template cublasStatus_t cublasOperations<int>::add2( Array<int> *A, Array<int> *B, Array<int> *C, std::string ID );
	template cublasStatus_t cublasOperations<float>::add2( Array<float> *A, Array<float> *B, Array<float> *C, std::string ID );
	template cublasStatus_t cublasOperations<double>::add2( Array<double> *A, Array<double> *B, Array<double> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexFloat>::add2( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID );
	template cublasStatus_t cublasOperations<ComplexDouble>::add2( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID );

	//template cublasStatus_t cublasOperations<ComplexFloat>::add2( Array<cuComplex> *A, Array<cuComplex> *B, Array<cuComplex> *C, std::string ID );
	//template cublasStatus_t cublasOperations<ComplexDouble>::add2( Array<cuDoubleComplex> *A, Array<cuDoubleComplex> *B, Array<cuDoubleComplex> *C, std::string ID );

	cublasStatus_t cublasOperations<int>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const int *alpha, const int *A, const int *beta, const int *B, int *C, _matrixSize matrix_size)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
	}

	cublasStatus_t cublasOperations<float>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const float *alpha, const float *A, const float *beta, const float *B, float *C, _matrixSize matrix_size)
	{
		return cublasSgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, alpha, A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	}

	cublasStatus_t cublasOperations<double>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const double *alpha, const double *A, const double *beta, const double *B, double *C, _matrixSize matrix_size)
	{
		return cublasDgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, alpha, A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	}

	// TODO
	//cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_complex( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const cuComplex *alpha, const cuComplex *A, const cuComplex *beta, const cuComplex *B, cuComplex *C, _matrixSize matrix_size)
	//{
	//	return CUBLAS_STATUS_NOT_INITIALIZED;
	//}
	//
	//cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_complex( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *beta, const cuDoubleComplex *B, cuDoubleComplex *C, _matrixSize matrix_size)
	//{
	//	return CUBLAS_STATUS_NOT_INITIALIZED;
	//}

	cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const ComplexFloat *alpha, const ComplexFloat *A, const ComplexFloat *beta, const ComplexFloat *B, ComplexFloat *C, _matrixSize matrix_size)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
		cuComplex *cuA, *cuB, *cuC;
		//cuA = A;
		//return cublasCgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, static_cast<cuComplex>(alpha), A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	}

	cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const ComplexDouble *alpha, const ComplexDouble *A, const ComplexDouble *beta, const ComplexDouble *B, ComplexDouble *C, _matrixSize matrix_size)
	{
		return CUBLAS_STATUS_NOT_INITIALIZED;
		//return cublasZgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, static_cast<cuComplex>(&alpha), A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	}

	//cublasStatus_t cublasOperations<cuCmplx>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const cuCmplx *alpha, const cuCmplx *A, const cuCmplx *beta, const cuCmplx *B, cuCmplx *C, _matrixSize matrix_size)
	//{
	//	return CUBLAS_STATUS_NOT_INITIALIZED;
	//}

	//cublasStatus_t cublasOperations<cuComplex>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const cuComplex *alpha, const cuComplex *A, const cuComplex *beta, const cuComplex *B, cuComplex *C, _matrixSize matrix_size)
	//{
	//	return cublasCgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, alpha, A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	//}
	//
	//cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, const cuDoubleComplex *alpha, const cuDoubleComplex *A, const cuDoubleComplex *beta, const cuDoubleComplex *B, cuDoubleComplex *C, _matrixSize matrix_size)
	//{
	//	return cublasZgeam( handle, transa, transb, matrix_size.Arows, matrix_size.Bcols, alpha, A, matrix_size.Arows, beta, B, matrix_size.Brows, C, matrix_size.Crows );
	//}
	// implementation of add

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::add( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		_matrixSize matrix_size;
		matrix_size.Acols = A->GetDescriptor().GetDim(1);
		matrix_size.Arows = A->GetDescriptor().GetDim(0);
		matrix_size.Bcols = B->GetDescriptor().GetDim(1);
		matrix_size.Brows = B->GetDescriptor().GetDim(0);
		matrix_size.Ccols = C->GetDescriptor().GetDim(1);
		matrix_size.Crows = C->GetDescriptor().GetDim(0);

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = sum_subtract_int( handle, A, B, C, "+", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = sum_subtract_float( handle, A, B, C, "+", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = sum_subtract_double( handle, A, B, C, "+", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = sum_subtract_ComplexFloat( handle, A, B, C, "+", matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = sum_subtract_ComplexDouble( handle, A, B, C, "+", matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::add( Array<int> *A, Array<int> *B, Array<int> *C );
	template cublasStatus_t cublasOperations<float>::add( Array<float> *A, Array<float> *B, Array<float> *C );
	template cublasStatus_t cublasOperations<double>::add( Array<double> *A, Array<double> *B, Array<double> *C );
	template cublasStatus_t cublasOperations<ComplexFloat>::add( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C );
	template cublasStatus_t cublasOperations<ComplexDouble>::add( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C );

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::sum_subtract_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template cublasStatus_t cublasOperations<float>::sum_subtract_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::sum_subtract_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::sum_subtract_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, std::string ID, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::sum_subtract_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template cublasStatus_t cublasOperations<int>::sum_subtract_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::sum_subtract_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::sum_subtract_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		float alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(float), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( float ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( float ), cudaMemcpyHostToDevice ) );

		CUBLAS_CALL( cublasSgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ); );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;

		// copy result from device to host
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( float ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::sum_subtract_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template cublasStatus_t cublasOperations<int>::sum_subtract_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::sum_subtract_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::sum_subtract_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		double alpha = 0, beta = 0;
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha, beta, op1, op2 , ID);	

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(double), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( double ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( double ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( double ), cudaMemcpyHostToDevice ) );

		CUBLAS_CALL( cublasDgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ); );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;

		// copy result from device to host
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( double ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}


	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template cublasStatus_t cublasOperations<int>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexFloat alpha2 = ComplexFloat( 0, 0 );
		ComplexFloat beta2 = ComplexFloat( 0, 0 );
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuFloatComplex alpha = make_cuFloatComplex( alpha2.real(), alpha2.imag() );
		cuFloatComplex beta = make_cuFloatComplex( beta2.real(), beta2.imag() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuFloatComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyHostToDevice ) );

		CUBLAS_CALL( cublasCgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ); );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;

		// copy result from device to host
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( cuFloatComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template<typename TElement>
	cublasStatus_t cublasOperations<TElement>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, std::string ID, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template cublasStatus_t cublasOperations<int>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, std::string ID, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, std::string ID, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::sum_subtract_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, std::string ID, _matrixSize matrix_size )
	{
		cudaError_t error;

		ComplexDouble alpha2 = ComplexDouble( 0, 0 );
		ComplexDouble beta2 = ComplexDouble( 0, 0 );
		cublasOperation_t op1 = CUBLAS_OP_N, op2 = CUBLAS_OP_N;

		define_sum_subtract_transpose_hermitian_operation( alpha2, beta2, op1, op2 , ID);	
		cuDoubleComplex alpha = make_cuDoubleComplex( alpha2.real(), alpha2.imag() );
		cuDoubleComplex beta = make_cuDoubleComplex( beta2.real(), beta2.imag() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuDoubleComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy to GPU
		CUDA_CALL( cudaMemcpy( d_A, A->m_data.GetElements(), A->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_B, B->m_data.GetElements(), B->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );
		CUDA_CALL( cudaMemcpy( d_C, C->m_data.GetElements(), C->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyHostToDevice ) );

		CUBLAS_CALL( cublasZgeam( handle, op1, op2, matrix_size.Arows, matrix_size.Bcols, &alpha, d_A, matrix_size.Arows, &beta, d_B, matrix_size.Brows, d_C, matrix_size.Crows ); );
	
		cudaDeviceSynchronize();
		error = cudaGetLastError();

		int incx = 1;
		int incy = 1;
		int lda = matrix_size.Arows;
		int ldb = matrix_size.Arows;

		// copy result from device to host
		CUDA_CALL( cudaMemcpy( C->m_data.GetElements(), d_C, C->m_data.m_numElements*sizeof( cuDoubleComplex ), cudaMemcpyDeviceToHost ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

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
			CUBLAS_CALL( multiply_MatrixMatrix_Matrix( A, B, C, matrix_size ) );
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

	// implementation of Matrix x Matrix = Matrix

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = multiply_MatrixMatrix_Matrix_int( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = multiply_MatrixMatrix_Matrix_float( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = multiply_MatrixMatrix_Matrix_double( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = multiply_MatrixMatrix_Matrix_ComplexFloat( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = multiply_MatrixMatrix_Matrix_ComplexDouble( handle, A, B, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix( Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix( Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix( Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixMatrix_Matrix( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(float), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(float), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const float alpha = 1.0;
		const float beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		//CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(float), d_C, incx, C->m_data.GetElements(), incy );
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(double), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(double), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const double alpha = 1.0;
		const double beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		//CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(double), d_C, incx, C->m_data.GetElements(), incy );
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixMatrix_Matrix_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuFloatComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuFloatComplex), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(cuFloatComplex), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuFloatComplex), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const cuFloatComplex alpha = make_cuFloatComplex( 1.0f, 0 );
		const cuFloatComplex beta = make_cuFloatComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		//CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(cuFloatComplex), d_C, incx, C->m_data.GetElements(), incy );
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuFloatComplex), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixMatrix_Matrix_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			*C = C->transpose();
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuDoubleComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(cuDoubleComplex), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const cuDoubleComplex alpha = make_cuDoubleComplex( 1.0f, 0 );
		const cuDoubleComplex beta = make_cuDoubleComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Bcols, matrix_size.Acols, &alpha, d_A, matrix_size.Arows, d_B, matrix_size.Brows, &beta, d_C, matrix_size.Crows) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		//CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(cuDoubleComplex), d_C, incx, C->m_data.GetElements(), incy );
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// implementation of Vector(transposed) x Vector = scalar (dot product)

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = multiply_VectorTranspVector_Scalar_int( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = multiply_VectorTranspVector_Scalar_float( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = multiply_VectorTranspVector_Scalar_double( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = multiply_VectorTranspVector_Scalar_ComplexFloat( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = multiply_VectorTranspVector_Scalar_ComplexDouble( handle, A, B, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar( Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar( Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar( Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspVector_Scalar( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(float) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(float) * size_B;

		// define device memory
		float *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(*d_A), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(*d_B), (void **) &d_B ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(*d_A), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(*d_B), B->m_data.GetElements(), 1, d_B, 1 ) );

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSdot( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );
	
		CUDA_CALL( cudaDeviceSynchronize()) ;

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
	
		*C->m_data.m_data = d_C;

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(double) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(double) * size_B;

		// define device memory
		double *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(*d_A), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(*d_B), (void **) &d_B ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(*d_A), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(*d_B), B->m_data.GetElements(), 1, d_B, 1 ) );

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDdot( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );
	
		CUDA_CALL( cudaDeviceSynchronize()) ;

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
	
		*C->m_data.m_data = d_C;

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspVector_Scalar_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(cuFloatComplex) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(cuFloatComplex) * size_B;

		// define device memory
		cuFloatComplex *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(*d_A), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(*d_B), (void **) &d_B ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(*d_A), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(*d_B), B->m_data.GetElements(), 1, d_B, 1 ) );

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCdotu( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );
	
		CUDA_CALL( cudaDeviceSynchronize()) ;

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
	
		*C->m_data.m_data = ComplexFloat( d_C.x, d_C.y );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspVector_Scalar_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int mem_size_A = sizeof(cuDoubleComplex) * size_A;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int mem_size_B = sizeof(cuDoubleComplex) * size_B;

		// define device memory
		cuDoubleComplex *d_A, *d_B, d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(*d_A), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(*d_B), (void **) &d_B ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(*d_A), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(*d_B), B->m_data.GetElements(), 1, d_B, 1 ) );

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZdotu( handle, imax( matrix_size.Acols, matrix_size.Arows ), d_A, 1, d_B, 1, &d_C ) );
	
		CUDA_CALL( cudaDeviceSynchronize()) ;

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
	
		*C->m_data.m_data = ComplexDouble( d_C.x, d_C.y );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = multiply_VectorVectorTransp_Matrix_int( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = multiply_VectorVectorTransp_Matrix_float( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = multiply_VectorVectorTransp_Matrix_double( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = multiply_VectorVectorTransp_Matrix_ComplexFloat( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = multiply_VectorVectorTransp_Matrix_ComplexDouble( handle, A, B, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix( Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix( Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix( Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorVectorTransp_Matrix( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(float), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(float), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(float), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const float alpha = 1.0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSger( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(float), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(double), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(double), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(double), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const double alpha = 1.0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDger( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(double), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorVectorTransp_Matrix_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuFloatComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(cuFloatComplex), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(cuFloatComplex), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuFloatComplex), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const cuFloatComplex alpha = make_cuFloatComplex( 1.0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgeru( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuFloatComplex), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorVectorTransp_Matrix_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size )
	{
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuDoubleComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(cuDoubleComplex), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(cuDoubleComplex), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), C->m_data.GetElements(), matrix_size.Crows, d_C, matrix_size.Crows ) );

		const cuDoubleComplex alpha = make_cuDoubleComplex( 1.0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		//// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgeru( handle, matrix_size.Crows, matrix_size.Ccols, &alpha, d_A, incx, d_B, incy, d_C, lda ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetMatrix( matrix_size.Crows, matrix_size.Ccols, sizeof(cuDoubleComplex), d_C, lda, C->m_data.GetElements(), ldb ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// Implementation of matrix x vector = vector OR vector(transp) x matrix = vector(transp)

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = multiply_MatrixVector_Vector_int( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = multiply_MatrixVector_Vector_float( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = multiply_MatrixVector_Vector_double( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = multiply_MatrixVector_Vector_ComplexFloat( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = multiply_MatrixVector_Vector_ComplexDouble( handle, A, B, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector( Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector( Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector( Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixVector_Vector( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size )
		{	
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
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(float), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(float), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(float), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(float), C->m_data.GetElements(), 1, d_C, 1 ) );

		const float alpha = 1.0f;
		const float beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(float), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size )
	{	
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
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(double), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(double), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(double), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(double), C->m_data.GetElements(), 1, d_C, 1 ) );

		const double alpha = 1.0f;
		const double beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(double), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::multiply_MatrixVector_Vector_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size )
	{	
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
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuFloatComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuFloatComplex), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(cuFloatComplex), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(cuFloatComplex), C->m_data.GetElements(), 1, d_C, 1 ) );

		const cuFloatComplex alpha = make_cuFloatComplex( 1.0f, 0 );
		const cuFloatComplex beta = make_cuFloatComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(cuFloatComplex), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::multiply_MatrixVector_Vector_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size )
	{	
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
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuDoubleComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Arows, matrix_size.Acols, sizeof(cuDoubleComplex), A->m_data.GetElements(), matrix_size.Arows, d_A, matrix_size.Arows ) );
		CUBLAS_CALL( cublasSetVector( size_B, sizeof(cuDoubleComplex), B->m_data.GetElements(), 1, d_B, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(cuDoubleComplex), C->m_data.GetElements(), 1, d_C, 1 ) );

		const cuDoubleComplex alpha = make_cuDoubleComplex( 1.0f, 0 );
		const cuDoubleComplex beta = make_cuDoubleComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Crows;
		const int ldb = lda;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgemv( handle, CUBLAS_OP_N, matrix_size.Arows, matrix_size.Acols, &alpha, d_A, lda, d_B, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Crows, sizeof(cuDoubleComplex), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	// vector(transp) x matrix = vector(transp)

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp( Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		cublasHandle_t handle;
		cudaError_t error;

		//std::cout << typeid(TElement).name() << std::endl;
		CUBLAS_CALL(cublasCreate(&handle));
		try
		{
			if( !_strcmpi(typeid(TElement).name(),"int") )
				stat = multiply_VectorTranspMatrix_VectorTransp_int( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"float") )
				stat = multiply_VectorTranspMatrix_VectorTransp_float( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"double") )
				stat = multiply_VectorTranspMatrix_VectorTransp_double( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexFloat") || !_strcmpi(typeid(TElement).name(),"class std::complex<float>") )
				stat = multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( handle, A, B, C, matrix_size );
			else if( !_strcmpi(typeid(TElement).name(),"ComplexDouble") || !_strcmpi(typeid(TElement).name(),"class std::complex<double>") )
				stat = multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( handle, A, B, C, matrix_size );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}

		cudaDeviceSynchronize();
		error = cudaGetLastError();

		// Destroy the handle
		CUBLAS_CALL(cublasDestroy(handle));

		return stat;
	}

	template cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp( Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp( Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp( Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspMatrix_VectorTransp( Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	template cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp( Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp_int( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size )
	{
		// TODO
		cublasStatus_t stat = CUBLAS_STATUS_NOT_INITIALIZED;
		return stat;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp_float( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		float *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(float), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(float), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(float), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(float), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(float), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(float), C->m_data.GetElements(), 1, d_C, 1 ) );

		const float alpha = 1.0f;
		const float beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasSgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Ccols, sizeof(float), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp_double( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		double *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(double), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(double), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(double), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(double), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(double), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(double), C->m_data.GetElements(), 1, d_C, 1 ) );

		const double alpha = 1.0f;
		const double beta = 0;
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasDgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Ccols, sizeof(double), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexFloat>::multiply_VectorTranspMatrix_VectorTransp_ComplexFloat( cublasHandle_t handle, Array<ComplexFloat> *A, Array<ComplexFloat> *B, Array<ComplexFloat> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuFloatComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuFloatComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuFloatComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuFloatComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(cuFloatComplex), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(cuFloatComplex), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(cuFloatComplex), C->m_data.GetElements(), 1, d_C, 1 ) );

		const cuFloatComplex alpha = make_cuFloatComplex( 1.0f, 0 );
		const cuFloatComplex beta = make_cuFloatComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasCgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Ccols, sizeof(cuFloatComplex), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	cublasStatus_t cublasOperations<TElement>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<TElement> *A, Array<TElement> *B, Array<TElement> *C, _matrixSize matrix_size )
	{
		return CUBLAS_STATUS_NOT_SUPPORTED;
	}

	cublasStatus_t cublasOperations<int>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<int> *A, Array<int> *B, Array<int> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<float>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<float> *A, Array<float> *B, Array<float> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<double>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<double> *A, Array<double> *B, Array<double> *C, _matrixSize matrix_size );
	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size );

	cublasStatus_t cublasOperations<ComplexDouble>::multiply_VectorTranspMatrix_VectorTransp_ComplexDouble( cublasHandle_t handle, Array<ComplexDouble> *A, Array<ComplexDouble> *B, Array<ComplexDouble> *C, _matrixSize matrix_size )
	{	
		if (C->GetDescriptor().GetDim(0) == 1)
		{		
			matrix_size.Acols = A->GetDescriptor().GetDim(1);
			matrix_size.Arows = A->GetDescriptor().GetDim(0);
			matrix_size.Bcols = B->GetDescriptor().GetDim(1);
			matrix_size.Brows = B->GetDescriptor().GetDim(0);
			matrix_size.Ccols = C->GetDescriptor().GetDim(1);
			matrix_size.Crows = C->GetDescriptor().GetDim(0);
		};
	
		CUBLAS_CALL( cublasInit() );

		// define host memory size for matrices A and B
		unsigned int size_A = matrix_size.Acols * matrix_size.Arows;
		unsigned int size_B = matrix_size.Bcols * matrix_size.Brows;
		unsigned int size_C = matrix_size.Ccols * matrix_size.Crows;

		// define device memory
		cuDoubleComplex *d_A, *d_B, *d_C;
	
		cudaDeviceProp deviceProp;
		int devID = 0;
		CUDA_CALL(cudaGetDeviceProperties(&deviceProp, devID));
	
		// allocate memmory
		CUBLAS_CALL( cublasAlloc( size_A, sizeof(cuDoubleComplex), (void **) &d_A ) );
		CUBLAS_CALL( cublasAlloc( size_B, sizeof(cuDoubleComplex), (void **) &d_B ) );
		CUBLAS_CALL( cublasAlloc( size_C, sizeof(cuDoubleComplex), (void **) &d_C ) );

		// copy vectors to GPU
		CUBLAS_CALL( cublasSetMatrix( matrix_size.Brows, matrix_size.Bcols, sizeof(cuDoubleComplex), B->m_data.GetElements(), matrix_size.Brows, d_B, matrix_size.Brows ) );
		CUBLAS_CALL( cublasSetVector( size_A, sizeof(cuDoubleComplex), A->m_data.GetElements(), 1, d_A, 1 ) );
		CUBLAS_CALL( cublasSetVector( size_C, sizeof(cuDoubleComplex), C->m_data.GetElements(), 1, d_C, 1 ) );

		const cuDoubleComplex alpha = make_cuDoubleComplex( 1.0f, 0 );
		const cuDoubleComplex beta = make_cuDoubleComplex( 0, 0 );
		const int incx = 1;
		const int incy = 1;
		const int lda = matrix_size.Arows;
		const int ldb = matrix_size.Brows;

		// CALL CUBLAS FUNCTION
		CUBLAS_CALL( cublasZgemv( handle, CUBLAS_OP_T, matrix_size.Bcols, matrix_size.Brows, &alpha, d_B, ldb, d_A, incx, &beta, d_C, incy ) );
	
		cudaDeviceSynchronize();
		cudaError_t error = cudaGetLastError();

		// copy result from device to host
		CUBLAS_CALL( cublasGetVector( matrix_size.Ccols, sizeof(cuDoubleComplex), d_C, incx, C->m_data.GetElements(), incy ) );

		// free memory
		CUBLAS_CALL( cublasFree( d_A ) );
		CUBLAS_CALL( cublasFree( d_B ) );
		CUBLAS_CALL( cublasFree( d_C ) );

		return CUBLAS_STATUS_SUCCESS;
	}

	template <typename TElement>
	void cublasOperations<TElement>::from_permutation_vector_to_permutation_matrix( Array<TElement> *pivotMatrix, Array<int> *pivotVector )
	{
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
		}
		//pivotMatrix->print();
	}

	template void cublasOperations<float>::from_permutation_vector_to_permutation_matrix( Array<float> *pivotMatrix, Array<int> *pivotVector );
	template void cublasOperations<double>::from_permutation_vector_to_permutation_matrix( Array<double> *pivotMatrix, Array<int> *pivotVector );
	template void cublasOperations<ComplexFloat>::from_permutation_vector_to_permutation_matrix( Array<ComplexFloat> *pivotMatrix, Array<int> *pivotVector );
	template void cublasOperations<ComplexDouble>::from_permutation_vector_to_permutation_matrix( Array<ComplexDouble> *pivotMatrix, Array<int> *pivotVector );
}