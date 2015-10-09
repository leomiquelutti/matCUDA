#include <time.h>

#include "common.h"

#include "curandOperations.h"

namespace matCUDA
{
	template<> curandStatus_t curandOperations<ComplexFloat>::rand( Array<ComplexFloat> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;

		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Allocate n doubles on device */ 
		CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float))); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n floats on device */ 
		CURAND_CALL(curandGenerateUniform(gen, devData, n)); 
	
		/* Copy device memory to host */ 
		CUDA_CALL(cudaMemcpy(out->m_data.GetElements(), devData, n * sizeof(float), cudaMemcpyDeviceToHost));
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 
		CUDA_CALL(cudaFree(devData)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<ComplexDouble>::rand( Array<ComplexDouble> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
		curandGenerator_t gen; 
		double *devData;
	
		/* Allocate n doubles on device */ 
		CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double))); 
		
		/////***** performance test *****/////
		//CUDA_CALL( cudaDeviceSynchronize() );
		//tic();
		//for( int i = 0; i < 10; i++ ) {

		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n doubles on device */ 
		CURAND_CALL(curandGenerateUniformDouble(gen, devData, n)); 
		
		//CURAND_CALL(curandDestroyGenerator(gen)); 
		//}
		//CUDA_CALL( cudaDeviceSynchronize() );
		//toc();
		////***** end of performance test *****/////
		
		/* Copy device memory to host */ 
		CUDA_CALL(cudaMemcpy(out->m_data.GetElements(), devData, n * sizeof(double), cudaMemcpyDeviceToHost));
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 
		CUDA_CALL(cudaFree(devData)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<float>::rand( Array<float> *out )
	{
		size_t n = out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;
	
		/* Allocate n doubles on device */ 
		CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float))); 
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n floats on device */ 
		CURAND_CALL(curandGenerateUniform(gen, devData, n)); 
	
		/* Copy device memory to host */ 
		CUDA_CALL(cudaMemcpy(out->m_data.GetElements(), devData, n * sizeof(float), cudaMemcpyDeviceToHost));
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 
		CUDA_CALL(cudaFree(devData)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<double>::rand( Array<double> *out )
	{
		size_t n = out->m_data.m_numElements;
		curandGenerator_t gen; 
		double *devData;
	
		/* Allocate n doubles on device */ 
		CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(double))); 
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n doubles on device */ 
		CURAND_CALL(curandGenerateUniformDouble(gen, devData, n)); 
	
		/* Copy device memory to host */ 
		CUDA_CALL(cudaMemcpy(out->m_data.GetElements(), devData, n * sizeof(double), cudaMemcpyDeviceToHost));
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 
		CUDA_CALL(cudaFree(devData)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<ComplexFloat>::rand_zerocopy( Array<ComplexFloat> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;
	
		/* set device pointer to host memory */
		CUDA_CALL( cudaHostGetDevicePointer( &devData, out->m_data.GetElements(), 0 ) );
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n floats on device */ 
		CURAND_CALL(curandGenerateUniform(gen, devData, n)); 
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<ComplexDouble>::rand_zerocopy( Array<ComplexDouble> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
		curandGenerator_t gen; 
		double *devData;
	
		/* set device pointer to host memory */
		CUDA_CALL( cudaHostGetDevicePointer( &devData, out->m_data.GetElements(), 0 ) );
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
		
		/////***** performance test *****/////
		//CUDA_CALL( cudaDeviceSynchronize() );
		//tic();
		//for( int i = 0; i < 10; i++ ) {
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n doubles on device */ 
		CURAND_CALL(curandGenerateUniformDouble(gen, devData, n)); 
		
		//}
		//CUDA_CALL( cudaDeviceSynchronize() );
		//toc();
		////***** end of performance test *****/////
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<float>::rand_zerocopy( Array<float> *out )
	{
		size_t n = out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;
	
		/* set device pointer to host memory */
		CUDA_CALL( cudaHostGetDevicePointer( &devData, out->m_data.GetElements(), 0 ) );
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n floats on device */ 
		CURAND_CALL(curandGenerateUniform(gen, devData, n)); 
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 

		return CURAND_STATUS_SUCCESS;
	}

	template<> curandStatus_t curandOperations<double>::rand_zerocopy( Array<double> *out )
	{
		size_t n = out->m_data.m_numElements;
		curandGenerator_t gen; 
		double *devData;
	
		// pass host pointer to device
		CUDA_CALL( cudaHostGetDevicePointer( &devData, out->m_data.GetElements(), 0 ) );
	
		/* Create pseudo-random number generator */ 
		CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	
		/* Set seed */ 
		CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock())); 
	
		/* Generate n doubles on device */ 
		CURAND_CALL(curandGenerateUniformDouble(gen, devData, n)); 
	
		/* Cleanup */ 
		CURAND_CALL(curandDestroyGenerator(gen)); 

		return CURAND_STATUS_SUCCESS;
	}
}