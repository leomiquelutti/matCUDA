#include "curandOperations.h"
#include <time.h>

namespace matCUDA
{
	template<> curandStatus_t curandOperations<ComplexFloat>::rand( Array<ComplexFloat> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;
	
		/* Allocate n floats on device */ 
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

	template<> curandStatus_t curandOperations<ComplexDouble>::rand( Array<ComplexDouble> *out )
	{
		size_t n = 2*out->m_data.m_numElements;
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

	template<> curandStatus_t curandOperations<float>::rand( Array<float> *out )
	{
		size_t n = out->m_data.m_numElements;
		curandGenerator_t gen; 
		float *devData;

		/* Allocate n floats on device */ 
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
}