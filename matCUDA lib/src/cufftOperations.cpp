#include "common.h"

#include "cufftOperations.h"

namespace matCUDA
{
	template<> cufftResult_t cufftOperations<ComplexFloat>::fft( Array<ComplexFloat> *in, Array<ComplexFloat> *out )
	{
		cufftHandle plan;
		cufftComplex *dataIn, *dataOut;

		size_t NX = in->GetDescriptor().GetDim(0);
		size_t NY = in->GetDescriptor().GetDim(1);
		size_t BATCH = in->GetDescriptor().GetDim(1);

		size_t size_in = sizeof(cufftComplex)*NX*BATCH;
		size_t size_out = size_in;
		size_t workSize[1];

		CUDA_CALL( cudaMalloc((void**)&dataIn, size_in) );
		CUDA_CALL( cudaMalloc((void**)&dataOut, size_out) );

		CUDA_CALL( cudaMemcpy( dataIn, in->m_data.GetElements(), size_in, cudaMemcpyHostToDevice ) );

		CUFFT_CALL( cufftCreate( &plan ) );
		CUFFT_CALL( cufftMakePlan1d( plan, NX, CUFFT_C2C, BATCH, workSize ) ); 

		CUFFT_CALL( cufftExecC2C(plan, dataIn, dataOut, CUFFT_FORWARD) );

		CUDA_CALL( cudaThreadSynchronize() );

		CUDA_CALL( cudaMemcpy( out->m_data.GetElements(), dataOut, size_out, cudaMemcpyDeviceToHost ) );

		CUFFT_CALL( cufftDestroy(plan) );

		CUDA_CALL( cudaFree(dataIn) );
		CUDA_CALL( cudaFree(dataOut) );
		
		return CUFFT_SUCCESS;
	}

	template<> cufftResult_t cufftOperations<ComplexDouble>::fft( Array<ComplexDouble> *in, Array<ComplexDouble> *out )
	{
		cufftHandle plan;
		cufftDoubleComplex *dataIn, *dataOut;

		size_t NX = in->GetDescriptor().GetDim(0);
		size_t NY = in->GetDescriptor().GetDim(1);
		size_t BATCH = in->GetDescriptor().GetDim(1);

		size_t size_in = sizeof(cufftDoubleComplex)*NX*BATCH;
		size_t size_out = size_in;
		size_t workSize[1];

		CUDA_CALL( cudaMalloc((void**)&dataIn, size_in) );
		CUDA_CALL( cudaMalloc((void**)&dataOut, size_out) );

		CUDA_CALL( cudaMemcpy( dataIn, in->m_data.GetElements(), size_in, cudaMemcpyHostToDevice ) );

		CUFFT_CALL( cufftCreate( &plan ) );
		CUFFT_CALL( cufftMakePlan1d( plan, NX, CUFFT_Z2Z, BATCH, workSize ) ); 

		CUFFT_CALL( cufftExecZ2Z(plan, dataIn, dataOut, CUFFT_FORWARD) );

		CUDA_CALL( cudaThreadSynchronize() );

		CUDA_CALL( cudaMemcpy( out->m_data.GetElements(), dataOut, size_out, cudaMemcpyDeviceToHost ) );

		CUFFT_CALL( cufftDestroy(plan) );

		CUDA_CALL( cudaFree(dataIn) );
		CUDA_CALL( cudaFree(dataOut) );
		
		return CUFFT_SUCCESS;
	}

	template <typename TElement>
	cufftResult_t cufftOperations<TElement>::fft( Array<TElement> *in, Array<TElement> *out )
	{
		return CUFFT_EXEC_FAILED;
	}

	template cufftResult_t cufftOperations<int>::fft( Array<int> *in, Array<int> *out );
	template cufftResult_t cufftOperations<float>::fft( Array<float> *in, Array<float> *out );
	template cufftResult_t cufftOperations<double>::fft( Array<double> *in, Array<double> *out );

	cufftResult_t cufftOperations<ComplexFloat>::fft( Array<float> *in, Array<ComplexFloat> *out )
	{
		cufftHandle plan;
		float *dataIn;
		cufftComplex *dataOut;

		size_t NX = in->GetDescriptor().GetDim(0);
		size_t NY = in->GetDescriptor().GetDim(1);
		size_t BATCH = in->GetDescriptor().GetDim(1);

		size_t size_in = sizeof(float)*NX*BATCH;
		size_t size_out = sizeof(cufftComplex)*BATCH*( 1 + NX/2 );
		size_t workSize[1];

		CUDA_CALL( cudaMalloc((void**)&dataIn, size_in) );
		CUDA_CALL( cudaMalloc((void**)&dataOut, size_out) );

		CUDA_CALL( cudaMemcpy( dataIn, in->m_data.GetElements(), size_in, cudaMemcpyHostToDevice ) );

		CUFFT_CALL( cufftCreate( &plan ) );
		CUFFT_CALL( cufftMakePlan1d( plan, NX, CUFFT_R2C, BATCH, workSize ) ); 

		CUFFT_CALL( cufftExecR2C( plan, dataIn, dataOut ) );

		CUDA_CALL( cudaThreadSynchronize() );

		CUDA_CALL( cudaMemcpy( out->m_data.GetElements(), dataOut, size_out, cudaMemcpyDeviceToHost ) );

		CUFFT_CALL( cufftDestroy(plan) );

		CUDA_CALL( cudaFree(dataIn) );
		CUDA_CALL( cudaFree(dataOut) );
		
		return CUFFT_SUCCESS;
	}

	cufftResult_t cufftOperations<ComplexDouble>::fft( Array<double> *in, Array<ComplexDouble> *out )
	{
		cufftHandle plan;
		double *dataIn;
		cufftDoubleComplex *dataOut;

		size_t NX = in->GetDescriptor().GetDim(0);
		size_t NY = in->GetDescriptor().GetDim(1);
		size_t BATCH = in->GetDescriptor().GetDim(1);

		size_t size_in = sizeof(double)*NX*BATCH;
		size_t size_out = sizeof(cufftDoubleComplex)*BATCH*( 1 + NX/2 );
		size_t workSize[1];

		CUDA_CALL( cudaMalloc((void**)&dataIn, size_in) );
		CUDA_CALL( cudaMalloc((void**)&dataOut, size_out) );

		CUDA_CALL( cudaMemcpy( dataIn, in->m_data.GetElements(), size_in, cudaMemcpyHostToDevice ) );

		CUFFT_CALL( cufftCreate( &plan ) );
		CUFFT_CALL( cufftMakePlan1d( plan, NX, CUFFT_D2Z, BATCH, workSize ) ); 

		CUFFT_CALL( cufftExecD2Z( plan, dataIn, dataOut ) );

		CUDA_CALL( cudaThreadSynchronize() );

		CUDA_CALL( cudaMemcpy( out->m_data.GetElements(), dataOut, size_out, cudaMemcpyDeviceToHost ) );

		CUFFT_CALL( cufftDestroy(plan) );

		CUDA_CALL( cudaFree(dataIn) );
		CUDA_CALL( cudaFree(dataOut) );
		
		return CUFFT_SUCCESS;
	}
}