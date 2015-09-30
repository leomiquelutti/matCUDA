#include <fstream>
#include <iostream>
#include <stack>
#include <boost/exception/all.hpp>

#include "common.h"

#include "Array.h"

namespace matCUDA
{
	// identity matrix generation
	template <typename TElement>
	Array<TElement> eye( index_t N )
	{	
		Array<TElement> result( N, N );

		//// CPU version
		//for (int i = 0; i < result.GetDescriptor().GetDim(0); i++) {
		//	for (int j = 0; j < result.GetDescriptor().GetDim(1); j++) 
		//		result(i,j) = ( i == j );
		//}

		// GPU version
		cudaEye<TElement>( result.data(), N );

		return result;
	}

	template Array<int> eye( index_t N );
	template Array<float> eye( index_t N );
	template Array<double> eye( index_t N );
	template Array<ComplexFloat> eye( index_t N );
	template Array<ComplexDouble> eye( index_t N );

	// dpss generation following Gruenbacher and Hummels, 1994
	template <typename TElement>
	Array<TElement> dpss( index_t N, double NW, index_t degree )
	{
		cusolverStatus_t stat = CUSOLVER_STATUS_NOT_INITIALIZED;
		cusolverOperations<TElement> op;
		Array<TElement> eigenvector( N );

		try
		{
			stat = op.dpss( &eigenvector, N, NW, degree );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return eigenvector;
	}

	template Array<float> dpss( index_t N, double NW, index_t degree );
	template Array<double> dpss( index_t N, double NW, index_t degree );

	Array<ComplexFloat> fft( Array<float> *in )
	{
		cufftResult_t stat = CUFFT_NOT_IMPLEMENTED;
		cufftOperations<ComplexFloat> op;
		Array<ComplexFloat> result( in->GetDescriptor().GetDim(0)/2+1, in->GetDescriptor().GetDim(1) );

		try
		{
			stat = op.fft_stream( in, &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return result;
	}

	Array<ComplexDouble> fft( Array<double> *in )
	{
		cufftResult_t stat = CUFFT_NOT_IMPLEMENTED;
		cufftOperations<ComplexDouble> op;
		Array<ComplexDouble> result( in->GetDescriptor().GetDim(0)/2+1, in->GetDescriptor().GetDim(1) );

		try
		{
			stat = op.fft_stream( in, &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return result;
	}

	// read files for unit tests
	template<> Array<ComplexFloat> read_file_vector( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			size_t dim = std::stoi( aux );
			Array<ComplexFloat> result(dim);

			double real, imag;
			for( int i = 0; i < dim; i++ ) {
				std::getline( inputFile, aux );
				real = std::stof( aux );
				std::getline( inputFile, aux );
				imag = std::stof( aux );
				result( i ) = ComplexFloat( real, imag );
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<ComplexFloat> (1);
		}
	}

	template<> Array<ComplexDouble> read_file_vector( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			size_t dim = std::stoi( aux );
			Array<ComplexDouble> result(dim);

			double real, imag;
			for( int i = 0; i < dim; i++ ) {
				std::getline( inputFile, aux );
				real = std::stod( aux );
				std::getline( inputFile, aux );
				imag = std::stod( aux );
				result( i ) = ComplexDouble( real, imag );
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<ComplexDouble> (1);
		}
	}

	template<> Array<float> read_file_vector( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			size_t dim = std::stoi( aux );
			Array<float> result(dim);

			for( int i = 0; i < dim; i++ ) {
				std::getline( inputFile, aux );
				result( i ) = std::stof( aux );
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<float> (1);
		}
	}

	template<> Array<double> read_file_vector( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			size_t dim = std::stoi( aux );
			Array<double> result(dim);

			for( int i = 0; i < dim; i++ ) {
				std::getline( inputFile, aux );
				result( i ) = std::stof( aux );
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<double> (1);
		}
	}

	template<> Array<ComplexFloat> read_file_matrix( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			index_t idxComma = aux.find_first_of( ',', 0 );
			size_t rows = std::stoi( aux.substr( 0, idxComma - 1 ) );
			size_t cols = std::stoi( aux.substr( idxComma + 1, aux.size() ) );
			Array<ComplexFloat> result(rows,cols);

			double real, imag;
			for( int i = 0; i < cols; i++ ) {
				for( int j = 0; j < rows; j++ ) {
				std::getline( inputFile, aux );
				real = std::stof( aux );
				std::getline( inputFile, aux );
				imag = std::stof( aux );
				result( j, i ) = ComplexFloat( real, imag );
				}
			}

			inputFile.close();

			return  result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<ComplexFloat> (1,1);
		}
	}

	template<> Array<ComplexDouble> read_file_matrix( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			index_t idxComma = aux.find_first_of( ',', 0 );		
			size_t rows = std::stoi( aux.substr( 0, idxComma - 1 ) );
			size_t cols = std::stoi( aux.substr( idxComma + 1, aux.size() ) );
			Array<ComplexDouble> result(rows,cols);

			double real, imag;
			for( int i = 0; i < cols; i++ ) {
				for( int j = 0; j < rows; j++ ) {
				std::getline( inputFile, aux );
				real = std::stod( aux );
				std::getline( inputFile, aux );
				imag = std::stod( aux );
				result( j, i ) = ComplexDouble( real, imag );
				}
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<ComplexDouble> (1,1);
		}
	}

	template<> Array<float> read_file_matrix( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			index_t idxComma = aux.find_first_of( ',', 0 );
			size_t rows = std::stoi( aux.substr( 0, idxComma - 1 ) );
			size_t cols = std::stoi( aux.substr( idxComma + 1, aux.size() ) );
			Array<float> result(rows,cols);

			for( int i = 0; i < cols; i++ ) {
				for( int j = 0; j < rows; j++ ) {
					std::getline( inputFile, aux );
					result( j, i ) = std::stof( aux );
				}
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<float> (1,1);
		}
	}

	template<> Array<double> read_file_matrix( std::string s )
	{
		const size_t lim = 128;
		std::string aux;
		std::ifstream inputFile;
		inputFile.open( s );
		if( inputFile.good() )
		{
			std::getline( inputFile, aux );
			index_t idxComma = aux.find_first_of( ',', 0 );
			size_t rows = std::stoi( aux.substr( 0, idxComma - 1 ) );
			size_t cols = std::stoi( aux.substr( idxComma + 1, aux.size() ) );
			Array<double> result(rows,cols);

			for( int i = 0; i < cols; i++ ) {
				for( int j = 0; j < rows; j++ ) {
					std::getline( inputFile, aux );
					result( j, i ) = std::stof( aux );
				}
			}

			inputFile.close();

			return result;
		}
		else
		{
			std::cout<< "Could not open " << s << std::endl;
			EXIT_FAILURE;

			return Array<double> (1,1);
		}
	}

	template <typename TElement>
	Array<TElement> rand(index_t u)
	{
		curandStatus_t stat = CURAND_STATUS_NOT_INITIALIZED;
		curandOperations<TElement> op;
		Array<TElement> result( u );

		try
		{
			stat = op.rand( &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return result;
	}

	template Array<float> rand(index_t u);
	template Array<double> rand(index_t u);
	template Array<ComplexFloat> rand(index_t u);
	template Array<ComplexDouble> rand(index_t u);

	template <typename TElement>
	Array<TElement> rand(index_t u1, index_t u2)
	{
		curandStatus_t stat = CURAND_STATUS_NOT_INITIALIZED;
		curandOperations<TElement> op;
		Array<TElement> result( u1, u2 );

		try
		{
			//stat = op.rand( &result );
			stat = op.rand_zerocopy( &result );
		}
		catch(std::exception &e)
		{
			std::cerr << boost::diagnostic_information(e);
		}
	
		return result;
	}

	template Array<float> rand(index_t u1, index_t u2);
	template Array<double> rand(index_t u1, index_t u2);
	template Array<ComplexFloat> rand(index_t u1, index_t u2);
	template Array<ComplexDouble> rand(index_t u1, index_t u2);

	std::stack<clock_t> tictoc_stack;
	void tic() {
		tictoc_stack.push(clock());
	}
	void toc() {
		std::cout << "Time elapsed: "
				  << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
				  << " seconds"
				  << std::endl;
		tictoc_stack.pop();
	}
	long double toc( long double in ) {
		long double out = in + ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC;
		return out;
	}
}

//template <typename TElement>
//Array<TElement> rand(index_t u, ...)
//{
//	va_list args;
//	va_start(args, u);
//
//	int count = u;
//
//	for(int i = 1; i < count; i++)
//		m_indexer->m_pos[i] = va_arg(args, index_t);
//
//	va_end(args);
//	return Array<TElement>(1);
//}
//
//template Array<int> rand(index_t u, ...);
//template Array<float> rand(index_t u, ...);
//template Array<double> rand(index_t u, ...);
//template Array<ComplexFloat> rand(index_t u, ...);
//template Array<ComplexDouble> rand(index_t u, ...);