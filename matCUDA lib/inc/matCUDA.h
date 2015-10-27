#pragma once

#define PI 3.141592653589793238462643

#include <vector>
#include <boost/preprocessor.hpp>

#include "allocators.h"

typedef __int32 index_t;

#include <complex>
typedef std::complex<float> ComplexFloat;
typedef std::complex<double> ComplexDouble;

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define NOMINMAX

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int Acols, Arows, Bcols, Brows, Ccols, Crows;
} sMatrixSize;

template<class T> struct MappedHostAllocator;

namespace matCUDA
{
	#undef min
	#undef max

	class ArrayDescriptor;
	class LinearIndexer;
	template <typename TElement, class TAllocator = MappedHostAllocator<TElement>> class ArrayData;

	template <typename TElement>
	class Array
	{
		template <typename TElement> friend class cublasOperations;
		template <typename TElement> friend class cufftOperations;
		template <typename TElement> friend class curandOperations;
		template <typename TElement> friend class cusolverOperations;
	public:

		Array(	ArrayDescriptor &descriptor,
				const TElement& defaulTElement = TElement());
	
		Array(Array &source);

		#include "file.h"

		~Array();
	
		// info from base classes
		const ArrayData<TElement>* GetArrayData() const { return &m_data; }
		ArrayDescriptor& GetDescriptor();
		TElement *data() { return m_data.m_data; };
		size_t			getDim( index_t dim ) { return this->GetDescriptor().GetDim( dim ); };
		size_t			getNDim() { return this->GetDescriptor().GetNDim(); };
		size_t			getNElements() { return (size_t)this->GetDescriptor().GetNumberOfElements(); };

		// operators
		bool operator == (Array<TElement> a);
		bool operator != (Array<TElement> a);
		const TElement& operator () (index_t u, ...) const;
		TElement& operator () (index_t u, ...);
		Array<TElement>& operator = (Array<TElement> &a);
		Array<TElement>& operator = (TElement a);
		Array<TElement> operator + (Array<TElement>  &a); // gpu
		Array<TElement> operator + (TElement a);
		Array<TElement> operator - (Array<TElement> &a); // gpu
		Array<TElement> operator - (TElement a);
		Array<TElement> operator * (Array<TElement> &a); // gpu
		Array<TElement> operator * (TElement a);
		Array<TElement> operator / (Array<TElement> &a); // gpu
		Array<TElement> operator / (TElement a);
		Array<TElement>& operator += (Array<TElement> &a); // gpu
		Array<TElement>& operator -= (Array<TElement> &a); // gpu
		Array<TElement>& operator *= (Array<TElement> &a); // gpu


		// functions
		Array<TElement> acos();
		Array<TElement> acosd();
		Array<TElement> asin();
		Array<TElement> asind();
		Array<TElement> atan();
		Array<TElement> atand();
		bool			check_close( Array<TElement> a );
		Array<TElement> conj(); // gpu
		Array<TElement> conjugate(); // gpu
		Array<TElement> cos();
		Array<TElement> cosd();
		TElement		determinant(); // gpu
		Array<TElement> detrend(); // gpu + cpu
		Array<TElement> diff();
		Array<TElement> eig( Array<TElement> *eigenvectors ); // gpu + cpu
		Array<TElement> fft(); // gpu
		Array<TElement> getColumn( const index_t col );
		Array<TElement> hermitian(); // gpu
		Array<TElement> invert(); // gpu
		Array<TElement> ls( Array<TElement> *A ); // gpu
		void			lu( Array<TElement> *L, Array<TElement> *U, Array<TElement> *P ); // gpu
		void			lu( Array<TElement> *L, Array<TElement> *U ); // gpu
		Array<TElement>	max(); // gpu
		Array<TElement>	max( Array<TElement> *idx ); // gpu
		Array<TElement>	min(); // gpu
		Array<TElement>	min( Array<TElement> *idx ); // gpu
		Array<TElement> minor(const int row, const int column);
		TElement		norm(); // gpu
		void			print();
		void			qr( Array<TElement> *Q, Array<TElement> *R ); // gpu
		Array<TElement> sin();
		Array<TElement> sind();
		Array<TElement> submatrix( const index_t rowBegin, const index_t rowEnd, const index_t colBegin, const index_t colEnd ); // partial
		Array<TElement> tan();
		Array<TElement> tand();
		Array<TElement> transpose(); // gpu
		void			write2file( std::string s = arrayname2str() );

		// TODO functions
		Array<TElement> addColumn( Array<TElement> *col_to_add ); // TODO
		void			Array2cuSparseCooMatrix( int n, int nnz, int *cooRowIndexHostPtr, int *cooColIndexHostPtr, TElement *cooValHostPtr ); 
		Array<TElement> elementWiseAdd( Array<TElement> *A ); // TODO
		Array<TElement> elementWiseDivide( Array<TElement> *A ); // TODO
		Array<TElement> elementWiseMultiply( Array<TElement> *A ); // TODO
		Array<TElement> operator ^ (TElement a); // TODO
		Array<TElement> roots(); // TODO
		Array<TElement> sqrt(); // TODO

	private:
	
		static std::string	arrayname2str(); 
		ArrayData<TElement>	m_data;
		LinearIndexer		*m_indexer;
		bool				m_padded;	
	};

	class ArrayUtil 
	{
	public:
		static ArrayDescriptor& GetArrayDescriptor(int count, ...);
		static ArrayDescriptor& GetArrayDescriptor(ArrayDescriptor &source);
		static void ReleaseArrayDescriptor(ArrayDescriptor &descriptor);
	};

	class ArrayDescriptor
	{
		friend class ArrayUtil;
		friend class LinearIndexer;
		template <typename TElement> friend class Array;
		template <typename TElement> friend class cublasOperations;
		template <typename TElement> friend class cufftOperations;
		template <typename TElement> friend class curandOperations;
		template <typename TElement> friend class cusolverOperations;

	public:
		ArrayDescriptor(int count);
		ArrayDescriptor(ArrayDescriptor &source);
		ArrayDescriptor(std::vector<index_t> &source);
		ArrayDescriptor(int source[]);
		~ArrayDescriptor();
	
		int GetDim(int dim);
		int GetNDim();
		int GetNumberOfElements();
		bool operator == (const ArrayDescriptor &other) const;
		bool operator != (const ArrayDescriptor &other) const;
	
	private:
		void Swap();
		index_t *m_dim;
		int m_count;
	};

	template <typename TElement, class TAllocator>
	class ArrayData
	{
		template <typename TElement> friend class Array;
		template <typename TElement> friend class cublasOperations;
		template <typename TElement> friend class cufftOperations;
		template <typename TElement> friend class curandOperations;
		template <typename TElement> friend class cusolverOperations;

	public:
		ArrayData(	ArrayDescriptor &descriptor,
					const TElement &defaulTElement = TElement(),
					const TAllocator &allocator = TAllocator());

		ArrayData(	index_t numElements,
					const TAllocator &allocator = TAllocator());

		ArrayData(ArrayData &source);
		~ArrayData();

		const TElement& operator [] (index_t index) const;
		TElement& operator [] (index_t index);
		bool operator == (ArrayData<TElement, TAllocator> &a);
		bool operator != (ArrayData<TElement, TAllocator> &a);
		bool check_close( ArrayData<TElement, TAllocator> a );
		//bool operator > (ArrayData<TElement, TAllocator> &a);
		//bool operator < (ArrayData<TElement, TAllocator> &a);
		//bool operator >= (ArrayData<TElement, TAllocator> &a);
		//bool operator <= (ArrayData<TElement, TAllocator> &a);
		ArrayData<TElement, TAllocator>& operator = (ArrayData<TElement, TAllocator> &a);
		ArrayData<TElement, TAllocator> operator + (ArrayData<TElement, TAllocator> &a);
		ArrayData<TElement, TAllocator> operator - (ArrayData<TElement, TAllocator> &a);
		ArrayData<TElement, TAllocator>& operator += (ArrayData<TElement, TAllocator> &a);
		ArrayData<TElement, TAllocator>& operator -= (ArrayData<TElement, TAllocator> &a);
	
	private:
		void conj();
		void transpose(int width, int height);
		bool AlmostEqual2sComplement(double A, double B);
		bool areEqualRel( TElement a, TElement b );
		bool AlmostEqualRelativeAndAbs(TElement A, TElement B);

		TElement* GetElements() { return m_data; }
	
		TElement *m_data;
		index_t m_numElements;
		TAllocator m_allocator;
		index_t m_size;
	};

	class LinearIndexer
	{
		template <typename TElement> friend class Array;

	public:
		LinearIndexer(ArrayDescriptor &descriptor);
		LinearIndexer(std::vector<index_t> &descriptor);
		~LinearIndexer();

		ArrayDescriptor& GetDescriptor();
		index_t GetIndex() const;
		index_t GetSize() const;
		static index_t GetNeededBufferCapacity(ArrayDescriptor &descriptor);
		static index_t GetNeededBufferCapacity(std::vector<index_t> &descriptor);

	private:
		ArrayDescriptor m_descriptor;
		index_t *m_pos;
	};

	template <typename TElement>
	Array<TElement> eye( index_t N );

	template <typename TElement>
	Array<TElement> dpss( index_t N, double NW, index_t degree );

	Array<ComplexFloat> fft( Array<float> *in );
	Array<ComplexDouble> fft( Array<double> *in );

	template <typename TElement>
	Array<TElement> read_file_vector( std::string s );

	template <typename TElement>
	Array<TElement> read_file_matrix( std::string s );

	template <typename TElement>
	Array<TElement> rand(index_t u);

	template <typename TElement>
	Array<TElement> rand(index_t u1, index_t u2);

	// measure time and print
	void tic();
	void toc();
	long double toc( long double in );

	Array<ComplexFloat> Array_S2C( Array<float> ); // TODO
	Array<ComplexDouble> Array_D2Z( Array<double> ); // TODO
	//Array<TElement> transpose( Array<TElement> ); // TODO
	//Array<TElement> conj( Array<TElement> ); // TODO
	//Array<TElement> hermitian( Array<TElement> ); // TODO
	//Array<TElement> invert( Array<TElement> ); // TODO
	//TElement determinant( Array<TElement> ); // TODO
}
