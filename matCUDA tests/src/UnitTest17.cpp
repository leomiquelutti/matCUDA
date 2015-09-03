#include "unitTests.h"

void test_dpss_float_1()
{
	Array<float> test = dpss<float>( 64, 2, 0 );
	//test.print();
}

void test_dpss_float_2(){}

void test_dpss_double_1()
{
	Array<double> test = dpss<double>( 64, 6, 0 );
	test.write2file( "test_dpss.txt" );
	//test.print();
}

void test_dpss_double_2(){}

void test_submatrix_float_1()
{
	index_t size = 1024;
	index_t rowBegin = 9, rowEnd = 203, colBegin = 0, colEnd = 0;
	
	Array<float> v1( size );
	for( int i = 0; i < size; i++ )
		v1(i) = i;
	
	Array<float> control( rowEnd - rowBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ )
		control(i) = i + rowBegin;

	TEST_CALL( control, v1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

void test_submatrix_float_2()
{
	index_t row = 1024, col = 5;
	index_t rowBegin = 450, rowEnd = 479, colBegin = 2, colEnd = 3;
	
	Array<float> m1( row, col );
	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ ) 
			m1( i, j ) = i*j;
	}
	
	Array<float> control( rowEnd - rowBegin + 1, colEnd - colBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ ) {
		for( int j = 0; j <= colEnd - colBegin; j++ ) 
			control( i, j ) = (i + rowBegin)*( j + colBegin );
	}

	TEST_CALL( control, m1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

void test_submatrix_double_1()
{
	index_t size = 1024;
	index_t rowBegin = 9, rowEnd = 203, colBegin = 0, colEnd = 0;
	
	Array<double> v1( size );
	for( int i = 0; i < size; i++ )
		v1(i) = i;
	
	Array<double> control( rowEnd - rowBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ )
		control(i) = i + rowBegin;

	TEST_CALL( control, v1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

void test_submatrix_double_2()
{
	index_t row = 1024, col = 5;
	index_t rowBegin = 450, rowEnd = 479, colBegin = 2, colEnd = 3;
	
	Array<double> m1( row, col );
	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ ) 
			m1( i, j ) = i*j;
	}
	
	Array<double> control( rowEnd - rowBegin + 1, colEnd - colBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ ) {
		for( int j = 0; j <= colEnd - colBegin; j++ ) 
			control( i, j ) = (i + rowBegin)*( j + colBegin );
	}

	TEST_CALL( control, m1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

void test_submatrix_complex_1()
{
	index_t size = 1024;
	index_t rowBegin = 9, rowEnd = 203, colBegin = 0, colEnd = 0;
	
	Array<Complex> v1( size );
	for( int i = 0; i < size; i++ )
		v1(i) = Complex(i,-i);
	
	Array<Complex> control( rowEnd - rowBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ )
		control(i) = Complex( i + rowBegin, -i - rowBegin );

	TEST_CALL( control, v1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

void test_submatrix_complex_2()
{
	index_t row = 1024, col = 5;
	index_t rowBegin = 450, rowEnd = 479, colBegin = 2, colEnd = 3;
	
	Array<Complex> m1( row, col );
	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ ) 
			m1( i, j ) = Complex(i*j,-i*j);
	}
	
	Array<Complex> control( rowEnd - rowBegin + 1, colEnd - colBegin + 1 );
	for( int i = 0; i <= rowEnd - rowBegin; i++ ) {
		for( int j = 0; j <= colEnd - colBegin; j++ ) 
			control( i, j ) = Complex((i + rowBegin)*( j + colBegin ),(i + rowBegin)*( j + colBegin )*(-1));
	}

	TEST_CALL( control, m1.submatrix( rowBegin, rowEnd, colBegin, colEnd ), BOOST_CURRENT_FUNCTION );
}

