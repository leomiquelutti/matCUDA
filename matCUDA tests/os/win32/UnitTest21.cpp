#include "unitTests.h"

void test_elementwise_multiplication_float_1()
{
	size_t size = 6;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_float_2()
{
	size_t size = 1e3;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_float_3()
{
	size_t size = 6;
	Array<float> m1 = rand<float>( 2*size, size );
	Array<float> m2 = rand<float>( 2*size, size );

	Array<float> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_float_4()
{
	size_t size = 1e3;
	Array<float> m1 = rand<float>( 2*size, size );
	Array<float> m2 = rand<float>( 2*size, size );

	Array<float> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_double_1()
{
	size_t size = 6;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_double_2()
{
	size_t size = 1e3;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_double_3()
{
	size_t size = 6;
	Array<double> m1 = rand<double>( 2*size, size );
	Array<double> m2 = rand<double>( 2*size, size );

	Array<double> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_double_4()
{
	size_t size = 1e3;
	Array<double> m1 = rand<double>( 2*size, size );
	Array<double> m2 = rand<double>( 2*size, size );

	Array<double> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_complex_1()
{
	size_t size = 6;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_complex_2()
{
	size_t size = 1e3;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)*v2(i);
	
	TEST_CALL( control, v1.elementWiseMultiply( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_complex_3()
{
	size_t size = 6;
	Array<Complex> m1 = rand<Complex>( 2*size, size );
	Array<Complex> m2 = rand<Complex>( 2*size, size );

	Array<Complex> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_multiplication_complex_4()
{
	size_t size = 1e3;
	Array<Complex> m1 = rand<Complex>( 2*size, size );
	Array<Complex> m2 = rand<Complex>( 2*size, size );

	Array<Complex> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)*m2(i,j);

	TEST_CALL( control, m1.elementWiseMultiply( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_float_1()
{
	size_t size = 6;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_float_2()
{
	size_t size = 1e3;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_float_3()
{
	size_t size = 6;
	Array<float> m1 = rand<float>( 2*size, size );
	Array<float> m2 = rand<float>( 2*size, size );

	Array<float> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_float_4()
{
	size_t size = 1e3;
	Array<float> m1 = rand<float>( 2*size, size );
	Array<float> m2 = rand<float>( 2*size, size );

	Array<float> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_double_1()
{
	size_t size = 6;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_double_2()
{
	size_t size = 1e3;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_double_3()
{
	size_t size = 6;
	Array<double> m1 = rand<double>( 2*size, size );
	Array<double> m2 = rand<double>( 2*size, size );

	Array<double> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_double_4()
{
	size_t size = 1e3;
	Array<double> m1 = rand<double>( 2*size, size );
	Array<double> m2 = rand<double>( 2*size, size );

	Array<double> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_complex_1()
{
	size_t size = 6;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_complex_2()
{
	size_t size = 1e3;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	for( int i = 0; i < v1.getDim( 0 ); i++ )
		control( i ) = v1(i)/v2(i);
	
	TEST_CALL( control, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_complex_3()
{
	size_t size = 6;
	Array<Complex> m1 = rand<Complex>( 2*size, size );
	Array<Complex> m2 = rand<Complex>( 2*size, size );

	Array<Complex> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}

void test_elementwise_division_complex_4()
{
	size_t size = 1e3;
	Array<Complex> m1 = rand<Complex>( 2*size, size );
	Array<Complex> m2 = rand<Complex>( 2*size, size );

	Array<Complex> control( 2*size, size );
	for( int i = 0; i < m1.getDim( 0 ); i++ )
		for( int j = 0; j < m1.getDim( 1 ); j++ )
			control( i, j ) = m1(i,j)/m2(i,j);

	TEST_CALL( control, m1.elementWiseDivide( &m2 ), BOOST_CURRENT_FUNCTION );
}