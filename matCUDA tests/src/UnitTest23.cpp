#include "unitTests.h"

void test_abs_complex_1() 
{
	size_t size = 6;
	
	Array<Complex> data = rand<Complex>( size );
	Array<Complex> control( size );

	for( int i = 0; i < data.getDim(0); i++ )
		control(i) = sqrt( data(i).real()*data(i).real() + data(i).imag()*data(i).imag());
	
	Array<Complex> result = data.abs();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs_complex_2()
{
	size_t size = 6;
	
	Array<Complex> data = rand<Complex>( size, 2*size );
	Array<Complex> control( size, 2*size );

	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			control(i,j) = sqrt( data(i,j).real()*data(i,j).real() + data(i,j).imag()*data(i,j).imag());
	
	Array<Complex> result = data.abs();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs_complex_3() 
{
	size_t size = 4096;
	
	Array<Complex> data = rand<Complex>( size );
	Array<Complex> control( size );

	for( int i = 0; i < data.getDim(0); i++ )
		control(i) = sqrt( data(i).real()*data(i).real() + data(i).imag()*data(i).imag());
	
	Array<Complex> result = data.abs();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs_complex_4()
{
	size_t size = 3000;
	
	Array<Complex> data = rand<Complex>( size, 2*size );
	Array<Complex> control( size, 2*size );

	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			control(i,j) = sqrt( data(i,j).real()*data(i,j).real() + data(i,j).imag()*data(i,j).imag());
	
	Array<Complex> result = data.abs();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs2_complex_1() 
{
	size_t size = 6;
	
	Array<Complex> data = rand<Complex>( size );
	Array<Complex> control( size );

	for( int i = 0; i < data.getDim(0); i++ )
		control(i) = ( data(i).real()*data(i).real() + data(i).imag()*data(i).imag());
	
	Array<Complex> result = data.abs2();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs2_complex_2()
{
	size_t size = 6;
	
	Array<Complex> data = rand<Complex>( size, 2*size );
	Array<Complex> control( size, 2*size );

	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			control(i,j) = ( data(i,j).real()*data(i,j).real() + data(i,j).imag()*data(i,j).imag());
	
	Array<Complex> result = data.abs2();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs2_complex_3() 
{
	size_t size = 4096;
	
	Array<Complex> data = rand<Complex>( size );
	Array<Complex> control( size );

	for( int i = 0; i < data.getDim(0); i++ )
		control(i) = ( data(i).real()*data(i).real() + data(i).imag()*data(i).imag());
	
	Array<Complex> result = data.abs2();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_abs2_complex_4()
{
	size_t size = 3000;
	
	Array<Complex> data = rand<Complex>( size, 2*size );
	Array<Complex> control( size, 2*size );

	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			control(i,j) = ( data(i,j).real()*data(i,j).real() + data(i,j).imag()*data(i,j).imag());
	
	Array<Complex> result = data.abs2();
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}