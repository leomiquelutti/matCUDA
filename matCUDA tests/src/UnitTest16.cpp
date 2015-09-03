#include "unitTests.h"

void test_diff_float_1()
{
	size_t size = 1024;
	Array<float> a(size), control(size - 1);
	a(0) = 0;
	for( int i = 1; i < size; i++ ) {
		a(i) = i;
		control(i-1) = 1;
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}

void test_diff_float_2()
{
	size_t rows = 2048, cols = 5;
	Array<float> a(rows,cols), control(rows - 1,cols);
	a(0,0) = 0;
	a(0,1) = 0;
	a(0,2) = 0;
	a(0,3) = 0;
	a(0,4) = 0;
	for( int i = 0; i < cols; i++ ) {
		for( int j = 1; j < rows; j++ ) {
			a(j,i) = i*j;
			control(j-1,i) = i;
		}
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}

void test_diff_double_1()
{
	size_t size = 1024;
	Array<double> a(size), control(size - 1);
	a(0) = 0;
	for( int i = 1; i < size; i++ ) {
		a(i) = i;
		control(i-1) = 1;
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}

void test_diff_double_2()
{
	size_t rows = 2048, cols = 5;
	Array<double> a(rows,cols), control(rows - 1,cols);
	a(0,0) = 0;
	a(0,1) = 0;
	a(0,2) = 0;
	a(0,3) = 0;
	a(0,4) = 0;
	for( int i = 0; i < cols; i++ ) {
		for( int j = 1; j < rows; j++ ) {
			a(j,i) = i*j;
			control(j-1,i) = i;
		}
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}

void test_diff_complex_1()
{
	size_t size = 1024;
	Array<Complex> a(size), control(size - 1);
	a(0) = 0;
	for( int i = 1; i < size; i++ ) {
		a(i) = Complex(i,i);
		control(i-1) = Complex(1,1);
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}

void test_diff_complex_2()
{
	size_t rows = 2048, cols = 5;
	Array<Complex> a(rows,cols), control(rows - 1,cols);
	a(0,0) = 0;
	a(0,1) = 0;
	a(0,2) = 0;
	a(0,3) = 0;
	a(0,4) = 0;
	for( int i = 0; i < cols; i++ ) {
		for( int j = 1; j < rows; j++ ) {
			a(j,i) = Complex(i*j,i*j);
			control(j-1,i) = Complex(i,i);
		}
	}

	TEST_CALL( a.diff(), control, BOOST_CURRENT_FUNCTION );
}