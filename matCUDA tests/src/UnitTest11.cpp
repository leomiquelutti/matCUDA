#include "unitTests.h"

// float type
void test_LU_float_1()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_LU_float_1//data.txt") );
	Array<float> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<float> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<float> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );

	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}

void test_LU_float_2()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_LU_float_2//data.txt") );
	Array<float> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<float> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<float> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );

	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}

// double type
void test_LU_double_1()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_LU_double_1//data.txt") );
	Array<double> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<double> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<double> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );
	//data.print();
	//l.print();
	//u.print();
	//p.print();

	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}

void test_LU_double_2()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_LU_double_2//data.txt") );
	Array<double> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<double> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<double> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );

	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}

// complex type
void test_LU_complex_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_LU_complex_1//data.txt") );
	Array<Complex> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<Complex> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<Complex> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );
	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}

void test_LU_complex_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_LU_complex_2//data.txt") );
	Array<Complex> p( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	Array<Complex> l( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );
	Array<Complex> u( p.GetDescriptor().GetDim(0), p.GetDescriptor().GetDim(1) );

	data.LU( &l, &u, &p );

	TEST_CALL( l*u, p*data, BOOST_CURRENT_FUNCTION );
}