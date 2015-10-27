#include "unitTests.h"

void test_QR_float_1()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_1//data.txt") );
	Array<float> Q = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_1//q.txt") );
	Array<float> R = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_1//r.txt") );

	Array<float> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<float> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_float_2()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_2//data.txt") );
	Array<float> Q = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_2//q.txt") );
	Array<float> R = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_2//r.txt") );

	Array<float> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<float> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );
	
	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_float_3()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_3//data.txt") );
	Array<float> Q = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_3//q.txt") );
	Array<float> R = read_file_matrix<float>( std::string("unit_tests_files//test_QR_float_3//r.txt") );

	Array<float> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<float> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );
	
	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_double_1()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_1//data.txt") );
	Array<double> Q = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_1//q.txt") );
	Array<double> R = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_1//r.txt") );

	Array<double> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<double> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_double_2()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_2//data.txt") );
	Array<double> Q = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_2//q.txt") );
	Array<double> R = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_2//r.txt") );

	Array<double> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<double> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_double_3()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_3//data.txt") );
	Array<double> Q = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_3//q.txt") );
	Array<double> R = read_file_matrix<double>( std::string("unit_tests_files//test_QR_double_3//r.txt") );

	Array<double> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<double> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_complex_1()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_1//data.txt") );
	Array<Complex> Q = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_1//q.txt") );
	Array<Complex> R = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_1//r.txt") );

	Array<Complex> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<Complex> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_complex_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_2//data.txt") );
	Array<Complex> Q = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_2//q.txt") );
	Array<Complex> R = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_2//r.txt") );

	Array<Complex> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<Complex> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}

void test_QR_complex_3()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_3//data.txt") );
	Array<Complex> Q = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_3//q.txt") );
	Array<Complex> R = read_file_matrix<Complex>( std::string("unit_tests_files//test_QR_complex_3//r.txt") );

	Array<Complex> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	Array<Complex> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );

	data.qr( &q, &r );

	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
}