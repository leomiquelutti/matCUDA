#include "unitTests.h"

void test_detrend_float_1()
{
	Array<float> data = read_file_vector<float>( std::string("unit_tests_files//test_detrend_float_1//data.txt") );
	Array<float> control = read_file_vector<float>( std::string("unit_tests_files//test_detrend_float_1//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_float_2()
{
	Array<float> data = read_file_vector<float>( std::string("unit_tests_files//test_detrend_float_2//data.txt") );
	Array<float> control = read_file_vector<float>( std::string("unit_tests_files//test_detrend_float_2//control.txt") );

	Array<float> data2 = data.detrend();
	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_float_3()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_detrend_float_3//data.txt") );
	Array<float> control = read_file_matrix<float>( std::string("unit_tests_files//test_detrend_float_3//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_float_4()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_detrend_float_4//data.txt") );
	Array<float> control = read_file_matrix<float>( std::string("unit_tests_files//test_detrend_float_4//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_double_1()
{
	Array<double> data = read_file_vector<double>( std::string("unit_tests_files//test_detrend_double_1//data.txt") );
	Array<double> control = read_file_vector<double>( std::string("unit_tests_files//test_detrend_double_1//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_double_2()
{
	Array<double> data = read_file_vector<double>( std::string("unit_tests_files//test_detrend_double_2//data.txt") );
	Array<double> control = read_file_vector<double>( std::string("unit_tests_files//test_detrend_double_2//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_double_3()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_detrend_double_3//data.txt") );
	Array<double> control = read_file_matrix<double>( std::string("unit_tests_files//test_detrend_double_3//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}

void test_detrend_double_4()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_detrend_double_4//data.txt") );
	Array<double> control = read_file_matrix<double>( std::string("unit_tests_files//test_detrend_double_4//control.txt") );

	TEST_CALL( data.detrend(), control, BOOST_CURRENT_FUNCTION );
}