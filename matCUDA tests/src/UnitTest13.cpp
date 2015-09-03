#include "unitTests.h"
//#include "global_func.h"

void test_FFT_C2C_1()
{
	Array<Complex> data = read_file_vector<Complex>( std::string("unit_tests_files//test_FFT_C2C_1//data.txt") );
	Array<Complex> control = read_file_vector<Complex>( std::string("unit_tests_files//test_FFT_C2C_1//control.txt") );

	TEST_CALL( control, data.fft(), BOOST_CURRENT_FUNCTION );
}

void test_FFT_C2C_2()
{
	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//test_FFT_C2C_2//data.txt") );
	Array<Complex> control = read_file_matrix<Complex>( std::string("unit_tests_files//test_FFT_C2C_2//control.txt") );

	TEST_CALL( control, data.fft(), BOOST_CURRENT_FUNCTION );
}

void test_FFT_C2C_3()
{
	Array<Complex> data = read_file_vector<Complex>( std::string("unit_tests_files//test_FFT_C2C_3//data.txt") );
	Array<Complex> control = read_file_vector<Complex>( std::string("unit_tests_files//test_FFT_C2C_3//control.txt") );
	
	TEST_CALL( control, data.fft(), BOOST_CURRENT_FUNCTION );
}

void test_FFT_R2C_1()
{
	Array<float> data = read_file_vector<float>( std::string("unit_tests_files//test_FFT_R2C_1//data.txt") );
	Array<ComplexFloat> control = read_file_vector<ComplexFloat>( std::string("unit_tests_files//test_FFT_R2C_1//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}

void test_FFT_R2C_2()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_FFT_R2C_2//data.txt") );
	Array<ComplexFloat> control = read_file_matrix<ComplexFloat>( std::string("unit_tests_files//test_FFT_R2C_2//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}

void test_FFT_R2C_3()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_FFT_R2C_3//data.txt") );
	Array<ComplexFloat> control = read_file_matrix<ComplexFloat>( std::string("unit_tests_files//test_FFT_R2C_3//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}

void test_FFT_D2Z_1()
{
	Array<double> data = read_file_vector<double>( std::string("unit_tests_files//test_FFT_D2Z_1//data.txt") );
	Array<ComplexDouble> control = read_file_vector<ComplexDouble>( std::string("unit_tests_files//test_FFT_D2Z_1//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}

void test_FFT_D2Z_2()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_FFT_D2Z_2//data.txt") );
	Array<ComplexDouble> control = read_file_matrix<ComplexDouble>( std::string("unit_tests_files//test_FFT_D2Z_2//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}

void test_FFT_D2Z_3()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_FFT_D2Z_3//data.txt") );
	Array<ComplexDouble> control = read_file_matrix<ComplexDouble>( std::string("unit_tests_files//test_FFT_D2Z_3//control.txt") );

	TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
}
