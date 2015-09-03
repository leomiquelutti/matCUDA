#include "unitTests.h"

void test_max_float_1()
{
	index_t size = 7;
	Array<float> v(size);
	Array<float> idx(1,1);
	Array<float> control_v(1);
	Array<float> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(float)size/2;

	control_idx(0,0) = 0;

	Array<float> test = v.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_max_float_2()
{
	index_t row = 10;
	index_t col = 3;
	Array<float> m(row,col);
	Array<float> idx(1,col);
	Array<float> control_m(1,col);
	Array<float> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(float)row/2)*(j+1)*pow(-1,j);
	}

	control_idx(0,0) = row-1;
	control_idx(0,1) = 0;
	control_idx(0,2) = row-1;

	Array<float> test = m.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_max_float_3()
{
	index_t size = 7;
	Array<float> v(size);
	Array<float> idx(1,1);
	Array<float> control_v(1);
	Array<float> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(float)size/2;

	control_v(0) = v(0);
	
	TEST_CALL( control_v, v.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_float_4()
{
	index_t row = 10;
	index_t col = 3;
	Array<float> m(row,col);
	Array<float> idx(1,col);
	Array<float> control_m(1,col);
	Array<float> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(float)row/2)*(j+1)*pow(-1,j);
	}

	control_m(0,0) = m(row-1,0);
	control_m(0,1) = m(0,1);
	control_m(0,2) = m(row-1,2);

	TEST_CALL( control_m, m.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_float_5()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_max_float_5//data.txt") );
	Array<float> control_data = read_file_matrix<float>( std::string("unit_tests_files//test_max_float_5//control_data.txt") );

	TEST_CALL( control_data, data.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_float_6()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_max_float_6//data.txt") );
	Array<float> control_idx = read_file_matrix<float>( std::string("unit_tests_files//test_max_float_6//control_idx.txt") );

	Array<float> idx( 1, data.GetDescriptor().GetDim( 1 ) );
	Array<float> test = data.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_max_double_1()
{
	index_t size = 7;
	Array<double> v(size);
	Array<double> idx(1,1);
	Array<double> control_v(1);
	Array<double> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(double)size/2;

	control_idx(0,0) = 0;

	Array<double> test = v.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_max_double_2()
{
	index_t row = 10;
	index_t col = 3;
	Array<double> m(row,col);
	Array<double> idx(1,col);
	Array<double> control_m(1,col);
	Array<double> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(double)row/2)*(j+1)*pow(-1,j);
	}

	control_idx(0,0) = row-1;
	control_idx(0,1) = 0;
	control_idx(0,2) = row-1;

	Array<double> test = m.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_max_double_3()
{
	index_t size = 7;
	Array<double> v(size);
	Array<double> idx(1,1);
	Array<double> control_v(1);
	Array<double> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(double)size/2;

	control_v(0) = v(0);
	
	TEST_CALL( control_v, v.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_double_4()
{
	index_t row = 10;
	index_t col = 3;
	Array<double> m(row,col);
	Array<double> idx(1,col);
	Array<double> control_m(1,col);
	Array<double> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(double)row/2)*(j+1)*pow(-1,j);
	}

	control_m(0,0) = m(row-1,0);
	control_m(0,1) = m(0,1);
	control_m(0,2) = m(row-1,2);

	TEST_CALL( control_m, m.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_double_5()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_max_double_5//data.txt") );
	Array<double> control_data = read_file_matrix<double>( std::string("unit_tests_files//test_max_double_5//control_data.txt") );

	TEST_CALL( control_data, data.max(), BOOST_CURRENT_FUNCTION );
}

void test_max_double_6()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_max_double_6//data.txt") );
	Array<double> control_idx = read_file_matrix<double>( std::string("unit_tests_files//test_max_double_6//control_idx.txt") );

	Array<double> idx( 1, data.GetDescriptor().GetDim( 1 ) );
	Array<double> test = data.max( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_float_1()
{
	index_t size = 7;
	Array<float> v(size);
	Array<float> idx(1,1);
	Array<float> control_v(1);
	Array<float> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(float)size/2;

	control_idx(0,0) = size-1;

	Array<float> test = v.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_float_2()
{
	index_t row = 10;
	index_t col = 3;
	Array<float> m(row,col);
	Array<float> idx(1,col);
	Array<float> control_m(1,col);
	Array<float> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(float)row/2)*(j+1)*pow(-1,j);
	}

	control_idx(0,0) = 0;
	control_idx(0,1) = row-1;
	control_idx(0,2) = 0;

	Array<float> test = m.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_float_3()
{
	index_t size = 7;
	Array<float> v(size);
	Array<float> idx(1,1);
	Array<float> control_v(1);
	Array<float> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(float)size/2;

	control_v(0) = v(size-1);
	
	TEST_CALL( control_v, v.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_float_4()
{
	index_t row = 10;
	index_t col = 3;
	Array<float> m(row,col);
	Array<float> idx(1,col);
	Array<float> control_m(1,col);
	Array<float> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(float)row/2)*(j+1)*pow(-1,j);
	}

	control_m(0,0) = m(0,0);
	control_m(0,1) = m(row-1,1);
	control_m(0,2) = m(0,2);

	TEST_CALL( control_m, m.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_float_5()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_min_float_5//data.txt") );
	Array<float> control_data = read_file_matrix<float>( std::string("unit_tests_files//test_min_float_5//control_data.txt") );

	TEST_CALL( control_data, data.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_float_6()
{
	Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//test_min_float_6//data.txt") );
	Array<float> control_idx = read_file_matrix<float>( std::string("unit_tests_files//test_min_float_6//control_idx.txt") );

	Array<float> idx( 1, data.GetDescriptor().GetDim( 1 ) );
	Array<float> test = data.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_double_1()
{
	index_t size = 7;
	Array<double> v(size);
	Array<double> idx(1,1);
	Array<double> control_v(1);
	Array<double> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(double)size/2;

	control_idx(0,0) = size-1;

	Array<double> test = v.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_double_2()
{
	index_t row = 10;
	index_t col = 3;
	Array<double> m(row,col);
	Array<double> idx(1,col);
	Array<double> control_m(1,col);
	Array<double> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(double)row/2)*(j+1)*pow(-1,j);
	}

	control_idx(0,0) = 0;
	control_idx(0,1) = row-1;
	control_idx(0,2) = 0;

	Array<double> test = m.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}

void test_min_double_3()
{
	index_t size = 7;
	Array<double> v(size);
	Array<double> idx(1,1);
	Array<double> control_v(1);
	Array<double> control_idx(1,1);

	for( int i = 0; i < size; i++ )
		v(i) = -i+(double)size/2;

	control_v(0) = v(size-1);
	
	TEST_CALL( control_v, v.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_double_4()
{
	index_t row = 10;
	index_t col = 3;
	Array<double> m(row,col);
	Array<double> idx(1,col);
	Array<double> control_m(1,col);
	Array<double> control_idx(1,col);

	for( int i = 0; i < row; i++ ) {
		for( int j = 0; j < col; j++ )
			m(i,j) = (i-(double)row/2)*(j+1)*pow(-1,j);
	}

	control_m(0,0) = m(0,0);
	control_m(0,1) = m(row-1,1);
	control_m(0,2) = m(0,2);

	TEST_CALL( control_m, m.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_double_5()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_min_double_5//data.txt") );
	Array<double> control_data = read_file_matrix<double>( std::string("unit_tests_files//test_min_double_5//control_data.txt") );

	TEST_CALL( control_data, data.min(), BOOST_CURRENT_FUNCTION );
}

void test_min_double_6()
{
	Array<double> data = read_file_matrix<double>( std::string("unit_tests_files//test_min_double_6//data.txt") );
	Array<double> control_idx = read_file_matrix<double>( std::string("unit_tests_files//test_min_double_6//control_idx.txt") );

	Array<double> idx( 1, data.GetDescriptor().GetDim( 1 ) );
	Array<double> test = data.min( &idx );

	TEST_CALL( control_idx, idx, BOOST_CURRENT_FUNCTION );
}