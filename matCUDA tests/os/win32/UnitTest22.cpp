#include "unitTests.h"

void test_remove_row_int_1()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<int> data( size );
	for( int i = 0; i < data.getDim(0); i++ )
		data(i) = i;

	Array<int> control( size - 1 );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow )
			control( idx++ ) = data( i );

	Array<int> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_int_2()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<int> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<int> control( size - 1, size );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow ) {
			for( int j = 0; j < data.getDim(1); j++ )
				control(idx,j) = data( i, j );
			idx++;
		}

	Array<int> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_float_1()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<float> data( size );
	for( int i = 0; i < data.getDim(0); i++ )
		data(i) = i;

	Array<float> control( size - 1 );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow )
			control( idx++ ) = data( i );

	Array<float> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_float_2()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<float> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<float> control( size - 1, size );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow ) {
			for( int j = 0; j < data.getDim(1); j++ )
				control(idx,j) = data( i, j );
			idx++;
		}

	Array<float> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_double_1()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<double> data( size );
	for( int i = 0; i < data.getDim(0); i++ )
		data(i) = i;

	Array<double> control( size - 1 );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow )
			control( idx++ ) = data( i );

	Array<double> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_double_2()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<double> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<double> control( size - 1, size );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow ) {
			for( int j = 0; j < data.getDim(1); j++ )
				control(idx,j) = data( i, j );
			idx++;
		}

	Array<double> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_complex_1()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<Complex> data( size );
	for( int i = 0; i < data.getDim(0); i++ )
		data(i) = i;

	Array<Complex> control( size - 1 );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow )
			control( idx++ ) = data( i );

	Array<Complex> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_row_complex_2()
{
	size_t size = 6;
	size_t removeRow = 3; // 4th element
	size_t idx = 0;
	
	Array<Complex> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<Complex> control( size - 1, size );
	for( int i = 0; i < data.getDim(0); i++ )
		if( i != removeRow ) {
			for( int j = 0; j < data.getDim(1); j++ )
				control(idx,j) = data( i, j );
			idx++;
		}

	Array<Complex> result = data.removeRow( removeRow );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_int_1()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<int> data( 1, size );
	for( int i = 0; i < data.getDim(1); i++ )
		data( 0, i ) = i;

	Array<int> control( 1, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol )
			control( 0, idx++ ) = data( i );

	Array<int> result = data.removeCol( removeCol );

	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_int_2()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<int> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<int> control( size, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol ) {
			for( int j = 0; j < data.getDim(0); j++ )
				control( j, idx ) = data( j, i );
			idx++;
		}

	Array<int> result = data.removeCol( removeCol );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_float_1()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<float> data( 1, size );
	for( int i = 0; i < data.getDim(1); i++ )
		data( 0, i ) = i;

	Array<float> control( 1, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol )
			control( 0, idx++ ) = data( i );

	Array<float> result = data.removeCol( removeCol );

	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_float_2()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<float> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<float> control( size, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol ) {
			for( int j = 0; j < data.getDim(0); j++ )
				control( j, idx ) = data( j, i );
			idx++;
		}

	Array<float> result = data.removeCol( removeCol );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_double_1()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<double> data( 1, size );
	for( int i = 0; i < data.getDim(1); i++ )
		data( 0, i ) = i;

	Array<double> control( 1, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol )
			control( 0, idx++ ) = data( i );

	Array<double> result = data.removeCol( removeCol );

	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_double_2()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<double> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<double> control( size, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol ) {
			for( int j = 0; j < data.getDim(0); j++ )
				control( j, idx ) = data( j, i );
			idx++;
		}

	Array<double> result = data.removeCol( removeCol );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_complex_1()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<Complex> data( 1, size );
	for( int i = 0; i < data.getDim(1); i++ )
		data( 0, i ) = i;

	Array<Complex> control( 1, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol )
			control( 0, idx++ ) = data( i );

	Array<Complex> result = data.removeCol( removeCol );

	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}

void test_remove_col_complex_2()
{
	size_t size = 6;
	size_t removeCol = 3; // 4th element
	size_t idx = 0;
	
	Array<Complex> data( size, size );
	for( int i = 0; i < data.getDim(0); i++ )
		for( int j = 0; j < data.getDim(1); j++ )
			data(i,j) = j + i*data.getDim(0);

	Array<Complex> control( size, size - 1 );
	for( int i = 0; i < data.getDim(1); i++ )
		if( i != removeCol ) {
			for( int j = 0; j < data.getDim(0); j++ )
				control( j, idx ) = data( j, i );
			idx++;
		}

	Array<Complex> result = data.removeCol( removeCol );
	
	TEST_CALL( result, control, BOOST_CURRENT_FUNCTION );
}