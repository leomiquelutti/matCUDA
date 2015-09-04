#include "unitTests.h"

void test_rand_float_1()
{
	index_t size = 10;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	control = 0;

	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_float_2()
{
	index_t size = 10;
	Array<float> v1 = rand<float>( size, size );
	Array<float> v2 = rand<float>( size, size );

	Array<float> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_float_3()
{
	index_t size = 10000;
	Array<float> v1 = rand<float>( size );
	Array<float> v2 = rand<float>( size );

	Array<float> control( size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_float_4()
{
	index_t size = 3e3;
	Array<float> v1 = rand<float>( size, size );
	Array<float> v2 = rand<float>( size, size );

	Array<float> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_double_1()
{
	index_t size = 10;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	control = 0;

	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_double_2()
{
	index_t size = 10;
	Array<double> v1 = rand<double>( size, size );
	Array<double> v2 = rand<double>( size, size );

	Array<double> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_double_3()
{
	index_t size = 10000;
	Array<double> v1 = rand<double>( size );
	Array<double> v2 = rand<double>( size );

	Array<double> control( size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_double_4()
{
	index_t size = 3e3;
	Array<double> v1 = rand<double>( size, size );
	Array<double> v2 = rand<double>( size, size );

	Array<double> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_complex_1()
{
	index_t size = 10;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	control = 0;

	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_complex_2()
{
	index_t size = 10;
	Array<Complex> v1 = rand<Complex>( size, size );
	Array<Complex> v2 = rand<Complex>( size, size );

	Array<Complex> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_complex_3()
{
	index_t size = 10000;
	Array<Complex> v1 = rand<Complex>( size );
	Array<Complex> v2 = rand<Complex>( size );

	Array<Complex> control( size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}

void test_rand_complex_4()
{
	index_t size = 3e3;
	Array<Complex> v1 = rand<Complex>( size, size );
	Array<Complex> v2 = rand<Complex>( size, size );

	Array<Complex> control( size, size );
	control = 0;
	
	if ( v1 - v2 == control )
		BOOST_FAIL("");
	else									          	
		fprintf(stderr, "%s success\n",BOOST_CURRENT_FUNCTION);
}
