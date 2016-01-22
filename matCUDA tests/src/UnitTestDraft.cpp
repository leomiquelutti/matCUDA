#include "unitTests.h"

void draft()
{		
	size_t size = 3;

	Array<ComplexDouble> v1(size);
	v1(0) = ComplexDouble( 1, 0 );
	v1(1) = ComplexDouble( 2, 0 );
	v1(2) = ComplexDouble( 3, 4 );

	Array<ComplexDouble> v2 = v1.abs();

	v2.print();

	Array<ComplexDouble> m1 = rand<ComplexDouble>(size,size);

	m1.print();

	m1.abs().print();

	//Array<ComplexDouble> v1 = rand<ComplexDouble>( size );
	//Array<ComplexDouble> v2 = rand<ComplexDouble>( size );
	//for( int i = 0; i < v1.getDim( 0 ); i++ ) {
	//	v1(i) = i + 1;
	//	v2(i) = i + 1;
	//}

	//Array<ComplexDouble> control2( size );
	//for( int i = 0; i < v1.getDim( 0 ); i++ )
	//	control2( i ) = v1(i)/v2(i);

	//Array<ComplexDouble> control3( size );
	//for( int i = 0; i < v1.getDim( 0 ); i++ )
	//	control3( i ) = v1(i)*v2(i);
	//
	//Array<ComplexDouble> v3 = v1.elementWiseDivide( &v2 );
	//Array<ComplexDouble> v4 = v1.elementWiseMultiply( &v2 );
	//v1.print();
	//v2.print();
	//v3.print();
	//control2.print();
	//v4.print();	
	//control3.print();

	//Array<ComplexDouble> m1 = rand<ComplexDouble>( 2*size, size );
	//Array<ComplexDouble> m2 = rand<ComplexDouble>( 2*size, size );
	//for( int i = 0; i < m1.getDim( 0 ); i++ )
	//	for( int j = 0; j < m1.getDim( 1 ); j++ ) {
	//		m1(i,j) = i + j + 1;
	//		m2(i,j) = i + j + 1;
	//	}

	//Array<ComplexDouble> control( 2*size, size );
	//for( int i = 0; i < m1.getDim( 0 ); i++ )
	//	for( int j = 0; j < m1.getDim( 1 ); j++ )
	//		control( i, j ) = m1(i,j)/m2(i,j);

	//Array<ComplexDouble> m3 = m1.elementWiseDivide( &m2 );
	//Array<ComplexDouble> m4 = m1.elementWiseMultiply( &m2 );
	//m1.print();
	//m2.print();
	//m3.print();
	//control.print();
	//m4.print();

	//TEST_CALL( control2, v1.elementWiseDivide( &v2 ), BOOST_CURRENT_FUNCTION );
	
	
	// here you do your stuff

	//size_t size = 3;

	Array<double> m = rand<double>( size, size );
	m(0,0) = 2;
	m(0,1) = 0;
	m(0,2) = 1;
	m(1,0) = 0;
	m(1,1) = 2;
	m(1,2) = 0;
	m(2,0) = 1;
	m(2,1) = 0;
	m(2,2) = 2;
	m.print();
	
	Array<double> q( size, size );
	Array<double> r( size, size );
	Array<double> a( size, size );
	
	m.qr( &q, &r );
	for( int i = 0; i < 100; i++ ) {
		a = r*q;
		a.qr( &q, &r );
	}
	a = r*q;

	a.print();
	q.print();
	r.print();

	Array<double> eigvec( size, size );
	Array<double> eigval = m.eig( &eigvec );

	eigval.print();
	eigvec.print();
}


//void draft()
//{	
//	// here you do your stuff
//
//	size_t size = 5e3, nIter = 10;
//
//	/////* test 01 - rand *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//
//	/////* test 02 - add *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m3( size, size );
//	//m3 = m2 + m1;	
//
//	/////* test 03 - times *//////
//	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m3( size, size );
//	//m3 = m2*m1;	
//
//	/////* test 04 - lu *//////
//	//Array<ComplexDouble> m = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> l( size, size );
//	//Array<ComplexDouble> u( size, size );
//	//m.lu( &l, &u );	
//
//	/////* test 05 - fft *//////
//	//Array<ComplexDouble> v = rand<ComplexDouble>( pow(2,26) );
//	//Array<ComplexDouble> V( pow(2,26) );
//	// V = v.fft();	
//
//	/////* test 06 - ls *//////
//	//size_t size1 = 1e4, size2 = 1e2, size3 = 1e2;
//	//Array<ComplexDouble> a = rand<ComplexDouble>( size1, size2 );
//	//Array<ComplexDouble> x = rand<ComplexDouble>( size1, size3 );
//	//Array<ComplexDouble> p( size2, size3 );
//	//p = x.ls( &a );
//
//	/////* test 07 - inv *//////
//	//Array<ComplexDouble> m = rand<ComplexDouble>( size, size );
//	//Array<ComplexDouble> m_inv( size, size );
//	//m_inv = m.invert();
//}

//#include "matCUDA.h"
//using namespace matCUDA;
//
//typedef std::complex<double> ComplexDouble;
//
//void example()
//{ 
//	size_t N = 1024;
//
//	// creates complex-vector with N random elements
//	Array<ComplexDouble> v1 = rand<ComplexDouble>( N );
//
//	// creates complex-matrix with N x N elements
//	Array<ComplexDouble> m1( N, N ), m2( N, N );
//
//	// fills in m1 and m2 with random numbers
//	m1 = rand<ComplexDouble>( N, N );
//	
//	// multiplies m1 times v1 and stores result in newly declarated v2 
//	Array<ComplexDouble> v2 = m1*v1; 
//
//	// compares if v1 and v2 are equal element-wise and returns true of false 
//	bool trueOrFalse = v1 == v2; 
//	
//	// multiplies m1 times hermitian of m1 and stores the result in m2
//	m2 = m1*m1.hermitian();
//
//	// adds v1 to vector m2*v2 and stores in v1
//	// equivalent to v1 = v1+ m2*v2
//	v1 += m2*v2;
//
//	// multiplies m2 times the inverse of m1 and stores the result in m1 
//	// equivalente to m1 = m2*m1.inverse();
//	m1 = m2/m1;
//     
//	// multiplies all elements of m1 times 4, adds to m2 and then
//	// subtracts 5 of all elements and stores the result in m2
//	m2 = m1*4 + m2 - 5; 
//	
//	// evaluates m2 QR decomposition
//	Array<ComplexDouble> q( N, N );
//	Array<ComplexDouble> r( N, N );
//	m2.qr( &q, &r );
//
//	// solves linear problems represented by Ap = x by least square method 
//	// in this example, A -> m1; p -> v1; x -> v2
//	v1 = v2.ls( &m1 );
//
//	// print v1 on screen
//	v1.print();
//}