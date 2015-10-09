#include "unitTests.h"

#include "matCUDA.h"
using namespace matCUDA;

void draft()
{	
	// here you do your stuff

	size_t size = 5e3, nIter = 1;

	/* test01 - rand */
	//Array<ComplexDouble> m1( size, size );
	////tic();
	//for( int i = 0; i < nIter; i++ )
	//	m1 = rand<ComplexDouble>( size, size );
	////toc();

	/* test02 - add */
	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
	//Array<ComplexDouble> m3( size, size );
	////tic();
	//for( int i = 0; i < nIter; i++ )
	//	m3 = m1 + m2;
	////toc();

	/* test03 - times */
	//Array<ComplexDouble> m1 = rand<ComplexDouble>( size, size );
	//Array<ComplexDouble> m2 = rand<ComplexDouble>( size, size );
	//Array<ComplexDouble> m3( size, size );
	////tic();
	//for( int i = 0; i < nIter; i++ )
	//	m3 = m1*m2;
	////toc();

	/* test04 - lu */
	Array<ComplexDouble> a = rand<ComplexDouble>( size, size );
	Array<ComplexDouble> l( size, size );
	Array<ComplexDouble> u( size, size );
	tic();
	for( int i = 0; i < nIter; i++ )
		a.LU( &l, &u );
	toc();	

	/* test05 - fft */
	//Array<ComplexDouble> v = rand<ComplexDouble>( size );
	//Array<ComplexDouble> V( size );
	////tic();
	//for( int i = 0; i < nIter; i++ )
	//	V = v.fft();
	////toc();
	
	/* test06 - ls */
	//size_t size1 = 1e4, size2 = 1e2, size3 = 1e2, nIter = 1;
	//Array<ComplexDouble> a = rand<ComplexDouble>( size1, size2 );
	//Array<ComplexDouble> x = rand<ComplexDouble>( size1, size3 );
	//Array<ComplexDouble> p( size2, size3 );
	//tic();
	//for( int i = 0; i < nIter; i++ )
	//	p = x.LS( &a );	
	//toc();
	
	//Array<double> a = rand<double>( size, size );
	//Array<double> l( size, size );
	//Array<double> u( size, size );
	//tic();
	//for( int i = 0; i < nIter; i++ )
	//	a.LU( &l, &u );
	//toc();	

	//Array<double> a( 2, 2 );	
	//Array<double> l( 2, 2 );
	//Array<double> u( 2, 2 );
	//Array<double> p( 2, 2 );

	//a(0,0) = 4;
	//a(0,1) = 3;
	//a(1,0) = 6;
	//a(1,1) = 3;

	//a.print();

	//a.LU( &l, &u, &p );

	//a.print();
	//l.print();
	//u.print();
	//p.print();
	//
	//(l*u).print();
	//(p*a).print();
}