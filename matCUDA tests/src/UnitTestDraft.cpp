#include "unitTests.h"

void test_alloc_Array( Array<float> *a )
{
	a = new Array<float>(2);
	a->print();
	(*a)(0) = 1;
	a->print();
}

void draft()
{
	// item 2
	index_t sizeSmall = 4;
	Array<ComplexFloat> m1( sizeSmall, sizeSmall );
	std::cout << m1.getNDim() << std::endl;
	std::cout << m1.getDim(0) << "\t" << m1.getDim(1) << std::endl;
}

//void draft()
//{
//	// item 2
//	index_t sizeSmall = 4;
//	Array<float> m1( sizeSmall, sizeSmall );
//	for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
//		for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
//			m1(i,j) = -1.0;
//	}
//
//	Array<float> m2( sizeSmall, sizeSmall );
//	for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
//		for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
//			m2(i,j) = 2.0;
//	}
//
//	Array<float> m3( sizeSmall, sizeSmall );
//
//	(m2 + m1).print();
//	
//	Array<float> m4( sizeSmall, sizeSmall );
//	Array<float> m5( sizeSmall, sizeSmall );
//	m3.print();	
//	m4.print();
//	m5.print();
//
//	// item 5
//	Array<float> *a = NULL;
//	test_alloc_Array( a );
//	a->print();
//}


	//Array<float> data = read_file_vector<float>( std::string("unit_tests_files//draft//data.txt") );
	//Array<ComplexFloat> control = read_file_vector<ComplexFloat>( std::string("unit_tests_files//draft//control.txt") );
	//Array<ComplexFloat> cccc = fft( &data );
	//cccc.print();
	//control.print();
	//TEST_CALL( control, fft(&data), BOOST_CURRENT_FUNCTION );
	//int sizeSmall = 4;
	//Array<float> m1(sizeSmall, 2*sizeSmall);
	//for (int i = 0; i < m1.GetDescriptor().GetDim( 0 ); i++) {
	//	for (int j = 0; j < m1.GetDescriptor().GetDim( 1 ); j++)
	//		m1(i,j) = float(i*m1.GetDescriptor().GetDim( 1 ) + j + 1)/10;
	//}
	//
	//Array<float> m2(2*sizeSmall, sizeSmall);
	//for (int i = 0; i < m2.GetDescriptor().GetDim( 0 ); i++) {
	//	for (int j = 0; j < m2.GetDescriptor().GetDim( 1 ); j++)
	//		m2(i,j) = float(j*m2.GetDescriptor().GetDim( 0 ) + i + 1)/10;
	//}
	//
	//Array<float> m3 = m1.transpose();
	//m3.print();
	//
	//Array<float> data = read_file_matrix<float>( std::string("unit_tests_files//draft//data.txt") );
	//Array<float> Q = read_file_matrix<float>( std::string("unit_tests_files//draft//q.txt") );
	//Array<float> R = read_file_matrix<float>( std::string("unit_tests_files//draft//r.txt") );
	//
	//Array<float> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
	//Array<float> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );
	//
	//data.QR( &q, &r );
	//
	//Q.print();
	//q.print();
	//
	//R.print();
	//r.print();
	//
	//TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );

//{
//	Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//draft//data.txt") );
//	Array<Complex> Q = read_file_matrix<Complex>( std::string("unit_tests_files//draft//q.txt") );
//	Array<Complex> R = read_file_matrix<Complex>( std::string("unit_tests_files//draft//r.txt") );
//
//	data.print();
//
//	Array<Complex> q( Q.GetDescriptor().GetDim(0), Q.GetDescriptor().GetDim(1) );
//	Array<Complex> r( R.GetDescriptor().GetDim(0), R.GetDescriptor().GetDim(1) );
//
//	data.QR( &q, &r );
//
//	Q.print();
//	q.print();
//
//	R.print();
//	r.print();
//
//	TEST_CALL( q*r, data, BOOST_CURRENT_FUNCTION );
//}

////// draft!!
//#include "matCUDA.h"
//
//using namespace matCUDA;
//
//void example()
//{
//	size_t N = 1024;
//
//	// creates double-type column-vector with N elements
//	Array<double> v1( N );
//
//	// creates double-type row-vector with N elements
//	Array<double> v2( 1, N );
//
//	// fills in v1 and v2
//	// {...}
//
//	// creates double-type matrix with NxN random elements
//	Array<double> m1 = rand<double>( N, N );
//
//	// creates double-type matrix with the NxN
//	// elements resulting from v1 times v2
//	Array<double> m2 = v1*v2;
//
//	// adds m1 to m2 and stores in m2
//	m2 = m1 + m2;
//
//	// multplies m1 times v1 and stores in v1
//	v1 = m1*v1;
//
//	// transpose v2 and stores in v1
//	v1 = v2.transpose();
//
//	// creates complex-type matrix with the results from a
//	// fast fourier transform on each column of m2
//	Array<ComplexDouble> m3 = fft( &m2 );
//
//	// perform the LU decoomposition of m2
//	// in a way that l*u = p*m2
//	Array<double> p( N, N );
//	Array<double> l( N, N );
//	Array<double> u( N, N );
//	m2.LU( &l, &u, &p );
//	
//	// evaluates the sine of m1 elements (considered in degrees)
//	m2 = m1.sind();
//	
//	// evaluates m3 QR decomposition
//	Array<ComplexDouble> q( N, N );
//	Array<ComplexDouble> r( N, N );
//	m3.QR( &q, &r );
//	
//	// solves linear problems represented by Ap = x by least square method 
//	// in this example, A -> m1; p -> v1; x -> v2
//	v1 = v2.LS( m1 );
//	
//	// print v1 on screen
//	v1.print();
//}

// Exemplo 3
//void draft()
//{
//	size_t N = 1024;
//
//	// criar vetores tipo Complex com N posições
//	Array<Complex> v1( N ), v2( N );
//
//	// criar matrizes tipo Complex com NxN posições
//	Array<Complex> m1( N, N ), m2( N, N );
//
//	// preenche v1, v2 e m1 com valores quaisquer
//	{...}
//
//	// calcula a norma do vetor e retorna variável (ao invés de Array)
//	Complex norm = v1.norm();
//
//	// calcula o determinante da matriz e retorna variável (ao invés de Array)
//	Complex det = m1.determinant();
//
//	// calcula a hermitiana de v1 e armazena em v3
//	// v3 é, portanto, um vetor linha de dimensões v3( 1, N );
//	Array<Complex> v3 = v1.hermitian();
//
//	// calcula a transposta de v1 e armazena em v3
//	v3 = v1.transpose();
//
//	// calcula o complexo conjugado de v1 e armazena em v2
//	v2 = v1.conj();
//
//	// calcula a matriz gerada pela multiplicação de 1 vetor coluna com 1 vetor linha
//	m2 = v1*v2.transpose();
//
//	// calcula a inversa da matriz m1 e armazena em m2
//	m2 = m1.invert();
//
//	// calcula o seno dos elementos (em radianos) da matriz m1
//	m2 = m1.sin();
//
//	// calcula o seno dos elementos (em graus) da matriz m1
//	m2 = m1.sind();
//
//	// calcula o arco-cosseno (em radianos) dos elementos da matriz m1
//	m2 = m1.acos();
//	
//	// calcula o arco-cosseno (em graus) dos elementos da matriz m1
//	m2 = m1.acosd();
//
//	// calcula a decomposição LU da matriz m2
//	// de forma que l*u = p*m2
//	Array<Complex> l( N, N );
//	Array<Complex> u( N, N );
//	Array<Complex> p( N, N );
//	m2.LU( &l, &u, &p ); 
//
//	// calcula a decomposição QR da matriz m2
//	Array<Complex> q( N, N );
//	Array<Complex> r( N, N );
//	m2.QR( &q, &r );
//
//	// remove a tendência linear de v1 e armazena em v2
//	v2 = v1.detrend();
//
//	// calcula os autovalores da matriz m1 e armazena em v1
//	// e armazena os autovetores em m2
//	v1 = m1.eig( &m2 );
//
//	// calcula a fft do vetor v1 e armazena em v2
//	v2 = v1.fft();
//
//	// calcula a fft das colunas da matriz m1
//	// e armazena nas colunas de m2
//	m2 = m1.fft();
//
//	// exclui linha 7 e coluna 200 escolhidas da matriz m1
//	// e armazena em m4( N-1, N-1 );
//	Array<Complex> m4 = m2.minor( 7, 200 );
//
//	// extrai coluna 3 de m2 e armazena em v1
//	v1 = m2.get_column( 3 );
//
//	// calcula primeira derivada do vetor v1
//	// e armazena em v4( N - 1 )
//	Array<Complex> v4 = v1.diff();
//
//	// resolve por métodos dos mínimos quadrados
//	// problemas lineares tipo AX = B
//	// A -> m1; X -> v1; B -> v2
//	v1 = v2.LS( m1 );
//
//	// imprime v1 na tela
//	v1.print();
//}

////// draft!!
//example_01()
//{
//	size_t N = 1024;
//
//	// criar vetor tipo float com N posições
//	Array<float> v1(1024);
//
//	// criar matriz tipo double com NxN posições
//	Array<double> m1(1024);
//
//	// preenche v1
//	for( int i = 0; i < N; i++ )
//		v1( i ) = i;
//
//	// cria v2 com o mesmo tipo, dimensões e valores de v1
//	Array<float> v2 = v1;
//}

	//Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//draft//data.txt") );
	//Array<Complex> L = read_file_matrix<Complex>( std::string("unit_tests_files//draft//l.txt") );
	//Array<Complex> U = read_file_matrix<Complex>( std::string("unit_tests_files//draft//u.txt") );
	//Array<Complex> P( data.GetDescriptor().GetDim(0), data.GetDescriptor().GetDim(1) );

	//Array<Complex> l( L.GetDescriptor().GetDim(0), L.GetDescriptor().GetDim(1) );
	//Array<Complex> u( U.GetDescriptor().GetDim(0), U.GetDescriptor().GetDim(1) );
	//
	//data.print();
	//
	//data.LU( &l, &u, &P );
	//
	//Array<Complex> data2 = l*u;
	//data2.print();
	//
	//TEST_CALL( P*data, l*u, BOOST_CURRENT_FUNCTION );
	//TEST_CALL( L*U, l*u, BOOST_CURRENT_FUNCTION );
	//TEST_CALL( L, l, BOOST_CURRENT_FUNCTION );
	//TEST_CALL( U, u, BOOST_CURRENT_FUNCTION );
	//
	//Array<Complex> data = read_file_matrix<Complex>( std::string("unit_tests_files//draft//data.txt") );
	//Array<Complex> L = read_file_matrix<Complex>( std::string("unit_tests_files//draft//l.txt") );
	//Array<Complex> U = read_file_matrix<Complex>( std::string("unit_tests_files//draft//u.txt") );
	//
	//Array<Complex> l( L.GetDescriptor().GetDim(0), L.GetDescriptor().GetDim(1) );
	//Array<Complex> u( U.GetDescriptor().GetDim(0), U.GetDescriptor().GetDim(1) );
	//
	//data.LU( &l, &u );
	//
	//Array<Complex> data2 = l*u;
	//
	//TEST_CALL( L*U, l*u, BOOST_CURRENT_FUNCTION );
	//TEST_CALL( L, l, BOOST_CURRENT_FUNCTION );
	//TEST_CALL( U, u, BOOST_CURRENT_FUNCTION );
//}