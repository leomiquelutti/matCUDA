#include "matCUDA.h"

using namespace matCUDA;

int main()
{
	// just an example
	Array<ComplexDouble> test = rand<ComplexDouble>( 10, 10 );
	test.print();

	return 0;
}