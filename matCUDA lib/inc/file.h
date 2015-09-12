
// Test

#define rep(z, n, text) \
	descriptor.push_back(x ## n);

#define rep2(z, n, text) \
	*x##n

#if !BOOST_PP_IS_ITERATING

   #ifndef FILE_H_
   #define FILE_H_

   #include <boost/preprocessor/iteration/iterate.hpp>

   #define BOOST_PP_FILENAME_1 "file.h"
   #define BOOST_PP_ITERATION_LIMITS (1, 32)
   #include BOOST_PP_ITERATE()

   #endif

#else

	Array(	BOOST_PP_ENUM_PARAMS(BOOST_PP_ITERATION(), const index_t x))
			: m_data(1 BOOST_PP_REPEAT(BOOST_PP_ITERATION(), rep2, ~)),
			m_indexer(NULL),
			m_padded(false)
	{
		std::vector<index_t> descriptor;
		BOOST_PP_REPEAT(BOOST_PP_ITERATION(), rep, ~)

		int elements = m_data.m_numElements;
		if(descriptor.size() == 1)
		{
			descriptor.push_back(1);
			m_padded = true;
		}
		
		m_indexer = new LinearIndexer(descriptor);
	}

#endif