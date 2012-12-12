#ifndef _TESTFEATUREMATRIXT_H
#define _TESTFEATUREMATRIXT_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/FeatureMatrixT.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that all important methods of the Feature Matrix perform as desired
 */
class TestFeatureMatrixT : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestFeatureMatrixT );
	 CPPUNIT_TEST(testSetup);
	 CPPUNIT_TEST(testMatlabIO);
      
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();

    /**
    * Constructor / Destructor testing 
    */  
		void testSetup();
		void testMatlabIO();
};

#endif // _TESTFEATUREMATRIXT_H
