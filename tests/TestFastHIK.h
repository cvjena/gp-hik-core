#ifndef _TESTFASTHIK_H
#define _TESTFASTHIK_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GMHIKernel.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that all important methods of the gp-hik framework perform as desired
 */
class TestFastHIK : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestFastHIK );
    
    CPPUNIT_TEST(testKernelMultiplication);
    CPPUNIT_TEST(testKernelMultiplicationFast);
    CPPUNIT_TEST(testKernelSum);
    CPPUNIT_TEST(testKernelSumFast);
    CPPUNIT_TEST(testLUTUpdate);
    CPPUNIT_TEST(testLinSolve);
    CPPUNIT_TEST(testKernelVector);
    
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();

    /**
    * Constructor / Destructor testing 
    */  
    void testKernelMultiplication();
    void testKernelMultiplicationFast();
    void testKernelSum();
    void testKernelSumFast();
    void testLUTUpdate();
    void testLinSolve();
    void testKernelVector();

};

#endif // _TESTFASTHIK_H
