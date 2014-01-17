#ifndef _TESTGPHIKREGRESSION_H
#define _TESTGPHIKREGRESSION_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GPHIKRegression.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that GPHIKRegression works as desired.
 * @author Alexander Freytag
 * @date 16-01-2014 (dd-mm-yyyy)
 */
class TestGPHIKRegression : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestGPHIKRegression );
      CPPUNIT_TEST(testRegressionHoldInData);
      CPPUNIT_TEST(testRegressionHoldOutData);
      CPPUNIT_TEST(testRegressionOnlineLearning);
      
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();

    void testRegressionHoldInData();
    void testRegressionHoldOutData();    
    
    void testRegressionOnlineLearning();
    
};

#endif // _TESTGPHIKREGRESSION_H
