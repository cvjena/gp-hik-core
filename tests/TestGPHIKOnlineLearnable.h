#ifndef _TESTGPHIKONLINELEARNABLE_H
#define _TESTGPHIKONLINELEARNABLE_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GPHIKClassifier.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that GPHIKClassifierIL methods herited from OnlineLearnable (addExample and addMultipleExamples) work as desired.
 * @author Alexander Freytag
 * @date 03-11-2014 (dd-mm-yyyy)
 */
class TestGPHIKOnlineLearnable : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestGPHIKOnlineLearnable );
	 CPPUNIT_TEST(testOnlineLearningMethods);
      
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();


    void testOnlineLearningMethods();
};

#endif // _TESTGPHIKONLINELEARNABLE_H
