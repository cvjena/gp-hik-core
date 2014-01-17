#ifndef _TESTGPHIKONLINELEARNABLE_H
#define _TESTGPHIKONLINELEARNABLE_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GPHIKClassifier.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that GPHIKClassifierIL methods herited from OnlineLearnable (addExample and addMultipleExamples) work as desired.
 * @author Alexander Freytag
 * @date 03-01-2014 (dd-mm-yyyy)
 */
class TestGPHIKOnlineLearnable : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestGPHIKOnlineLearnable );
      CPPUNIT_TEST(testOnlineLearningStartEmpty);
      CPPUNIT_TEST(testOnlineLearningOCCtoBinary);
      CPPUNIT_TEST(testOnlineLearningBinarytoMultiClass);
      CPPUNIT_TEST(testOnlineLearningMultiClass);
      
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();

    void testOnlineLearningStartEmpty();    
    
    void testOnlineLearningOCCtoBinary();
    
    void testOnlineLearningBinarytoMultiClass();

    void testOnlineLearningMultiClass();
};

#endif // _TESTGPHIKONLINELEARNABLE_H
