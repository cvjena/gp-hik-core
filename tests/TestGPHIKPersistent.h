#ifndef _TESTGPHIKPERSISTENT_H
#define _TESTGPHIKPERSISTENT_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GPHIKClassifier.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that GPHIKClassifier methods herited from Persistent (store and restore) work as desired.
 * @author Alexander Freytag
 * @date 21-12-2013
 */
class TestGPHIKPersistent : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestGPHIKPersistent );
	 CPPUNIT_TEST(testPersistentMethods);
      
    CPPUNIT_TEST_SUITE_END();
  
 private:
 
 public:
    void setUp();
    void tearDown();


    void testPersistentMethods();
};

#endif // _TESTGPHIKPERSISTENT_H
