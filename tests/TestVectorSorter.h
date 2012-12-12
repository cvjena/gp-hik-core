#ifndef _TESTVECTORSORTER_H
#define _TESTVECTORSORTER_H

#include <cppunit/extensions/HelperMacros.h>
#include <gp-hik-core/GMHIKernel.h>

/**
 * CppUnit-Testcase. 
 * @brief CppUnit-Testcase to verify that all important methods of the SortedVectorSparse class perform as desired
 */
class TestVectorSorter : public CppUnit::TestFixture {

    CPPUNIT_TEST_SUITE( TestVectorSorter );
    
    CPPUNIT_TEST(testVectorSorter);
    CPPUNIT_TEST(testMultiMap);
    
    CPPUNIT_TEST_SUITE_END();
  
 private:
    void checkData ( const std::vector<double> & all_elements, const NICE::SortedVectorSparse<double> & vSS, double sparse_tolerance = 1e-7 );
 
 public:
    void setUp();
    void tearDown();

    /**
    * Constructor / Destructor testing 
    */  
    void testVectorSorter();

    void testMultiMap();

};

#endif // _TESTFASTHIK_H
