#ifdef NICE_USELIB_CPPUNIT

#include <string>
#include <exception>
#include <map>

#include <gp-hik-core/SortedVectorSparse.h>
#include "TestVectorSorter.h"


using namespace NICE;
using namespace std;

const bool verboseStartEnd = true;

CPPUNIT_TEST_SUITE_REGISTRATION( TestVectorSorter );

void TestVectorSorter::setUp() {
}

void TestVectorSorter::tearDown() {
}

void TestVectorSorter::checkData ( const vector<double> & all_elements, const NICE::SortedVectorSparse<double> & vSS, double sparse_tolerance )
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::checkData ===================== " << std::endl;
  
  vector< pair<double, int> > all_elements_sorted;

  vector< pair<double, int> > nonzero_elements;
  for (uint i = 0 ; i < all_elements.size(); i++ )
  {
    if ( fabs(all_elements[i]) > sparse_tolerance ) {
      nonzero_elements.push_back( pair<double, int> ( all_elements[i], i ) );
      all_elements_sorted.push_back( pair<double, int> ( all_elements[i], i ) );
    } else {
      all_elements_sorted.push_back( pair<double, int> ( 0.0, i ) );
    }
  }

  sort ( nonzero_elements.begin(), nonzero_elements.end() );
  sort ( all_elements_sorted.begin(), all_elements_sorted.end() );

  // looping through all non-zero values
  uint k = 0;
  for (NICE::SortedVectorSparse<double>::const_elementpointer it = vSS.nonzeroElements().begin(); it != vSS.nonzeroElements().end(); it++,k++)
	{
    CPPUNIT_ASSERT_DOUBLES_EQUAL( nonzero_elements[k].first, it->first, 0.0 );
    CPPUNIT_ASSERT_EQUAL( nonzero_elements[k].second, it->second.first );
	}

  // 2 3 0 1 5 4
	std::vector<int> vSSPerm(vSS.getPermutation());
	for (int k = 0;k < vSSPerm.size(); k++)
	{
    CPPUNIT_ASSERT_EQUAL( all_elements_sorted[k].second, vSSPerm[k] );
	}
	
	std::vector<int> vSSPermNNZ (vSS.getPermutationNonZero());
  vector<pair<int,double> > sv ( vSS.getOrderInSeparateVector() );
	for (int k = 0;k < vSSPermNNZ.size(); k++)
	{
    CPPUNIT_ASSERT_EQUAL( nonzero_elements[k].second, vSSPermNNZ[k] );
    CPPUNIT_ASSERT_EQUAL( sv[k].first, vSSPermNNZ[k] );
    CPPUNIT_ASSERT_EQUAL( sv[k].second, vSS.access( sv[k].first ) );
	}

//   cerr << endl;
  for (int k = 0;k < vSS.getN();k++)
  {
    CPPUNIT_ASSERT_DOUBLES_EQUAL( all_elements[k], vSS.access(k), sparse_tolerance ); 
//     cerr << "Element " << k << " = " << vSS.access(k) << endl;
  }
//     vSS.print();

  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::checkData done ===================== " << std::endl;  

}

void TestVectorSorter::testVectorSorter() 
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::testVectorSorter ===================== " << std::endl;
  
  vector<double> all_elements;
  all_elements.push_back(2);
  all_elements.push_back(4);
  all_elements.push_back(0);
  all_elements.push_back(1e-7);
  all_elements.push_back(7);
  all_elements.push_back(5);

  double sparse_tolerance = 1e-7;

  // Now we put everything in a vectorsortersparse object
  NICE::SortedVectorSparse<double> vSS;
	vSS.setTolerance(sparse_tolerance);
  for (uint i = 0 ; i < all_elements.size(); i++ )
    vSS.insert( all_elements[i] );

  checkData( all_elements, vSS );
  

//   cerr << endl;
//   cerr << "v[1] = 3.0 ";
  vSS.set(1, 3.0);
  all_elements[1] = 3.0;
  checkData( all_elements, vSS );
  
//   cerr << endl;
//   cerr << "v[1] = 0.0 ";
  vSS.set(1, 0.0);
  all_elements[1] = 0.0;
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "v[5] = -3.0 ";
  vSS.set(5, -3.0);
  all_elements[5] = -3.0;
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "add 13.0 ";
  vSS.insert(13.0);
  all_elements.push_back(13.0);
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "add 0.0 ";
  vSS.insert(0.0);
  all_elements.push_back(0.0);
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "v[0] = -10.0 ";
  vSS.set(0, -10.0);
  all_elements[0] = -10.0;
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "v[5] = 2.0 ";
  vSS.set(5, 2.0);
  all_elements[5] = 2.0;
  checkData( all_elements, vSS );

//   cerr << endl;
//   cerr << "v[5] = 0.0 ";
  vSS.set(5, 0.0);
  all_elements[5] = 0.0;
  checkData( all_elements, vSS ); 

  SortedVectorSparse<double> vSS_copy;
  vSS_copy = vSS;
  checkData ( all_elements, vSS_copy );

  SortedVectorSparse<double> vSS_all;
  vSS_all.insert ( all_elements );
  checkData ( all_elements, vSS_all, 0.0 );
  
  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::testVectorSorter done ===================== " << std::endl;  
}

void TestVectorSorter::testMultiMap()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::testMultiMap ===================== " << std::endl;
  
  multimap<int, double> d;
  multimap<int, double>::iterator it1 = d.insert ( pair<int, double> ( 3, 3.0 ) );
  multimap<int, double>::iterator it2 = d.insert ( pair<int, double> ( 1, 1.0 ) );
  multimap<int, double>::iterator it3 = d.insert ( pair<int, double> ( 2, 2.0 ) );
  multimap<int, double>::iterator it4 = d.insert ( pair<int, double> ( 5, 5.0 ) );

  it1->second = 1.5;
  CPPUNIT_ASSERT_EQUAL ( 1.5, it1->second );
  CPPUNIT_ASSERT_EQUAL ( 1.0, it2->second );
  CPPUNIT_ASSERT_EQUAL ( 2.0, it3->second );
  CPPUNIT_ASSERT_EQUAL ( 5.0, it4->second );
  
  if (verboseStartEnd)
    std::cerr << "================== TestVectorSorter::testMultiMap done ===================== " << std::endl;  
}

#endif
