/** 
 * @file TestGPHIKPersistent.cpp
 * @brief CppUnit-Testcase to verify that GPHIKClassifier methods herited from Persistent (store and restore) work as desired.
 * @author Alexander Freytag
 * @date 21-12-2013
*/

#ifdef NICE_USELIB_CPPUNIT

// STL includes
#include <iostream>
#include <vector>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKClassifier.h"

#include "TestGPHIKPersistent.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

const bool verboseStartEnd = true;
const bool verbose = false;


CPPUNIT_TEST_SUITE_REGISTRATION( TestGPHIKPersistent );

void TestGPHIKPersistent::setUp() {
}

void TestGPHIKPersistent::tearDown() {
}
void TestGPHIKPersistent::testPersistentMethods()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKPersistent::testPersistentMethods ===================== " << std::endl;  
  
  NICE::Config conf;
  std::string trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  NICE::GPHIKClassifier * classifier;  
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain; 

  std::ifstream ifsTrain ( trainData.c_str() , ios::in );

  if ( ifsTrain.good() )
  {
    ifsTrain >> dataTrain;
    ifsTrain >> yBinTrain;
    ifsTrain >> yMultiTrain;
    ifsTrain.close();  
  }
  else 
  {
    std::cerr << "Unable to read training data from file " << trainData << " -- aborting." << std::endl;
    CPPUNIT_ASSERT ( ifsTrain.good() );
  } 
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }  
  
  // TRAIN CLASSIFIER FROM SCRATCH
  
  classifier = new GPHIKClassifier ( &conf );  
  
  yBinTrain *= 2;
  yBinTrain -= 1;
  yBinTrain *= -1;
  
  if ( verbose )
  {
    std::cerr << yBinTrain << std::endl;
  
    std::cerr << "train classifier with artifially disturbed labels" << std::endl;
    
  }
  
  classifier->train ( examplesTrain , yBinTrain);//yMultiTrain );
  
  
  // TEST STORING ABILITIES
  
  std::string s_destination_save ( "myClassifier.txt" );
  
  std::filebuf fbOut;
  fbOut.open ( s_destination_save.c_str(), ios::out );
  std::ostream os (&fbOut);
  //
  classifier->store( os );
  //   
  fbOut.close(); 
  
  
  // TEST RESTORING ABILITIES
    
  NICE::GPHIKClassifier * classifierRestored = new GPHIKClassifier();  
      
  std::string s_destination_load ( "myClassifier.txt" );
  
  std::filebuf fbIn;
  fbIn.open ( s_destination_load.c_str(), ios::in );
  std::istream is (&fbIn);
  //
  classifierRestored->restore( is );
  //   
  fbIn.close();   
  
  
  // TEST both classifiers to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 

  std::string testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  std::ifstream ifsTest ( testData.c_str(), ios::in );
  if ( ifsTest.good() )
  {
    ifsTest >> dataTest;
    ifsTest >> yBinTest;
    ifsTest >> yMultiTest;
    ifsTest.close();  
  }
  else 
  {
    std::cerr << "Unable to read test data, aborting." << std::endl;
    CPPUNIT_ASSERT ( ifsTest.good() );
  }
  
  // ------------------------------------------
  // ------------- PREPARATION --------------
  // ------------------------------------------   
  
  // determine classes known during training and corresponding mapping
  // thereby allow for non-continous class labels
  std::set< uint > classesKnownTraining = classifier->getKnownClassNumbers();
  
  uint noClassesKnownTraining ( classesKnownTraining.size() );
  std::map< uint, uint > mapClNoToIdxTrain;
  std::set< uint >::const_iterator clTrIt = classesKnownTraining.begin();
  for ( uint i=0; i < noClassesKnownTraining; i++, clTrIt++ )
      mapClNoToIdxTrain.insert ( std::pair< uint, uint > ( *clTrIt, i )  );
  
  // determine classes known during testing and corresponding mapping
  // thereby allow for non-continous class labels
  std::set< uint > classesKnownTest;
  classesKnownTest.clear();
  

  // determine which classes we have in our label vector
  // -> MATLAB: myClasses = unique(y);
  for ( NICE::Vector::const_iterator it = yMultiTest.begin(); it != yMultiTest.end(); it++ )
  {
    if ( classesKnownTest.find ( *it ) == classesKnownTest.end() )
    {
      classesKnownTest.insert ( *it );
    }
  }          
  
  uint noClassesKnownTest ( classesKnownTest.size() );  
  std::map< uint, uint > mapClNoToIdxTest;
  std::set< uint >::const_iterator clTestIt = classesKnownTest.begin();
  for ( uint i=0; i < noClassesKnownTest; i++, clTestIt++ )
      mapClNoToIdxTest.insert ( std::pair< uint, uint > ( *clTestIt, i )  ); 
          
  
  if ( verbose )
  {
    std::cout << "Train data mapping: " << std::endl;
    for ( std::map< uint, uint >::const_iterator clTrainIt = mapClNoToIdxTrain.begin(); clTrainIt != mapClNoToIdxTrain.end(); clTrainIt++ )
    {
      std::cout << " " << clTrainIt->first << " " << clTrainIt->second << std::endl;
    }

    std::cout << "Test data mapping: " << std::endl;
    for ( std::map< uint, uint >::const_iterator clTestIt = mapClNoToIdxTest.begin(); clTestIt != mapClNoToIdxTest.end(); clTestIt++ )
    {
      std::cout << " " << clTestIt->first << " " << clTestIt->second << std::endl;
    }    
  }
  
  NICE::Matrix confusionMatrix         ( noClassesKnownTraining, noClassesKnownTest, 0.0);
  NICE::Matrix confusionMatrixRestored ( noClassesKnownTraining, noClassesKnownTest, 0.0);
  
  int i_loopEnd  ( (int)dataTest.rows() );
  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( dataTest.getRow(i) );
    NICE::SparseVector scores;
    uint result;
    
    // classify with trained classifier 
    classifier->classify( &example, result, scores );
       
    
    confusionMatrix( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;

    // classify with restored classifier 
    scores.clear();
    classifierRestored->classify( &example, result, scores );
    
    confusionMatrixRestored( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;
    
    
  }  
    
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixRestored.normalizeColumnsL1();
  double arrRestored ( confusionMatrixRestored.trace()/confusionMatrixRestored.cols() );
  
  if ( verbose )
  {
    std::cout << "confusionMatrix: " << confusionMatrix << std::endl;
    std::cout << "confusionMatrixRestored: " << confusionMatrixRestored << std::endl;
    std::cout << "arr: " << arr << std::endl;
    std::cout << "arrRestored: " << arrRestored << std::endl;   
    
  }

  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrRestored, 1e-8);
  
  // don't waste memory
  //TODO clean up of training data, also in TestGPHIKPersistent
  
  delete classifier;
  delete classifierRestored;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  } 
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKPersistent::testPersistentMethods done ===================== " << std::endl;  
  
}

#endif
