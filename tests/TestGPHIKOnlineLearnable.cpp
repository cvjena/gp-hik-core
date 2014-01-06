/** 
 * @file TestGPHIKOnlineLearnable.cpp
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

#include "TestGPHIKOnlineLearnable.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

const bool verboseStartEnd = true;


CPPUNIT_TEST_SUITE_REGISTRATION( TestGPHIKOnlineLearnable );

void TestGPHIKOnlineLearnable::setUp() {
}

void TestGPHIKOnlineLearnable::tearDown() {
}
void TestGPHIKOnlineLearnable::testOnlineLearningMethods()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningMethods ===================== " << std::endl;  
  
  NICE::Config conf;
  
  conf.sB ( "GPHIKClassifier", "eig_verbose", false);
  conf.sS ( "GPHIKClassifier", "optimization_method", "downhillsimplex");
  
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
  std::vector< NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows()-1 );
  
  std::vector< NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows()-1; i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }  
  
  // TRAIN INITIAL CLASSIFIER FROM SCRATCH
  
  classifier = new NICE::GPHIKClassifier ( &conf );

  //use all but the first example for training and add the first one lateron
  NICE::Vector yMultiRelevantTrain  ( yMultiTrain.getRangeRef( 0, yMultiTrain.size()-2  ) );

  std::cerr << "yMultiRelevantTrain: " << yMultiRelevantTrain << std::endl;
  
  classifier->train ( examplesTrain , yMultiRelevantTrain );
  
  std::cerr << "Training done -- start incremental learning " << std::endl;
  
  // RUN INCREMENTAL LEARNING
  
  bool performOptimizationAfterIncrement ( false );
  
  NICE::SparseVector * exampleToAdd = new NICE::SparseVector ( dataTrain.getRow( (int)dataTrain.rows()-1 ) );
  classifier->addExample ( exampleToAdd, yMultiTrain[ (int)dataTrain.rows()-2 ], performOptimizationAfterIncrement );
  
  std::cerr << "label of example to add: " << yMultiTrain[ (int)dataTrain.rows()-1 ] << std::endl;
  
  // TRAIN SECOND CLASSIFIER FROM SCRATCH USING THE SAME OVERALL AMOUNT OF EXAMPLES
  examplesTrain.push_back(  exampleToAdd );

  NICE::GPHIKClassifier * classifierScratch = new NICE::GPHIKClassifier ( &conf );
  classifierScratch->train ( examplesTrain, yMultiTrain );
  
  std::cerr << "trained both classifiers - now start evaluating them" << std::endl;
  
  
  // TEST that both classifiers produce equal store-files
   std::string s_destination_save_IL ( "myClassifierIL.txt" );
  
  std::filebuf fbOut;
  fbOut.open ( s_destination_save_IL.c_str(), ios::out );
  std::ostream os (&fbOut);
  //
  classifier->store( os );
  //   
  fbOut.close(); 
  
  std::string s_destination_save_scratch ( "myClassifierScratch.txt" );
  
  std::filebuf fbOutScratch;
  fbOutScratch.open ( s_destination_save_scratch.c_str(), ios::out );
  std::ostream osScratch (&fbOutScratch);
  //
  classifierScratch->store( osScratch );
  //   
  fbOutScratch.close(); 
  
  
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
  std::set<int> classesKnownTraining = classifier->getKnownClassNumbers();
  
  int noClassesKnownTraining ( classesKnownTraining.size() );
  std::map<int,int> mapClNoToIdxTrain;
  std::set<int>::const_iterator clTrIt = classesKnownTraining.begin();
  for ( int i=0; i < noClassesKnownTraining; i++, clTrIt++ )
      mapClNoToIdxTrain.insert ( std::pair<int,int> ( *clTrIt, i )  );
  
  // determine classes known during testing and corresponding mapping
  // thereby allow for non-continous class labels
  std::set<int> classesKnownTest;
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
  
  int noClassesKnownTest ( classesKnownTest.size() );  
  std::map<int,int> mapClNoToIdxTest;
  std::set<int>::const_iterator clTestIt = classesKnownTest.begin();
  for ( int i=0; i < noClassesKnownTest; i++, clTestIt++ )
      mapClNoToIdxTest.insert ( std::pair<int,int> ( *clTestIt, i )  ); 
          
  
  NICE::Matrix confusionMatrix         ( noClassesKnownTraining, noClassesKnownTest, 0.0);
  NICE::Matrix confusionMatrixScratch    ( noClassesKnownTraining, noClassesKnownTest, 0.0);
  
  std::cerr << "data preparation for testing is done "<< std::endl;
  
  int i_loopEnd  ( (int)dataTest.rows() );
  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( dataTest.getRow(i) );
    NICE::SparseVector scores;
    int result;
    
    
    // classify with incrementally trained classifier 
    classifier->classify( &example, result, scores );
    std::cerr << "results with IL classifier: " << std::endl;
    scores.store ( std::cerr );   
    
    confusionMatrix( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;

    // classify with classifier learned from scratch
    scores.clear();
    classifierScratch->classify( &example, result, scores );
    std::cerr << "Results with scratch classifier: " << std::endl;
    scores.store( std::cerr );
    std::cerr << std::endl;
    
    confusionMatrixScratch( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;
  }  
  
  //TODO also check that both classifiers result in the same store-files
    
  std::cerr <<  "postprocess confusion matrices " << std::endl;
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixScratch.normalizeColumnsL1();
  double arrScratch ( confusionMatrixScratch.trace()/confusionMatrixScratch.cols() );

  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrScratch, 1e-8);
  
  // don't waste memory
  //TODO clean up of training data, also in TestGPHIKPersistent
  
  delete classifier;
  delete classifierScratch;
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningMethods done ===================== " << std::endl;  
  
}

#endif
