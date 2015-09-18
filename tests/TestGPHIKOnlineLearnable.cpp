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
#include "gp-hik-core/GPHIKRawClassifier.h"

#include "TestGPHIKOnlineLearnable.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

const bool verboseStartEnd = true;
const bool verbose = false;
const bool writeClassifiersForVerification = false;


CPPUNIT_TEST_SUITE_REGISTRATION( TestGPHIKOnlineLearnable );

void TestGPHIKOnlineLearnable::setUp() {
}

void TestGPHIKOnlineLearnable::tearDown() {
}



void readData ( const std::string filename, NICE::Matrix & data, NICE::Vector & yBin, NICE::Vector & yMulti )
{
 std::ifstream ifs ( filename.c_str() , ios::in );

  if ( ifs.good() )
  {
    ifs >> data;
    ifs >> yBin;
    ifs >> yMulti;
    ifs.close();  
  }
  else 
  {
    std::cerr << "Unable to read data from file " << filename << " -- aborting." << std::endl;
    CPPUNIT_ASSERT ( ifs.good() );
  }    
}

void prepareLabelMappings (std::map< uint, uint > & mapClNoToIdxTrain, 
                           const GPHIKClassifier * classifier, 
                           std::map< uint,uint > & mapClNoToIdxTest, 
                           const NICE::Vector & yMultiTest
                          )
{
  // determine classes known during training and corresponding mapping
  // thereby allow for non-continous class labels
  std::set< uint > classesKnownTraining = classifier->getKnownClassNumbers();
  
  uint noClassesKnownTraining ( classesKnownTraining.size() );
  std::set< uint >::const_iterator clTrIt = classesKnownTraining.begin();
  for ( uint i=0; i < noClassesKnownTraining; i++, clTrIt++ )
      mapClNoToIdxTrain.insert ( std::pair< uint, uint > ( *clTrIt, i )  );
  
  // determine classes known during testing and corresponding mapping
  // thereby allow for non-continous class labels
  std::set< uint> classesKnownTest;
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
  std::set< uint >::const_iterator clTestIt = classesKnownTest.begin();
  for ( uint i=0; i < noClassesKnownTest; i++, clTestIt++ )
      mapClNoToIdxTest.insert ( std::pair< uint,uint > ( *clTestIt, i )  );   
}

void evaluateClassifier ( NICE::Matrix & confusionMatrix, 
                          const NICE::GPHIKClassifier * classifier, 
                          const NICE::Matrix & data,
                          const NICE::Vector & yMulti,
                          const std::map< uint,uint > & mapClNoToIdxTrain,
                          const std::map< uint,uint > & mapClNoToIdxTest
                        ) 
{
  int i_loopEnd  ( (int)data.rows() );  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( data.getRow(i) );
    NICE::SparseVector scores;
    uint result;    
    
    // classify with incrementally trained classifier 
    classifier->classify( &example, result, scores );
    
    confusionMatrix( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMulti[i])->second ) += 1.0;
  }
}

void evaluateClassifierRaw ( NICE::Matrix & confusionMatrix,
                          const NICE::GPHIKRawClassifier * classifier,
                          const NICE::Matrix & data,
                          const NICE::Vector & yMulti,
                          const std::map< uint,uint > & mapClNoToIdxTrain,
                          const std::map< uint,uint > & mapClNoToIdxTest
                        )
{
  int i_loopEnd  ( (int)data.rows() );

  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example_nonsparse ( data.getRow(i) );
    NICE::SparseVector example (example_nonsparse);
    NICE::SparseVector scores;
    uint result;

    // classify with incrementally trained classifier
    classifier->classify( &example, result, scores );

    uint gtlabel = mapClNoToIdxTest.find(yMulti[i])->second;
    uint predlabel = mapClNoToIdxTrain.find(result)->second;
    confusionMatrix( gtlabel, predlabel ) += 1.0;
  }
}


void compareClassifierOutputs ( const NICE::GPHIKClassifier * classifier,
                                const NICE::GPHIKClassifier * classifierScratch, 
                                const NICE::Matrix & data
                              )
{
  int i_loopEnd  ( (int)data.rows() );  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( data.getRow(i) );
    
    NICE::SparseVector scores;
    uint result;    
    
    // classify with incrementally trained classifier 
    classifier->classify( &example, result, scores );

    
    NICE::SparseVector scoresScratch;
    uint resultScratch;
    classifierScratch->classify( &example, resultScratch, scoresScratch );
    
    
    bool equal(true);
    NICE::SparseVector::const_iterator itScores        = scores.begin();
    NICE::SparseVector::const_iterator itScoresScratch = scoresScratch.begin();
    for ( ; itScores != scores.end(); itScores++, itScoresScratch++)
    {
      if ( fabs( itScores->second - itScores->second ) > 10e-3)
      {
        std::cerr << " itScores->second: " << itScores->second << " itScores->second: " << itScores->second << std::endl;
        equal = false;
        break;
      }        
    }
    
    CPPUNIT_ASSERT_EQUAL ( equal, true );     
  }  
}

void TestGPHIKOnlineLearnable::testOnlineLearningStartEmpty()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningStartEmpty ===================== " << std::endl;  
  
  NICE::Config conf;
  
  conf.sB ( "GPHIKClassifier", "eig_verbose", false);
  conf.sS ( "GPHIKClassifier", "optimization_method", "downhillsimplex");
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain; 
  
  readData ( s_trainData, dataTrain, yBinTrain, yMultiTrain );
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }
  
  //create classifier object
  NICE::GPHIKClassifier * classifier;
  classifier = new NICE::GPHIKClassifier ( &conf );  
  bool performOptimizationAfterIncrement ( true );

  // add training samples, but without running training method first
  classifier->addMultipleExamples ( examplesTrain,yMultiTrain, performOptimizationAfterIncrement );  
  
  // create second object trained in the standard way
  NICE::GPHIKClassifier * classifierScratch = new NICE::GPHIKClassifier ( &conf );
  classifierScratch->train ( examplesTrain, yMultiTrain );
  
    
  // TEST both classifiers to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yBinTest, yMultiTest );

    
  // ------------------------------------------
  // ------------- PREPARATION --------------
  // ------------------------------------------   
  
  // determine classes known during training/testing and corresponding mapping
  // thereby allow for non-continous class labels  
  std::map< uint,uint > mapClNoToIdxTrain;
  std::map< uint,uint > mapClNoToIdxTest;
  prepareLabelMappings (mapClNoToIdxTrain, classifier, mapClNoToIdxTest, yMultiTest);
  
  
  NICE::Matrix confusionMatrix         ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  NICE::Matrix confusionMatrixScratch  ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  
    
  // ------------------------------------------
  // ------------- CLASSIFICATION --------------
  // ------------------------------------------  
  evaluateClassifier ( confusionMatrix, classifier, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest ); 
  
  evaluateClassifier ( confusionMatrixScratch, classifierScratch, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest );  
  
    
  // post-process confusion matrices
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixScratch.normalizeColumnsL1();
  double arrScratch ( confusionMatrixScratch.trace()/confusionMatrixScratch.cols() );
  
  if ( verbose ) 
  {
    std::cerr << "confusionMatrix: " << confusionMatrix  << std::endl;
  
    std::cerr << "confusionMatrixScratch: " << confusionMatrixScratch << std::endl;
  }  

  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrScratch, 1e-8);
  
  // don't waste memory
  
  delete classifier;
  delete classifierScratch;  
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  }
  
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningStartEmpty done ===================== " << std::endl;   
}

void TestGPHIKOnlineLearnable::testOnlineLearningOCCtoBinary()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningOCCtoBinary ===================== " << std::endl;  
  
  NICE::Config conf;
  
  conf.sB ( "GPHIKClassifier", "eig_verbose", false);
  conf.sS ( "GPHIKClassifier", "optimization_method", "downhillsimplex");
  conf.sB ( "GPHIKClassifier", "verbose", true);
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain; 
  
  readData ( s_trainData, dataTrain, yBinTrain, yMultiTrain );
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  std::vector< const NICE::SparseVector *> examplesTrainPlus;
  std::vector< const NICE::SparseVector *> examplesTrainMinus;
  
  examplesTrain.resize( dataTrain.rows() );
  
  // to check whether non-consecutive and even wrongly odered class numbers work as well
  int clNoFirst  ( 2 );
  int clNoSecond ( 0 );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
    
    if ( yBinTrain[i] == 1 )
    {
      examplesTrainPlus.push_back ( *exTrainIt );
      yBinTrain[i] = clNoFirst;
    }
    else
    {
       examplesTrainMinus.push_back ( *exTrainIt );
       yBinTrain[i] = clNoSecond;
    }
  }
  NICE::Vector yBinPlus  ( examplesTrainPlus.size(), clNoFirst ) ;
  NICE::Vector yBinMinus ( examplesTrainMinus.size(), clNoSecond );
  
  
  //create classifier object
  NICE::GPHIKClassifier * classifier;
  classifier = new NICE::GPHIKClassifier ( &conf );  
  bool performOptimizationAfterIncrement ( true );

  // training with examples for positive class only
  classifier->train ( examplesTrainPlus, yBinPlus );
  // add samples for negative class, thereby going from OCC to binary setting
  classifier->addMultipleExamples ( examplesTrainMinus, yBinMinus, performOptimizationAfterIncrement );  
  
  // create second object trained in the standard way
  NICE::GPHIKClassifier * classifierScratch = new NICE::GPHIKClassifier ( &conf );
  classifierScratch->train ( examplesTrain, yBinTrain );
  
  NICE::GPHIKRawClassifier * classifierScratchRaw = new NICE::GPHIKRawClassifier ( &conf );
  classifierScratchRaw->train ( examplesTrain, yBinTrain );
    
  // TEST both classifiers to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yBinTest, yMultiTest );

    
  // ------------------------------------------
  // ------------- PREPARATION --------------
  // ------------------------------------------   
  
  // determine classes known during training/testing and corresponding mapping
  // thereby allow for non-continous class labels  
  std::map< uint,uint > mapClNoToIdxTrain;
  std::map< uint,uint > mapClNoToIdxTest;
  prepareLabelMappings (mapClNoToIdxTrain, classifier, mapClNoToIdxTest, yMultiTest);
  
  
  NICE::Matrix confusionMatrix         ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  NICE::Matrix confusionMatrixScratch  ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  NICE::Matrix confusionMatrixScratchRaw  ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
    
  // ------------------------------------------
  // ------------- CLASSIFICATION --------------
  // ------------------------------------------  
  evaluateClassifier ( confusionMatrix, classifier, dataTest, yBinTest,
                          mapClNoToIdxTrain, mapClNoToIdxTest );
  
  evaluateClassifier ( confusionMatrixScratch, classifierScratch, dataTest, yBinTest,
                          mapClNoToIdxTrain, mapClNoToIdxTest );
  
  evaluateClassifierRaw ( confusionMatrixScratchRaw, classifierScratchRaw, dataTest, yBinTest,
                          mapClNoToIdxTrain, mapClNoToIdxTest );
    
  // post-process confusion matrices
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixScratch.normalizeColumnsL1();
  double arrScratch ( confusionMatrixScratch.trace()/confusionMatrixScratch.cols() );

  confusionMatrixScratchRaw.normalizeColumnsL1();
  double arrScratchRaw ( confusionMatrixScratchRaw.trace()/confusionMatrixScratchRaw.cols() );
  
  if ( verbose ) 
  {
    std::cerr << "confusionMatrix: " << confusionMatrix  << std::endl;
  
    std::cerr << "confusionMatrixScratch: " << confusionMatrixScratch << std::endl;

    std::cerr << "confusionMatrixScratchRaw: " << confusionMatrixScratchRaw << std::endl;
  } 
  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrScratch, 1e-8);
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arrScratch, arrScratchRaw, 1e-8);
  
  // don't waste memory
  
  delete classifier;
  delete classifierScratch;  
  delete classifierScratchRaw;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  }  
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningOCCtoBinary done ===================== " << std::endl;   
}

void TestGPHIKOnlineLearnable::testOnlineLearningBinarytoMultiClass()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningBinarytoMultiClass ===================== " << std::endl;   

  NICE::Config conf;
  
  conf.sB ( "GPHIKClassifier", "eig_verbose", false);
  conf.sS ( "GPHIKClassifier", "optimization_method", "downhillsimplex");
//   conf.sS ( "GPHIKClassifier", "optimization_method", "none");
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain; 
  
  readData ( s_trainData, dataTrain, yBinTrain, yMultiTrain );
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  std::vector< const NICE::SparseVector *> examplesTrain12;
  std::vector< const NICE::SparseVector *> examplesTrain3;
  
  NICE::Vector yMulti12  ( yMultiTrain.size(), 1 ) ;
  NICE::Vector yMulti3  ( yMultiTrain.size(), 1 ) ;
  int cnt12 ( 0 );
  int cnt3 ( 0 );
  
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
    
    if ( ( yMultiTrain[i] == 0 ) || ( yMultiTrain[i] == 1 ) )
    {
      examplesTrain12.push_back ( *exTrainIt );
      yMulti12[ cnt12 ] = yMultiTrain[i];    
      cnt12++;
    }
    else
    {
       examplesTrain3.push_back ( *exTrainIt );
       yMulti3[cnt3] = 2;
       cnt3++;
    }
  }
  
  yMulti12.resize ( examplesTrain12.size() );
  yMulti3.resize ( examplesTrain3.size() );  
  
  //create classifier object
  NICE::GPHIKClassifier * classifier;
  classifier = new NICE::GPHIKClassifier ( &conf );  
  bool performOptimizationAfterIncrement ( true );

  // training with examples for first and second class only
  classifier->train ( examplesTrain12, yMulti12 );
  // add samples for third class, thereby going from binary to multi-class setting
  classifier->addMultipleExamples ( examplesTrain3, yMulti3, performOptimizationAfterIncrement );  
  
  // create second object trained in the standard way
  NICE::GPHIKClassifier * classifierScratch = new NICE::GPHIKClassifier ( &conf );
  classifierScratch->train ( examplesTrain, yMultiTrain );
  
    
  // TEST both classifiers to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yBinTest, yMultiTest );

    
  // ------------------------------------------
  // ------------- PREPARATION --------------
  // ------------------------------------------   
  
  // determine classes known during training/testing and corresponding mapping
  // thereby allow for non-continous class labels  
  std::map< uint,uint > mapClNoToIdxTrain;
  std::map< uint,uint > mapClNoToIdxTest;
  prepareLabelMappings (mapClNoToIdxTrain, classifier, mapClNoToIdxTest, yMultiTest);
  
  
  NICE::Matrix confusionMatrix         ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  NICE::Matrix confusionMatrixScratch  ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  
    
  // ------------------------------------------
  // ------------- CLASSIFICATION --------------
  // ------------------------------------------  
  
  compareClassifierOutputs ( classifier, classifierScratch, dataTest ); 
  evaluateClassifier ( confusionMatrix, classifier, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest ); 
  
  evaluateClassifier ( confusionMatrixScratch, classifierScratch, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest );  
  
    
  // post-process confusion matrices
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixScratch.normalizeColumnsL1();
  double arrScratch ( confusionMatrixScratch.trace()/confusionMatrixScratch.cols() );
  
  if ( verbose ) 
  {
    std::cerr << "confusionMatrix: " << confusionMatrix  << std::endl;
  
    std::cerr << "confusionMatrixScratch: " << confusionMatrixScratch << std::endl;
  }

  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrScratch, 1e-8);
  
  // don't waste memory
  
  delete classifier;
  delete classifierScratch;  
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  } 
  
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningBinarytoMultiClass done ===================== " << std::endl;   
}

void TestGPHIKOnlineLearnable::testOnlineLearningMultiClass()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningMultiClass ===================== " << std::endl;  
  
  NICE::Config conf;
  
  conf.sB ( "GPHIKClassifier", "eig_verbose", false);
  conf.sS ( "GPHIKClassifier", "optimization_method", "downhillsimplex");//downhillsimplex greedy
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain; 
  
  readData ( s_trainData, dataTrain, yBinTrain, yMultiTrain );

  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows()-1 );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows()-1; i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }  
  
  // TRAIN INITIAL CLASSIFIER FROM SCRATCH
  NICE::GPHIKClassifier * classifier;
  classifier = new NICE::GPHIKClassifier ( &conf );

  //use all but the first example for training and add the first one lateron
  NICE::Vector yMultiRelevantTrain  ( yMultiTrain.getRangeRef( 0, yMultiTrain.size()-2  ) );
  
  classifier->train ( examplesTrain , yMultiRelevantTrain );
  
  
  // RUN INCREMENTAL LEARNING
  
  bool performOptimizationAfterIncrement ( true );
  
  NICE::SparseVector * exampleToAdd = new NICE::SparseVector ( dataTrain.getRow( (int)dataTrain.rows()-1 ) );
  classifier->addExample ( exampleToAdd, yMultiTrain[ (int)dataTrain.rows()-2 ], performOptimizationAfterIncrement );
  
  if ( verbose )
    std::cerr << "label of example to add: " << yMultiTrain[ (int)dataTrain.rows()-1 ] << std::endl;
  
  // TRAIN SECOND CLASSIFIER FROM SCRATCH USING THE SAME OVERALL AMOUNT OF EXAMPLES
  examplesTrain.push_back(  exampleToAdd );

  NICE::GPHIKClassifier * classifierScratch = new NICE::GPHIKClassifier ( &conf );
  classifierScratch->train ( examplesTrain, yMultiTrain );
  
  if ( verbose )
    std::cerr << "trained both classifiers - now start evaluating them" << std::endl;
  
  
  // TEST that both classifiers produce equal store-files
  if ( writeClassifiersForVerification )
  {
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
  }
  
  
  // TEST both classifiers to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yBinTest, yMultiTest );

    
  // ------------------------------------------
  // ------------- PREPARATION --------------
  // ------------------------------------------   
  
  // determine classes known during training/testing and corresponding mapping
  // thereby allow for non-continous class labels  
  std::map< uint,uint > mapClNoToIdxTrain;
  std::map< uint,uint > mapClNoToIdxTest;
  prepareLabelMappings (mapClNoToIdxTrain, classifier, mapClNoToIdxTest, yMultiTest);
  
  
  NICE::Matrix confusionMatrix         ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
  NICE::Matrix confusionMatrixScratch    ( mapClNoToIdxTrain.size(), mapClNoToIdxTest.size(), 0.0);
    
  // ------------------------------------------
  // ------------- CLASSIFICATION --------------
  // ------------------------------------------  
  evaluateClassifier ( confusionMatrix, classifier, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest ); 
  
  evaluateClassifier ( confusionMatrixScratch, classifierScratch, dataTest, yMultiTest,
                          mapClNoToIdxTrain,mapClNoToIdxTest );  
  
    
  // post-process confusion matrices
  confusionMatrix.normalizeColumnsL1();
  double arr ( confusionMatrix.trace()/confusionMatrix.cols() );

  confusionMatrixScratch.normalizeColumnsL1();
  double arrScratch ( confusionMatrixScratch.trace()/confusionMatrixScratch.cols() );

  if ( verbose ) 
  {
    std::cerr << "confusionMatrix: " << confusionMatrix  << std::endl;
  
    std::cerr << "confusionMatrixScratch: " << confusionMatrixScratch << std::endl;
  }
  
  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( arr, arrScratch, 1e-8);
  
  // don't waste memory
  
  delete classifier;
  delete classifierScratch;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  } 
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKOnlineLearnable::testOnlineLearningMultiClass done ===================== " << std::endl;  
  
}

#endif
