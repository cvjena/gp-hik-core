/** 
 * @file TestGPHIKRegression.cpp
 * @brief CppUnit-Testcase to verify that GPHIKRegression works as desired.
 * @author Alexander Freytag
 * @date 16-01-2014 (dd-mm-yyyy)
*/

#ifdef NICE_USELIB_CPPUNIT

// STL includes
#include <iostream>
#include <vector>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKRegression.h"

#include "TestGPHIKRegression.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

const bool verboseStartEnd = true;
const bool verbose = false;


CPPUNIT_TEST_SUITE_REGISTRATION( TestGPHIKRegression );

void TestGPHIKRegression::setUp() {
}

void TestGPHIKRegression::tearDown() {
}



void readData ( const std::string filename, NICE::Matrix & data, NICE::Vector & yValues )
{
 std::ifstream ifs ( filename.c_str() , ios::in );

  if ( ifs.good() )
  {
    NICE::Vector tmp;
    ifs >> data;
    ifs >> tmp; //yBin;
    ifs >> yValues;
    ifs.close();  
  }
  else 
  {
    std::cerr << "Unable to read data from file " << filename << " -- aborting." << std::endl;
    CPPUNIT_ASSERT ( ifs.good() );
  }    
}

void evaluateRegressionMethod ( double & regressionLoss, 
                          const NICE::GPHIKRegression * regressionMethod, 
                          const NICE::Matrix & data,
                          const NICE::Vector & yValues
                        ) 
{
  regressionLoss = 0.0;
  
  int i_loopEnd  ( (int)data.rows() );  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( data.getRow(i) );
    double result;    
    
    // classify with previously trained regression method
    regressionMethod->estimate( &example, result );
    
    if ( verbose )
      std::cerr << "i: " << i << " gt: " << yValues[i] << " result: " << result << std::endl;
    
    //use L2-loss for evaluation
    regressionLoss +=  pow( yValues[i] - result, 2 ); 
  }
}

void TestGPHIKRegression::testRegressionHoldInData()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionHoldInData ===================== " << std::endl;  
  
  NICE::Config conf;
  
  conf.sB ( "GPHIKRegression", "eig_verbose", false);
  conf.sS ( "GPHIKRegression", "optimization_method", "downhillsimplex");
  // set pretty low built-in noise for hold-in regression estimation
  conf.sD ( "GPHIKRegression", "noise", 1e-6 );
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yValues; 
  
  readData ( s_trainData, dataTrain, yValues );
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }
    
  //create regressionMethod object
  NICE::GPHIKRegression * regressionMethod;
  regressionMethod = new NICE::GPHIKRegression ( &conf );
  regressionMethod->train ( examplesTrain , yValues );
  
  double holdInLoss ( 0.0 );
  
    
  // ------------------------------------------
  // ------------- REGRESSION --------------
  // ------------------------------------------  
  evaluateRegressionMethod ( holdInLoss, regressionMethod, dataTrain, yValues ); 
  
  
  if ( verbose ) 
  {
    std::cerr << " holdInLoss: " << holdInLoss << std::endl;
  }  

  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( 0.0, holdInLoss, 1e-8);
  
  // don't waste memory
  
  delete regressionMethod;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  }
  
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionHoldInData done ===================== " << std::endl;   
}

void TestGPHIKRegression::testRegressionHoldOutData()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionHoldOutData ===================== " << std::endl;  

  NICE::Config conf;
  
  conf.sB ( "GPHIKRegression", "eig_verbose", false);
  conf.sS ( "GPHIKRegression", "optimization_method", "downhillsimplex");
  // set higher built-in noise for hold-out regression estimation
  conf.sD ( "GPHIKRegression", "noise", 1e-4 );
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yValues; 
  
  readData ( s_trainData, dataTrain, yValues );
  
  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }
    
  //create regressionMethod object
  NICE::GPHIKRegression * regressionMethod;
  regressionMethod = new NICE::GPHIKRegression ( &conf, "GPHIKRegression" );
  regressionMethod->train ( examplesTrain , yValues );
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yValuesTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yValuesTest );  
  
  double holdOutLoss ( 0.0 );
  
    
  // ------------------------------------------
  // ------------- REGRESSION --------------
  // ------------------------------------------  
  evaluateRegressionMethod ( holdOutLoss, regressionMethod, dataTest, yValuesTest ); 

  // acceptable difference for every estimated y-value on average
  double diffOkay ( 0.4 );
  
  if ( verbose ) 
  {
    std::cerr << " holdOutLoss: " << holdOutLoss << " accepting: " << pow(diffOkay,2)*yValuesTest.size() << std::endl;
  }  
  
  CPPUNIT_ASSERT( pow(diffOkay,2)*yValuesTest.size() - holdOutLoss > 0.0);
  
  // don't waste memory
  
  delete regressionMethod;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  }  
  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionHoldOutData done ===================== " << std::endl;     
}
    
void TestGPHIKRegression::testRegressionOnlineLearning()
{
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionOnlineLearning ===================== " << std::endl;  

  NICE::Config conf;
  
  conf.sB ( "GPHIKRegressionMethod", "eig_verbose", false);
  conf.sS ( "GPHIKRegressionMethod", "optimization_method", "downhillsimplex");//downhillsimplex greedy
  // set higher built-in noise for hold-out regression estimation
  conf.sD ( "GPHIKRegression", "noise", 1e-4 );  
  
  std::string s_trainData = conf.gS( "main", "trainData", "toyExampleSmallScaleTrain.data" );
  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yValuesTrain; 
  
  readData ( s_trainData, dataTrain, yValuesTrain );

  //----------------- convert data to sparse data structures ---------
  std::vector< const NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows()-1 );
  
  std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows()-1; i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }  
  
  // TRAIN INITIAL CLASSIFIER FROM SCRATCH
  NICE::GPHIKRegression * regressionMethod;
  regressionMethod = new NICE::GPHIKRegression ( &conf, "GPHIKRegression" );

  //use all but the first example for training and add the first one lateron
  NICE::Vector yValuesRelevantTrain  ( yValuesTrain.getRangeRef( 0, yValuesTrain.size()-2  ) );
  
  regressionMethod->train ( examplesTrain , yValuesRelevantTrain );
  
  
  // RUN INCREMENTAL LEARNING
  
  bool performOptimizationAfterIncrement ( true );
  
  NICE::SparseVector * exampleToAdd = new NICE::SparseVector ( dataTrain.getRow( (int)dataTrain.rows()-1 ) );
  
  
  regressionMethod->addExample ( exampleToAdd, yValuesTrain[ (int)dataTrain.rows()-2 ], performOptimizationAfterIncrement );
  
  if ( verbose )
    std::cerr << "label of example to add: " << yValuesTrain[ (int)dataTrain.rows()-1 ] << std::endl;
  
  // TRAIN SECOND REGRESSOR FROM SCRATCH USING THE SAME OVERALL AMOUNT OF EXAMPLES
  examplesTrain.push_back(  exampleToAdd );

  NICE::GPHIKRegression * regressionMethodScratch = new NICE::GPHIKRegression ( &conf, "GPHIKRegression" );
  regressionMethodScratch->train ( examplesTrain, yValuesTrain );
  
  if ( verbose )
    std::cerr << "trained both regressionMethods - now start evaluating them" << std::endl;
  
  
  // TEST that both regressionMethods produce equal store-files
   std::string s_destination_save_IL ( "myRegressionMethodIL.txt" );
  
  std::filebuf fbOut;
  fbOut.open ( s_destination_save_IL.c_str(), ios::out );
  std::ostream os (&fbOut);
  //
  regressionMethod->store( os );
  //   
  fbOut.close(); 
  
  std::string s_destination_save_scratch ( "myRegressionMethodScratch.txt" );
  
  std::filebuf fbOutScratch;
  fbOutScratch.open ( s_destination_save_scratch.c_str(), ios::out );
  std::ostream osScratch (&fbOutScratch);
  //
  regressionMethodScratch->store( osScratch );
  //   
  fbOutScratch.close(); 
  
  
  // TEST both regressionMethods to produce equal results
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yValuesTest; 
  
  std::string s_testData = conf.gS( "main", "testData", "toyExampleTest.data" );  
  
  readData ( s_testData, dataTest, yValuesTest );

  
  // ------------------------------------------
  // ------------- REGRESSION --------------
  // ------------------------------------------  


  double holdOutLossIL ( 0.0 );
  double holdOutLossScratch ( 0.0 );
  
  evaluateRegressionMethod ( holdOutLossIL, regressionMethod, dataTest, yValuesTest ); 
  
  evaluateRegressionMethod ( holdOutLossScratch, regressionMethodScratch, dataTest, yValuesTest );  
  
    
  if ( verbose ) 
  {
    std::cerr << "holdOutLossIL: " << holdOutLossIL  << std::endl;
  
    std::cerr << "holdOutLossScratch: " << holdOutLossScratch << std::endl;
  }
  
  
  CPPUNIT_ASSERT_DOUBLES_EQUAL( holdOutLossIL, holdOutLossScratch, 1e-4);
  
  // don't waste memory
  
  delete regressionMethod;
  delete regressionMethodScratch;
  
  for (std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin(); exTrainIt != examplesTrain.end(); exTrainIt++)
  {
    delete *exTrainIt;
  } 

  
  if (verboseStartEnd)
    std::cerr << "================== TestGPHIKRegression::testRegressionOnlineLearning done ===================== " << std::endl;   
}

#endif
