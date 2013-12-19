/** 
* @file toyExample.cpp
* @brief Demo-Program to show how to call some methods of the GPHIKClassifier class
* @author Alexander Freytag
* @date 19-10-2012
*/

#include <iostream>
#include <vector>

#include <core/basics/Config.h>
#include <core/basics/Timer.h>
#include <core/vector/MatrixT.h>
#include <core/vector/VectorT.h>

#include "gp-hik-core/GPHIKClassifier.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

int main (int argc, char* argv[])
{  
  
  Config conf ( argc, argv );
  std::string trainData = conf.gS( "main", "trainData", "progs/toyExampleSmallScaleTrain.data" );
  bool b_debug = conf.gB( "main", "debug", false );

  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain;  

  if ( b_debug )
  { 
    dataTrain.resize(6,3);
    dataTrain.set(0);
    dataTrain(0,0) = 0.2; dataTrain(0,1) = 0.3; dataTrain(0,2) = 0.5;
    dataTrain(1,0) = 0.3; dataTrain(1,1) = 0.2; dataTrain(1,2) = 0.5;    
    dataTrain(2,0) = 0.9; dataTrain(2,1) = 0.0; dataTrain(2,2) = 0.1;
    dataTrain(3,0) = 0.8; dataTrain(3,1) = 0.1; dataTrain(3,2) = 0.1;    
    dataTrain(4,0) = 0.1; dataTrain(4,1) = 0.1; dataTrain(4,2) = 0.8;
    dataTrain(5,0) = 0.1; dataTrain(5,1) = 0.0; dataTrain(5,2) = 0.9;    
    
    yMultiTrain.resize(6);
    yMultiTrain[0] = 1; yMultiTrain[1] = 1;
    yMultiTrain[2] = 2; yMultiTrain[3] = 2;
    yMultiTrain[2] = 3; yMultiTrain[3] = 3;
  }
  else 
  {
    std::ifstream ifsTrain ( trainData.c_str() , ios::in );

    if (ifsTrain.good() )
    {
      ifsTrain >> dataTrain;
      ifsTrain >> yBinTrain;
      ifsTrain >> yMultiTrain;
      ifsTrain.close();  
    }
    else 
    {
      std::cerr << "Unable to read training data, aborting." << std::endl;
      return -1;
    }
  }
  
  //----------------- convert data to sparse data structures ---------
  std::vector< NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }
  
  std::cerr << "Number of training examples: " << examplesTrain.size() << std::endl;
  
  //----------------- train our classifier -------------
  conf.sB("GPHIKClassifier", "verbose", false);
  GPHIKClassifier * classifier  = new GPHIKClassifier ( &conf );  
  classifier->train ( examplesTrain , yMultiTrain );
  
  // ------------------------------------------
  // ------------- CLASSIFICATION --------------
  // ------------------------------------------
  
  
  //------------- read the test data --------------
  
  
  NICE::Matrix dataTest;
  NICE::Vector yBinTest;
  NICE::Vector yMultiTest; 
    
  if ( b_debug )
  { 
    dataTest.resize(1,3);
    dataTest.set(0);
    dataTest(0,0) = 0.3; dataTest(0,1) = 0.4; dataTest(0,2) = 0.3;
    
    yMultiTrain.resize(1);
    yMultiTrain[0] = 1;
  }
  else 
  {  
    std::string testData = conf.gS( "main", "testData", "progs/toyExampleTest.data" );  
    std::ifstream ifsTest ( testData.c_str(), ios::in );
    if (ifsTest.good() )
    {
      ifsTest >> dataTest;
      ifsTest >> yBinTest;
      ifsTest >> yMultiTest;
      ifsTest.close();  
    }
    else 
    {
      std::cerr << "Unable to read test data, aborting." << std::endl;
      return -1;
    }
  }
  
  //TODO adapt this to the actual number of classes
  NICE::Matrix confusionMatrix(3, 3, 0.0);
  
  NICE::Timer t;
  double testTime (0.0);
  
  double uncertainty;
  
  int i_loopEnd  ( (int)dataTest.rows() );
  
  if ( b_debug )
  {
    i_loopEnd = 1;
  }
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    //----------------- convert data to sparse data structures ---------
    NICE::SparseVector * example =  new NICE::SparseVector( dataTest.getRow(i) );
       
    int result;
    NICE::SparseVector scores;
   
    // and classify
    t.start();
    classifier->classify( example, result, scores );
    t.stop();
    testTime += t.getLast();
    
    std::cerr << " scores.size(): " << scores.size() << std::endl;
    scores.store(std::cerr);
    
    if ( b_debug )
    {    
      classifier->predictUncertainty( example, uncertainty );
      std::cerr << " uncertainty: " << uncertainty << std::endl;
    }
    else
    {
      confusionMatrix(result, yMultiTest[i]) += 1.0;
    }
  }
  
  if ( !b_debug )
  {
    std::cerr << "Time for testing: " << testTime << std::endl;
    
    confusionMatrix.normalizeColumnsL1();
    std::cerr << confusionMatrix << std::endl;

    std::cerr << "average recognition rate: " << confusionMatrix.trace()/confusionMatrix.rows() << std::endl;
  }
  
  
  return 0;
}
