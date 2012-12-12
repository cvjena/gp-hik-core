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

  
  //------------- read the training data --------------
  
  NICE::Matrix dataTrain;
  NICE::Vector yBinTrain;
  NICE::Vector yMultiTrain;  

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
  
  //----------------- convert data to sparse data structures ---------
  std::vector< NICE::SparseVector *> examplesTrain;
  examplesTrain.resize( dataTrain.rows() );
  
  std::vector< NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
  for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
  {
    *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
  }
  
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
  
  //TODO adapt this to the actual number of classes
  NICE::Matrix confusionMatrix(3, 3, 0.0);
  
  NICE::Timer t;
  double testTime (0.0);
  
  for (int i = 0; i < (int)dataTest.rows(); i++)
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
    
    confusionMatrix(result, yMultiTest[i]) += 1.0;
  }
  
  std::cerr << "Time for testing: " << testTime << std::endl;
  
  confusionMatrix.normalizeColumnsL1();
  std::cerr << confusionMatrix << std::endl;

  std::cerr << "average recognition rate: " << confusionMatrix.trace()/confusionMatrix.rows() << std::endl;
  
  
  return 0;
}
