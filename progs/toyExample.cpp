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
    yMultiTrain[4] = 3; yMultiTrain[5] = 3;
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
//   conf.sB("GPHIKClassifier", "verbose", false);
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
    
    yMultiTest.resize(1);
    yMultiTest[0] = 1;
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
          
  
  NICE::Matrix confusionMatrix( noClassesKnownTraining, noClassesKnownTest, 0.0);
  
  NICE::Timer t;
  double testTime (0.0);
  
  double uncertainty;
  
  int i_loopEnd  ( (int)dataTest.rows() );
  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( dataTest.getRow(i) );
    NICE::SparseVector scores;
    int result;
    
    // and classify
    t.start();
    classifier->classify( &example, result, scores );
    t.stop();
    testTime += t.getLast();
    
    std::cerr << " scores.size(): " << scores.size() << std::endl;
    scores.store(std::cerr);
    
    if ( b_debug )
    {    
      classifier->predictUncertainty( &example, uncertainty );
      std::cerr << " uncertainty: " << uncertainty << std::endl;
    }
    
    confusionMatrix( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;
  }
  

  std::cerr << "Time for testing: " << testTime << std::endl;
  
  confusionMatrix.normalizeColumnsL1();
  std::cerr << confusionMatrix << std::endl;

  std::cerr << "average recognition rate: " << confusionMatrix.trace()/confusionMatrix.cols() << std::endl;
  
  
  return 0;
}
