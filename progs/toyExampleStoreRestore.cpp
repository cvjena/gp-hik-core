/** 
* @file toyExampleStoreRestore.cpp
* @brief 
* @author Alexander Freytag
* @date 21-12-2013
*/

// STL includes
#include <iostream>
#include <vector>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKClassifier.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

int main (int argc, char* argv[])
{  
  
  NICE::Config conf ( argc, argv );
  std::string trainData = conf.gS( "main", "trainData", "progs/toyExampleSmallScaleTrain.data" );
  NICE::GPHIKClassifier * classifier;  
  
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
  
  // TRAIN CLASSIFIER FROM SCRATCH
  
  classifier = new GPHIKClassifier ( &conf );  
    
  classifier->train ( examplesTrain , yMultiTrain );
  
  
  // TEST STORING ABILITIES
  
  std::string s_destination_save ( "/home/alex/code/nice/gp-hik-core/progs/myClassifier.txt" );
  
  std::filebuf fbOut;
  fbOut.open ( s_destination_save.c_str(), ios::out );
  std::ostream os (&fbOut);
  //
  classifier->store( os );
  //   
  fbOut.close(); 
  
  
  // TEST RESTORING ABILITIES
    
  NICE::GPHIKClassifier * classifierRestored = new GPHIKClassifier;  
      
  std::string s_destination_load ( "/home/alex/code/nice/gp-hik-core/progs/myClassifier.txt" );
  
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
  NICE::Matrix confusionMatrixRestored ( noClassesKnownTraining, noClassesKnownTest, 0.0);
  
  NICE::Timer t;
  double testTime (0.0);
  
  double uncertainty;
  
  int i_loopEnd  ( (int)dataTest.rows() );
  
  
  for (int i = 0; i < i_loopEnd ; i++)
  {
    NICE::Vector example ( dataTest.getRow(i) );
    NICE::SparseVector scores;
    int result;
    
    // classify with trained classifier 
    t.start();
    classifier->classify( &example, result, scores );
    t.stop();
    testTime += t.getLast();
     
    
    confusionMatrix( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;

    // classify with restored classifier 
    t.start();
    classifierRestored->classify( &example, result, scores );
    t.stop();
    testTime += t.getLast();  
    
    confusionMatrixRestored( mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;
    
    
  }  
  
  confusionMatrix.normalizeColumnsL1();
  std::cerr << confusionMatrix << std::endl;

  std::cerr << "average recognition rate: " << confusionMatrix.trace()/confusionMatrix.cols() << std::endl;

  confusionMatrixRestored.normalizeColumnsL1();
  std::cerr << confusionMatrixRestored << std::endl;

  std::cerr << "average recognition rate of restored classifier: " << confusionMatrixRestored.trace()/confusionMatrixRestored.cols() << std::endl;
  
  
  return 0;
}
