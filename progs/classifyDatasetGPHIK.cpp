/** 
* @file classifyDatasetGPHIK.cpp
* @brief Perform classification on an arbitrary dataset with the GPHIK-Classifier
* @author Alexander Freytag
* @date 16-09-2013
*/

// STL-includes
#include <iostream>
#include <vector>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Exception.h>
#include <core/vector/MatrixT.h>
#include <core/vector/SparseVectorT.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKClassifier.h"


void readSparseExamples ( const std::string & fn,  std::vector< const NICE::SparseVector * > & examples, NICE::Vector & labels )
{
  // initially cleaning of variables
  examples.clear();
  labels.clear();
  
  std::vector<double> labels_std;
  labels_std.clear();
  
  std::cerr << "Reading " << fn << std::endl;
  std::ifstream ifs ( fn.c_str(), std::ios::in );
  if ( ! ifs.good() )
  {
      std::cerr <<  "Unable to read " << fn << std::endl;
      return;
  }
  
  // read until no new line is in the file anymore
  while ( !ifs.eof() )
  {
    int classno;
    if ( !(ifs >> classno) )
      break;
    
    labels_std.push_back( classno );
    
    NICE::SparseVector *v = new NICE::SparseVector; 
    /* needed format in every line: 
     * SVECTOR dimension size index value index value ... END
     * with 
     *      SVECTOR   -- starting flag
     *      dimension -- overall feature dimension and
     *      size      -- number of non-zero entries for the current feature vector 
     *      index     -- integer value specifying a non-zero dimension
     *      value     -- double value specifying the value for the corresp. non-zero dimension
     *      END       -- ending flag
     */
    try
    {
      v->restore ( ifs, NICE::SparseVector::FORMAT_INDEX );    
    }
    catch ( NICE::Exception excep)
    {
      std::cerr << "Error while reading features. Error message: " << excep.what() << std::endl;
      break;
    }
          
    
    examples.push_back ( v );
  }
  ifs.close();
  
  labels = NICE::Vector( labels_std );
}

void mapClassNumbersToIndices( const NICE::Vector & labels, std::map<int,int> & mapClassNoToIdx )
{
  mapClassNoToIdx.clear();
  int classCnt ( 0 );
  
  for ( NICE::Vector::const_iterator it_labels = labels.begin(); it_labels != labels.end(); it_labels++ )
  {
    if ( mapClassNoToIdx.find( *it_labels ) == mapClassNoToIdx.end() )
    {
      mapClassNoToIdx.insert( std::pair<int,int>( (int) round(*it_labels), classCnt ) );
      classCnt++;
    }
  }
}


int main (int argc, char* argv[])
{  
  std::set_terminate(__gnu_cxx::__verbose_terminate_handler);

  NICE::Config conf ( argc, argv );
 
  NICE::GPHIKClassifier classifier ( &conf, "GPHIKClassifier" );  
  
  // ========================================================================
  //                            TRAINING STEP
  // ========================================================================  
   
  // read training data
  std::vector< const NICE::SparseVector * > examplesTrain;
  NICE::Vector labelsTrain;
  
  std::string s_fn_trainingSet = conf.gS("main", "trainset");
  readSparseExamples ( s_fn_trainingSet, examplesTrain, labelsTrain );

  //map the occuring classes to a minimal set of indices
  std::map<int,int> map_classNoToClassIdx_train; // < classNo, Idx>
  
  mapClassNumbersToIndices( labelsTrain, map_classNoToClassIdx_train );
  
  //how many different classes do we have in the training set?
  int i_noClassesTrain ( map_classNoToClassIdx_train.size() );

  // train GPHIK classifier
  classifier.train ( examplesTrain, labelsTrain );
  
  // ========================================================================
  //                            TEST STEP
  // ========================================================================
  
  // read test data
  std::vector< const NICE::SparseVector * > examplesTest;
  NICE::Vector labelsTest;
  
  std::string s_fn_testSet = conf.gS("main", "testset");
  readSparseExamples ( s_fn_testSet, examplesTest, labelsTest );
  
  //map the occuring classes to a minimal set of indices
  std::map<int,int> map_classNoToClassIdx_test; // < classNo, Idx>
  
  mapClassNumbersToIndices( labelsTest, map_classNoToClassIdx_test );

  //how many different classes do we have in the test set?  
  int i_noClassesTest ( map_classNoToClassIdx_test.size() );
  
  // evaluate GPHIK classifier on unseen test data
  int idx ( 0 );
  NICE::SparseVector scores;  /* not needed in this evaluation, so we just declare it ones */
  
  NICE::Matrix confusion ( i_noClassesTest, i_noClassesTrain, 0.0 );
  
  for (std::vector< const NICE::SparseVector *>::const_iterator itTestExamples = examplesTest.begin(); itTestExamples != examplesTest.end(); itTestExamples++, idx++)
  {
    int classno_groundtruth = labelsTest( idx );
    int classno_predicted;

    classifier.classify ( *itTestExamples, classno_predicted, scores /* not needed anyway in that evaluation*/ );
    
    
    int idx_classno_groundtruth ( map_classNoToClassIdx_test[ classno_groundtruth ] );
    int idx_classno_predicted ( map_classNoToClassIdx_train[ classno_predicted ] );
        
    confusion( idx_classno_groundtruth, idx_classno_predicted ) += 1;
  }


  confusion.normalizeRowsL1();
  std::cerr << confusion << std::endl;

  std::cerr << "average recognition rate: " << confusion.trace()/confusion.rows() << std::endl;
  
  
  // ========================================================================
  //                           clean up memory
  // ========================================================================
  
  // release memore of feature vectors from training set
  for (std::vector< const NICE::SparseVector *>::const_iterator itTrainExamples = examplesTrain.begin(); itTrainExamples != examplesTrain.end(); itTrainExamples++ )
  {
    delete *itTrainExamples;
  }
  
  // release memore of feature vectors from test set
  for (std::vector< const NICE::SparseVector *>::const_iterator itTestExamples = examplesTest.begin(); itTestExamples != examplesTest.end(); itTestExamples++ )
  {
    delete *itTestExamples;
  }
  return 0;
}