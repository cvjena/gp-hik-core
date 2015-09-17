/**
* @file GPHIKRawClassifier.cpp
* @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning) (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/01/2012

*/

// STL includes
#include <iostream>

// NICE-core includes
#include <core/basics/numerictools.h>
#include <core/basics/Timer.h>

#include <core/algebra/ILSConjugateGradients.h>

// gp-hik-core includes
#include "GPHIKRawClassifier.h"
#include "GMHIKernelRaw.h"

using namespace std;
using namespace NICE;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////



/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
GPHIKRawClassifier::GPHIKRawClassifier( )
{
  this->b_isTrained = false;
  this->confSection = "";

  // in order to be sure about all necessary variables be setup with default values, we
  // run initFromConfig with an empty config
  NICE::Config tmpConfEmpty ;
  this->initFromConfig ( &tmpConfEmpty, this->confSection );

}

GPHIKRawClassifier::GPHIKRawClassifier( const Config *_conf,
                                  const string & _confSection
                                )
{
  ///////////
  // same code as in empty constructor - duplication can be avoided with C++11 allowing for constructor delegation
  ///////////

  this->b_isTrained = false;
  this->confSection = "";

  ///////////
  // here comes the new code part different from the empty constructor
  ///////////

  this->confSection = _confSection;

  // if no config file was given, we either restore the classifier from an external file, or run ::init with
  // an emtpy config (using default values thereby) when calling the train-method
  if ( _conf != NULL )
  {
    this->initFromConfig( _conf, _confSection );
  }
  else
  {
    // if no config was given, we create an empty one
    NICE::Config tmpConfEmpty ;
    this->initFromConfig ( &tmpConfEmpty, this->confSection );
  }
}

GPHIKRawClassifier::~GPHIKRawClassifier()
{
  delete solver;
}

void GPHIKRawClassifier::initFromConfig(const Config *_conf,
                                     const string & _confSection
                                    )
{
  this->d_noise     = _conf->gD( _confSection, "noise", 0.01);

  this->confSection = _confSection;
  this->b_verbose   = _conf->gB( _confSection, "verbose", false);
  this->b_debug     = _conf->gB( _confSection, "debug", false);

  string ilssection = "FMKGPHyperparameterOptimization";
  uint ils_max_iterations = _conf->gI( ilssection, "ils_max_iterations", 1000 );
  double ils_min_delta = _conf->gD( ilssection, "ils_min_delta", 1e-7 );
  double ils_min_residual = _conf->gD( ilssection, "ils_min_residual", 1e-7 );
  bool ils_verbose = _conf->gB( ilssection, "ils_verbose", false );
  this->solver = new ILSConjugateGradients( ils_verbose, ils_max_iterations, ils_min_delta, ils_min_residual );
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// /////////////////////

std::set<uint> GPHIKRawClassifier::getKnownClassNumbers ( ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );

  fthrow(Exception, "GPHIKRawClassifier::getKnownClassNumbers() not yet implemented");
}


///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

void GPHIKRawClassifier::classify ( const SparseVector * _example,
                                 uint & _result,
                                 SparseVector & _scores
                               ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );

  _scores.clear();

  if ( this->b_debug )
  {
    std::cerr << "GPHIKRawClassifier::classify (sparse)" << std::endl;
    _example->store( std::cerr );
  }

  // MAGIC happens here....


  // ...
  if ( this->b_debug )
  {
    _scores.store ( std::cerr );
    std::cerr << "_result: " << _result << std::endl;
  }

  if ( _scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << _example->size() );
  }
}

void GPHIKRawClassifier::classify ( const NICE::Vector * _example,
                                 uint & _result,
                                 SparseVector & _scores
                               ) const
{
    fthrow(Exception, "GPHIKRawClassifier::classify( Vector ... ) not yet implemented");
}


/** training process */
void GPHIKRawClassifier::train ( const std::vector< const NICE::SparseVector *> & _examples,
                              const NICE::Vector & _labels
                            )
{
  // security-check: examples and labels have to be of same size
  if ( _examples.size() != _labels.size() )
  {
    fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );
  }

  set<uint> classes;
  for ( uint i = 0; i < _labels.size(); i++ )
    classes.insert((uint)_labels[i]);

  std::map<uint, NICE::Vector> binLabels;
  for ( set<uint>::const_iterator j = classes.begin(); j != classes.end(); j++ )
  {
    uint current_class = *j;
    Vector labels_binary ( _labels.size() );
    for ( uint i = 0; i < _labels.size(); i++ )
        labels_binary[i] = ( _labels[i] == current_class ) ? 1.0 : -1.0;

    binLabels.insert ( pair<uint, NICE::Vector>( current_class, labels_binary) );
  }


  train ( _examples, binLabels );
}

void GPHIKRawClassifier::train ( const std::vector< const NICE::SparseVector *> & _examples,
                              std::map<uint, NICE::Vector> & _binLabels
                            )
{
  // security-check: examples and labels have to be of same size
  for ( std::map< uint, NICE::Vector >::const_iterator binLabIt = _binLabels.begin();
        binLabIt != _binLabels.end();
        binLabIt++
      )
  {
    if ( _examples.size() != binLabIt->second.size() )
    {
      fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );
    }
  }

  if ( this->b_verbose )
    std::cerr << "GPHIKRawClassifier::train" << std::endl;

  Timer t;
  t.start();

  // sort examples in each dimension and "transpose" the feature matrix
  // set up the GenericMatrix interface
  GMHIKernelRaw gm ( _examples, this->d_noise );

  // solve linear equations for each class
  for ( map<uint, NICE::Vector>::const_iterator i = _binLabels.begin();
          i != _binLabels.end(); i++ )
  {
    const Vector & y = i->second;
    Vector alpha;
    solver->solveLin( gm, y, alpha );
    // TODO: get lookup tables, A, B, etc. and store them
  }


  t.stop();
  if ( this->b_verbose )
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;


  //indicate that we finished training successfully
  this->b_isTrained = true;

  // clean up all examples ??
  if ( this->b_verbose )
    std::cerr << "Learning finished" << std::endl;
}


