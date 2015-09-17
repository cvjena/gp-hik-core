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
  this->nnz_per_dimension = NULL;

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
  this->q = NULL;

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



void GPHIKRawClassifier::classify ( const NICE::SparseVector * _xstar,
                                 uint & _result,
                                 SparseVector & _scores
                               ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );
  _scores.clear();

  GMHIKernelRaw::sparseVectorElement **dataMatrix = gm->getDataMatrix();

  uint maxClassNo = 0;
  for ( std::map<uint, PrecomputedType>::const_iterator i = this->precomputedA.begin() ; i != this->precomputedA.end(); i++ )
  {
    uint classno = i->first;
    maxClassNo = std::max ( maxClassNo, classno );
    double beta;

    if ( this->q != NULL ) {
      std::map<uint, double *>::const_iterator j = this->precomputedT.find ( classno );
      double *T = j->second;
      for (SparseVector::const_iterator i = _xstar->begin(); i != _xstar->end(); i++ )
      {
        uint dim = i->first;
        double v = i->second;
        uint qBin = q->quantize( v, dim );

        beta += T[dim * q->getNumberOfBins() + qBin];
      }
    } else {
      const PrecomputedType & A = i->second;
      std::map<uint, PrecomputedType>::const_iterator j = this->precomputedB.find ( classno );
      const PrecomputedType & B = j->second;

      beta = 0.0;
      for (SparseVector::const_iterator i = _xstar->begin(); i != _xstar->end(); i++)
      {
        uint dim = i->first;
        double fval = i->second;

        uint nnz = this->nnz_per_dimension[dim];
        uint nz = this->num_examples - nnz;

        if ( nnz == 0 ) continue;
        if ( fval < this->f_tolerance ) continue;

        uint position = 0;

        //this->X_sorted.findFirstLargerInDimension(dim, fval, position);
        GMHIKernelRaw::sparseVectorElement fval_element;
        fval_element.value = fval;
        GMHIKernelRaw::sparseVectorElement *it = upper_bound ( dataMatrix[dim], dataMatrix[dim] + nnz, fval_element );
        position = distance ( dataMatrix[dim], it );




        bool posIsZero ( position == 0 );
        if ( !posIsZero )
            position--;


        double firstPart = 0.0;
        if ( !posIsZero && ((position-nz) < this->num_examples) )
          firstPart = (A[dim][position-nz]);

        double secondPart( B[dim][this->num_examples-1-nz]);
        if ( !posIsZero && (position >= nz) )
            secondPart -= B[dim][position-nz];

        // but apply using the transformed one
        beta += firstPart + secondPart* fval;
      }
    }

    _scores[ classno ] = beta;
  }
  _scores.setDim ( maxClassNo + 1 );
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
  this->num_examples = _examples.size();

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

  // handle special binary case
  if ( classes.size() == 2 )
  {
    std::map<uint, NICE::Vector>::iterator it = binLabels.begin();
    it++;
    binLabels.erase( binLabels.begin(), it );
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

  precomputedA.clear();
  precomputedB.clear();
  precomputedT.clear();

  // sort examples in each dimension and "transpose" the feature matrix
  // set up the GenericMatrix interface
  gm = new GMHIKernelRaw ( _examples, this->d_noise );
  nnz_per_dimension = gm->getNNZPerDimension();

  // solve linear equations for each class
  // be careful when parallising this!
  for ( map<uint, NICE::Vector>::const_iterator i = _binLabels.begin();
          i != _binLabels.end(); i++ )
  {
    uint classno = i->first;
    const Vector & y = i->second;
    Vector alpha;
    solver->solveLin( *gm, y, alpha );
    // TODO: get lookup tables, A, B, etc. and store them
    precomputedA.insert ( pair<uint, PrecomputedType> ( classno, gm->getTableA() ) );
    precomputedB.insert ( pair<uint, PrecomputedType> ( classno, gm->getTableB() ) );
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


