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
#include <core/algebra/EigValues.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKRawClassifier.h"
#include "gp-hik-core/GMHIKernelRaw.h"

//
#include "gp-hik-core/quantization/Quantization1DAequiDist0To1.h"
#include "gp-hik-core/quantization/Quantization1DAequiDist0ToMax.h"
#include "gp-hik-core/quantization/QuantizationNDAequiDist0ToMax.h"

using namespace std;
using namespace NICE;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////


void GPHIKRawClassifier::clearSetsOfTablesAandB( )
{

    // delete all LUTs A which are needed when no quantization is activated
    for ( std::map< uint,PrecomputedType >::iterator itA = this->precomputedA.begin();
          itA != this->precomputedA.end();
          itA++
        )
    {
        for ( uint idxDim = 0 ; idxDim < this->num_dimension; idxDim++ )
        {
            if ( (itA->second)[idxDim] != NULL )
                delete [] (itA->second)[idxDim];
        }
        delete [] itA->second;
    }
    this->precomputedA.clear();


    // delete all LUTs B which are needed when no quantization is activated
    for ( std::map< uint,PrecomputedType >::iterator itB = this->precomputedB.begin();
          itB != this->precomputedB.end();
          itB++
        )
    {
        for ( uint idxDim = 0 ; idxDim < this->num_dimension; idxDim++ )
        {
            if ( (itB->second)[idxDim] != NULL )
                delete [] (itB->second)[idxDim];
        }
        delete [] itB->second;
    }
    this->precomputedB.clear();
}

void GPHIKRawClassifier::clearSetsOfTablesT( )
{
    // delete all LUTs used for quantization
    for ( std::map< uint, double * >::iterator itT = this->precomputedT.begin();
          itT != this->precomputedT.end();
          itT++
         )
    {
        delete [] itT->second;
    }
    this->precomputedT.clear();
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
GPHIKRawClassifier::GPHIKRawClassifier( )
{
  this->b_isTrained       = false;
  this->confSection       = "";

  this->nnz_per_dimension = NULL;
  this->num_examples      = 0;
  this->num_dimension     = 0;

  this->solver            = NULL;    
  this->q                 = NULL;
  this->gm                = NULL;



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

  this->b_isTrained       = false;
  this->confSection       = "";

  this->nnz_per_dimension = NULL;
  this->num_examples      = 0;
  this->num_dimension     = 0;

  this->solver            = NULL;    
  this->q                 = NULL;
  this->gm                = NULL;

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
  if ( this->solver != NULL )
  {
    delete this->solver;
    this->solver = NULL;
  }

  if ( this->gm != NULL)
  {
    delete this->gm;
    this->gm = NULL;
  }

  this->clearSetsOfTablesAandB();
  this->clearSetsOfTablesT();

  if ( this->q != NULL )
  {
      delete this->q;
      this->q = NULL;
  }
}

void GPHIKRawClassifier::initFromConfig(const Config *_conf,
                                     const string & _confSection
                                    )
{
  this->d_noise     = _conf->gD( _confSection, "noise", 0.01);

  this->confSection = _confSection;
  this->b_verbose   = _conf->gB( _confSection, "verbose", false);
  this->b_debug     = _conf->gB( _confSection, "debug", false);
  this->f_tolerance = _conf->gD( _confSection, "f_tolerance", 1e-10);

  //FIXME this is not used in that way for the standard GPHIKClassifier
  //string ilssection = "FMKGPHyperparameterOptimization";
  string ilssection       = _confSection;
  uint ils_max_iterations = _conf->gI( ilssection, "ils_max_iterations", 1000 );
  double ils_min_delta    = _conf->gD( ilssection, "ils_min_delta", 1e-7 );
  double ils_min_residual = _conf->gD( ilssection, "ils_min_residual", 1e-7 );
  bool ils_verbose        = _conf->gB( ilssection, "ils_verbose", false );
  this->solver            = new ILSConjugateGradients( ils_verbose,
                                                       ils_max_iterations,
                                                       ils_min_delta,
                                                       ils_min_residual
                                                     );
  if ( this->b_verbose )
  {
      std::cerr << "GPHIKRawClassifier::initFromConfig " <<std::endl;
      std::cerr << "   confSection " << confSection << std::endl;
      std::cerr << "   d_noise " << d_noise << std::endl;
      std::cerr << "   f_tolerance " << f_tolerance << std::endl;
      std::cerr << "   ils_max_iterations " << ils_max_iterations << std::endl;
      std::cerr << "   ils_min_delta " << ils_min_delta << std::endl;
      std::cerr << "   ils_min_residual " << ils_min_residual << std::endl;
      std::cerr << "   ils_verbose " << ils_verbose << std::endl;
  }

  //quantization during classification?
  bool useQuantization = _conf->gB ( _confSection, "use_quantization", false );

  if ( this->b_verbose )
  {
    std::cerr << "_confSection: " << _confSection << std::endl;
    std::cerr << "use_quantization: " << useQuantization << std::endl;
  }

  if ( _conf->gB ( _confSection, "use_quantization", false ) )
  {
    int numBins = _conf->gI ( _confSection, "num_bins", 100 );
    if ( this->b_verbose )
      std::cerr << "FMKGPHyperparameterOptimization: quantization initialized with " << numBins << " bins." << std::endl;


    std::string s_quantType = _conf->gS( _confSection, "s_quantType", "1d-aequi-0-1" );

    if ( s_quantType == "1d-aequi-0-1" )
    {
      this->q = new NICE::Quantization1DAequiDist0To1 ( numBins );
    }
    else if ( s_quantType == "1d-aequi-0-max" )
    {
      this->q = new NICE::Quantization1DAequiDist0ToMax ( numBins );
    }
    else if ( s_quantType == "nd-aequi-0-max" )
    {
      this->q = new NICE::QuantizationNDAequiDist0ToMax ( numBins );
    }
    else
    {
      fthrow(Exception, "Quantization type is unknown " << s_quantType);
    }
  }
  else
  {
    this->q = NULL;
  }
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// /////////////////////

std::set<uint> GPHIKRawClassifier::getKnownClassNumbers ( ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );

  return this->knownClasses;
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


    // classification with quantization of test inputs
    if ( this->q != NULL )
    {
        uint maxClassNo = 0;
        for ( std::map< uint, double * >::const_iterator itT = this->precomputedT.begin() ;
              itT != this->precomputedT.end();
              itT++
            )
        {
          uint classno = itT->first;
          maxClassNo   = std::max ( maxClassNo, classno );
          double beta  = 0;
          double *T    = itT->second;

          for (SparseVector::const_iterator i = _xstar->begin(); i != _xstar->end(); i++ )
          {
            uint dim  = i->first;
            double v  = i->second;
            uint qBin = this->q->quantize( v, dim );

            beta += T[dim * this->q->getNumberOfBins() + qBin];
          }//for-loop over dimensions of test input

          _scores[ classno ] = beta;

        }//for-loop over 1-vs-all models
    }
    // classification with exact test inputs, i.e., no quantization involved
    else
    {
        uint maxClassNo = 0;
        for ( std::map<uint, PrecomputedType>::const_iterator i = this->precomputedA.begin() ; i != this->precomputedA.end(); i++ )
        {
          uint classno = i->first;
          maxClassNo = std::max ( maxClassNo, classno );
          double beta = 0;
          GMHIKernelRaw::sparseVectorElement **dataMatrix = this->gm->getDataMatrix();

          const PrecomputedType & A = i->second;
          std::map<uint, PrecomputedType>::const_iterator j = this->precomputedB.find ( classno );
          const PrecomputedType & B = j->second;

          for (SparseVector::const_iterator i = _xstar->begin(); i != _xstar->end(); i++)
          {
            uint dim = i->first;
            double fval = i->second;

            uint nnz = this->nnz_per_dimension[dim];
            uint nz = this->num_examples - nnz;

            if ( nnz == 0 ) continue;
            // useful
            //if ( fval < this->f_tolerance ) continue;

            uint position = 0;

            //this->X_sorted.findFirstLargerInDimension(dim, fval, position);
            GMHIKernelRaw::sparseVectorElement fval_element;
            fval_element.value = fval;

            //std::cerr << "value to search for " << fval << endl;
            //std::cerr << "data matrix in dimension " << dim << endl;
            //for (int j = 0; j < nnz; j++)
            //    std::cerr << dataMatrix[dim][j].value << std::endl;

            GMHIKernelRaw::sparseVectorElement *it = upper_bound ( dataMatrix[dim], dataMatrix[dim] + nnz, fval_element );
            position = distance ( dataMatrix[dim], it );
            
//             /*// add zero elements
//             if ( fval_element.value > 0.0 )
//                 position += nz;*/


            bool posIsZero ( position == 0 );
            
            // special case 1:
            // new example is smaller than all known examples
            // -> resulting value = fval * sum_l=1^n alpha_l               
            if ( position == 0 )
            {
              beta += fval * B[ dim ][ nnz - 1 ];  
            }
            // special case 2:
            // new example is equal to or larger than the largest training example in this dimension
            // -> the term B[ dim ][ nnz-1 ] - B[ dim ][ indexElem ] is equal to zero and vanishes, which is logical, since all elements are smaller than the remaining prototypes!            
            else if ( position == nnz )
            {
              beta += A[ dim ][ nnz - 1 ];
            }
            // standard case: new example is larger then the smallest element, but smaller then the largest one in the corrent dimension        
            else
            {
              beta += A[ dim ][ position - 1 ] + fval * B[ dim ][ position - 1 ];
            }
            
//             // correct upper bound to correct position, only possible if new example is not the smallest value in this dimension
//             if ( !posIsZero )
//                 position--;
// 
// 
//             double firstPart = 0.0;
//             if ( !posIsZero  )
//               firstPart = ( A[ dim ][ position ] );
// 
//             double secondPart( B[ dim ][ this->num_examples-1-nz ]);
//             if ( !posIsZero && (position >= nz) )
//                 secondPart -= B[dim][ position ];
// 
//             // but apply using the transformed one
//             beta += firstPart + secondPart* fval;
          }//for-loop over dimensions of test input

          _scores[ classno ] = beta;

        }//for-loop over 1-vs-all models

    } // if-condition wrt quantization
  _scores.setDim ( *this->knownClasses.rbegin() + 1 );


  if ( this->knownClasses.size() > 2 )
  { // multi-class classification
    _result = _scores.maxElement();
  }
  else if ( this->knownClasses.size() == 2 ) // binary setting
  {
    uint class1 = *(this->knownClasses.begin());
    uint class2 = *(this->knownClasses.rbegin());
    uint class_for_which_we_have_a_score = _scores.begin()->first;
    uint class_for_which_we_dont_have_a_score = (class1 == class_for_which_we_have_a_score ? class2 : class1);

    _scores[class_for_which_we_dont_have_a_score] = - _scores[class_for_which_we_have_a_score];

    _result = _scores[class_for_which_we_have_a_score] > 0.0 ? class_for_which_we_have_a_score : class_for_which_we_dont_have_a_score;
  }

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

  this->knownClasses.clear();
  for ( uint i = 0; i < _labels.size(); i++ )
    this->knownClasses.insert((uint)_labels[i]);

  std::map<uint, NICE::Vector> binLabels;
  for ( set<uint>::const_iterator j = knownClasses.begin(); j != knownClasses.end(); j++ )
  {
    uint current_class = *j;
    Vector labels_binary ( _labels.size() );
    for ( uint i = 0; i < _labels.size(); i++ )
    {
        labels_binary[i] = ( _labels[i] == current_class ) ? 1.0 : -1.0;
    }

    binLabels.insert ( std::pair<uint, NICE::Vector>( current_class, labels_binary) );
  }

  // handle special binary case
  if ( knownClasses.size() == 2 )
  {
    std::map<uint, NICE::Vector>::iterator it = binLabels.begin();
    it++;
    binLabels.erase( binLabels.begin(), it );
  }

  this->train ( _examples, binLabels );
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

  this->clearSetsOfTablesAandB();
  this->clearSetsOfTablesT();


  // sort examples in each dimension and "transpose" the feature matrix
  // set up the GenericMatrix interface
  if ( this->gm != NULL )
    delete this->gm;

  this->gm = new GMHIKernelRaw ( _examples, this->d_noise, this->q );
  this->nnz_per_dimension = this->gm->getNNZPerDimension();
  this->num_dimension     = this->gm->getNumberOfDimensions();


  // compute largest eigenvalue of our kernel matrix
  // note: this guy is shared among all categories,
  //       since the kernel matrix is shared as well
  NICE::Vector eigenMax;
  NICE::Matrix eigenMaxV;
  // for reproducibility during debuggin
  //FIXME
  srand ( 0 );
  srand48 ( 0 );
  NICE::EigValues * eig = new EVArnoldi ( false /* verbose flag */,
                                          10 /*_maxiterations*/
                                        );
  eig->getEigenvalues( *gm, eigenMax, eigenMaxV, 1 /*rank*/ );
  delete eig;

  // set simple jacobi pre-conditioning
  NICE::Vector diagonalElements;
  this->gm->getDiagonalElements ( diagonalElements );
  this->solver->setJacobiPreconditioner ( diagonalElements );

  // solve linear equations for each class
  // be careful when parallising this!
  for ( std::map<uint, NICE::Vector>::const_iterator i = _binLabels.begin();
        i != _binLabels.end();
        i++
      )
  {
    uint classno = i->first;
    if (b_verbose)
        std::cerr << "Training for class " << classno << endl;
    const NICE::Vector & y = i->second;
    NICE::Vector alpha;


  /** About finding a good initial solution (see also GPLikelihoodApproximation)
    * K~ = K + sigma^2 I
    *
    * K~ \approx lambda_max v v^T
    * \lambda_max v v^T * alpha = k_*     | multiply with v^T from left
    * => \lambda_max v^T alpha = v^T k_*
    * => alpha = k_* / lambda_max could be a good initial start
    * If we put everything in the first equation this gives us
    * v = k_*
    *  This reduces the number of iterations by 5 or 8
    */
    alpha = (y * (1.0 / eigenMax[0]) );

    this->solver->solveLin( *gm, y, alpha );

    // get lookup tables, A, B, etc. and store them
    this->gm->updateTablesAandB( alpha );
    double **A = this->gm->getTableA();
    double **B = this->gm->getTableB();

    this->precomputedA.insert ( std::pair<uint, PrecomputedType> ( classno, A ) );
    this->precomputedB.insert ( std::pair<uint, PrecomputedType> ( classno, B ) );

    // Quantization for classification?
    if ( this->q != NULL )
    {
      this->gm->updateTableT( alpha );
      double *T = this->gm->getTableT ( );
      this->precomputedT.insert( std::pair<uint, double * > ( classno, T ) );

    }
  }

  // NOTE if quantization is turned on, we do not need LUTs A and B anymore
  if ( this->q != NULL )
  {
    this->clearSetsOfTablesAandB();
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
