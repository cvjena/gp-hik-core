/**
* @file FMKGPHyperparameterOptimization.cpp
* @brief Heart of the framework to set up everything, perform optimization, classification, and variance prediction (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/02/2012
*/

// STL includes
#include <iostream>
#include <map>

// NICE-core includes
#include <core/algebra/ILSConjugateGradients.h>
#include <core/algebra/ILSConjugateGradientsLanczos.h>
#include <core/algebra/ILSSymmLqLanczos.h>
#include <core/algebra/ILSMinResLanczos.h>
#include <core/algebra/ILSPlainGradient.h>
#include <core/algebra/EigValuesTRLAN.h>
#include <core/algebra/CholeskyRobust.h>
// 
#include <core/basics/Timer.h>
#include <core/basics/ResourceStatistics.h>
#include <core/basics/Exception.h>
// 
#include <core/vector/Algorithms.h>
// 
#include <core/optimization/blackbox/DownhillSimplexOptimizer.h>


// gp-hik-core includes
#include "gp-hik-core/FMKGPHyperparameterOptimization.h"
#include "gp-hik-core/FastMinKernel.h"
#include "gp-hik-core/GMHIKernel.h"
#include "gp-hik-core/IKMNoise.h"
// 
#include "gp-hik-core/parameterizedFunctions/PFIdentity.h"
#include "gp-hik-core/parameterizedFunctions/PFAbsExp.h"
#include "gp-hik-core/parameterizedFunctions/PFExp.h"
#include "gp-hik-core/parameterizedFunctions/PFMKL.h"
#include "gp-hik-core/parameterizedFunctions/PFWeightedDim.h"
// 
#include "gp-hik-core/quantization/Quantization1DAequiDist0To1.h"
#include "gp-hik-core/quantization/Quantization1DAequiDist0ToMax.h"
#include "gp-hik-core/quantization/QuantizationNDAequiDist0ToMax.h"



using namespace NICE;
using namespace std;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

void FMKGPHyperparameterOptimization::updateAfterIncrement ( 
      const std::set < uint > newClasses,
      const bool & performOptimizationAfterIncrement )
{
  if ( this->fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  std::map<uint, NICE::Vector> binaryLabels;
  std::set<uint> classesToUse;
  //TODO this could be made faster when storing the previous binary label vectors...
  
  if ( this->b_performRegression )
  {
    // for regression, we are not interested in regression scores, rather than in any "label" 
    uint regressionLabel ( 1 );  
    binaryLabels.insert ( std::pair< uint, NICE::Vector> ( regressionLabel, this->labels ) );
  }
  else
    this->prepareBinaryLabels ( binaryLabels, this->labels , classesToUse );
  
  if ( this->b_verbose )
    std::cerr << "labels.size() after increment: " << this->labels.size() << std::endl;

  NICE::Timer t1;

  NICE::GPLikelihoodApprox * gplike;
  uint parameterVectorSize;

  t1.start();
  this->setupGPLikelihoodApprox ( gplike, binaryLabels, parameterVectorSize );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  t1.start();
  if ( this->b_usePreviousAlphas && ( this->previousAlphas.size() > 0) )
  {
    //We initialize it with the same values as we use in GPLikelihoodApprox in batch training
    //default in GPLikelihoodApprox for the first time:
    // alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );
    double factor ( 1.0 / this->eigenMax[0] );
    
    // if we came from an OCC setting and are going to a binary setting,
    // we have to be aware that the 'positive' label is always the one associated with the previous alpha
    // otherwise, we would get into trouble when going to more classes...
    // note that this is needed, since knownClasses is a map, so we loose the order of insertion
    if ( ( this->previousAlphas.size () == 1 ) && ( this->knownClasses.size () == 2 ) )
    {
      // if the first class has a larger value then the currently added second class, we have to 
      // switch the index, which unfortunately is not sooo easy in the map
      if ( this->previousAlphas.begin()->first == this->i_binaryLabelNegative )
      {
        this->previousAlphas.insert( std::pair<int, NICE::Vector> ( this->i_binaryLabelPositive, this->previousAlphas.begin()->second) );
        this->previousAlphas.erase( this->i_binaryLabelNegative );
      }
    }
    
    
    std::map<uint, NICE::Vector>::const_iterator binaryLabelsIt = binaryLabels.begin();
    
    for ( std::map<uint, NICE::Vector>::iterator prevAlphaIt = this->previousAlphas.begin();
         prevAlphaIt != this->previousAlphas.end();
         prevAlphaIt++
        )
    {
      int oldSize ( prevAlphaIt->second.size() );
      prevAlphaIt->second.resize ( oldSize + 1 );

  

      if ( binaryLabelsIt->second[oldSize] > 0 ) //we only have +1 and -1, so this might be benefitial in terms of speed
        prevAlphaIt->second[oldSize] = factor;
      else
        prevAlphaIt->second[oldSize] = -factor; //we follow the initialization as done in previous steps
        //prevAlphaIt->second[oldSize] = 0.0; // following the suggestion of Yeh and Darrell

      binaryLabelsIt++;
      
    }

    //compute unaffected alpha-vectors for the new classes
    for (std::set<uint>::const_iterator newClIt =  newClasses.begin(); newClIt != newClasses.end(); newClIt++)
    {      
      NICE::Vector alphaVec = (binaryLabels[*newClIt] * factor ); //see GPLikelihoodApprox for an explanation
      previousAlphas.insert( std::pair<uint, NICE::Vector>(*newClIt, alphaVec) );
    }      

    gplike->setInitialAlphaGuess ( &previousAlphas );
  }
  else
  {
    //if we do not use previous alphas, we do not have to set up anything here
    gplike->setInitialAlphaGuess ( NULL );
  }
    
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the alpha-objects: " << t1.getLast() << std::endl;

  if ( this->b_verbose ) 
    std::cerr << "update Eigendecomposition " << std::endl;
  
  t1.start();
  // we compute all needed eigenvectors for standard classification and variance prediction at ones.
  // nrOfEigenvaluesToConsiderForVarApprox should NOT be larger than 1 if a method different than approximate_fine is used!
  this->updateEigenDecomposition(  std::max ( this->nrOfEigenvaluesToConsider, this->nrOfEigenvaluesToConsiderForVarApprox) );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  
  //////////////////////  //////////////////////
  //   RE-RUN THE OPTIMIZATION, IF DESIRED    //
  //////////////////////  //////////////////////    
    
  if ( this->b_verbose )
    std::cerr << "resulting eigenvalues for first class: " << eigenMax[0] << std::endl;
  
  // we can reuse the already given performOptimization-method:
  // OPT_GREEDY
  // for this strategy we can't reuse any of the previously computed scores
  // so come on, let's do the whole thing again...
  // OPT_DOWNHILLSIMPLEX
  // Here we can benefit from previous results, when we use them as initialization for our optimizer
  // ikmsums.begin()->second->getParameters ( currentParameters ); uses the previously computed optimal parameters
  // as initialization
  // OPT_NONE
  // nothing to do, obviously
    
  if ( this->b_verbose )
    std::cerr << "perform optimization after increment " << std::endl;
   
  OPTIMIZATIONTECHNIQUE optimizationMethodTmpCopy;
  if ( !performOptimizationAfterIncrement )
  {
    // if no optimization shall be carried out, we simply set the optimization method to NONE but run the optimization
    // call nonetheless, thereby computing alpha vectors, etc. which would be not initialized 
    optimizationMethodTmpCopy = this->optimizationMethod;
    this->optimizationMethod = OPT_NONE;
  }
  
  t1.start();
  this->performOptimization ( *gplike, parameterVectorSize);

  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

  if ( this->b_verbose )
    std::cerr << "Preparing after retraining for classification ..." << std::endl;

  t1.start();
  this->transformFeaturesWithOptimalParameters ( *gplike, parameterVectorSize );
  t1.stop();
  if ( this->b_verboseTime)
    std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;

  if ( !performOptimizationAfterIncrement )
  {
    this->optimizationMethod = optimizationMethodTmpCopy;
  }

  
  //NOTE unfortunately, the whole vector alpha differs, and not only its last entry.
  // If we knew any method, which could update this efficiently, we could also compute A and B more efficiently by updating them.
  // Since we are not aware of any such method, we have to compute them completely new
  // :/
  t1.start();
  this->computeMatricesAndLUTs ( *gplike );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the A'nB -objects: " << t1.getLast() << std::endl;

  //don't waste memory
  delete gplike;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization( )
{
  // initialize pointer variables
  this->pf = NULL;
  this->eig = NULL;
  this->linsolver = NULL;
  this->fmk = NULL;
  this->q = NULL;
  this->precomputedTForVarEst = NULL;
  this->ikmsum  = NULL;
  
  // initialize boolean flags
  this->b_verbose = false;
  this->b_verboseTime = false;
  this->b_debug = false;
  
  //stupid unneeded default values
  this->i_binaryLabelPositive = 0;
  this->i_binaryLabelNegative = 1;
  this->knownClasses.clear();  
  
  this->b_usePreviousAlphas = false;
  this->b_performRegression = false;
}

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization( const bool & _performRegression ) 
{
  ///////////
  // same code as in empty constructor - duplication can be avoided with C++11 allowing for constructor delegation
  ///////////
  
  // initialize pointer variables
  this->pf = NULL;
  this->eig = NULL;
  this->linsolver = NULL;
  this->fmk = NULL;
  this->q = NULL;
  this->precomputedTForVarEst = NULL;
  this->ikmsum  = NULL;
  
  // initialize boolean flags
  this->b_verbose = false;
  this->b_verboseTime = false;
  this->b_debug = false;
  
  //stupid unneeded default values
  this->i_binaryLabelPositive = 0;
  this->i_binaryLabelNegative = 1;
  this->knownClasses.clear();   
  
  this->b_usePreviousAlphas = false;
  this->b_performRegression = false;  
  
  ///////////
  // here comes the new code part different from the empty constructor
  ///////////  
  this->b_performRegression = _performRegression;
}

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization ( const Config *_conf, 
                                                                   const string & _confSection 
                                                                 )
{
  ///////////
  // same code as in empty constructor - duplication can be avoided with C++11 allowing for constructor delegation
  ///////////
  
  // initialize pointer variables
  this->pf = NULL;
  this->eig = NULL;
  this->linsolver = NULL;
  this->fmk = NULL;
  this->q = NULL;
  this->precomputedTForVarEst = NULL;
  this->ikmsum  = NULL;
  
  // initialize boolean flags
  this->b_verbose = false;
  this->b_verboseTime = false;
  this->b_debug = false;
  
  //stupid unneeded default values
  this->i_binaryLabelPositive = 0;
  this->i_binaryLabelNegative = 1;
  this->knownClasses.clear();  
  
  this->b_usePreviousAlphas = false;
  this->b_performRegression = false;  
  
  ///////////
  // here comes the new code part different from the empty constructor
  ///////////  
  this->initFromConfig ( _conf, _confSection );
}

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization ( const Config *_conf, 
                                                                   FastMinKernel *_fmk, 
                                                                   const string & _confSection 
                                                                 )
{
  ///////////
  // same code as in empty constructor - duplication can be avoided with C++11 allowing for constructor delegation
  ///////////
  
  // initialize pointer variables
  this->pf = NULL;
  this->eig = NULL;
  this->linsolver = NULL;
  this->fmk = NULL;
  this->q = NULL;
  this->precomputedTForVarEst = NULL;
  this->ikmsum  = NULL;
  
  // initialize boolean flags
  this->b_verbose = false;
  this->b_verboseTime = false;
  this->b_debug = false;
  
  //stupid unneeded default values
  this->i_binaryLabelPositive = 0;
  this->i_binaryLabelNegative = 1;
  this->knownClasses.clear();    
  
  this->b_usePreviousAlphas = false;
  this->b_performRegression = false;  
  
  ///////////
  // here comes the new code part different from the empty constructor
  ///////////  
  this->initFromConfig ( _conf, _confSection );
  this->setFastMinKernel( _fmk );
}

FMKGPHyperparameterOptimization::~FMKGPHyperparameterOptimization()
{
  
  //////////////////////////////////////
  // classification related variables //
  //////////////////////////////////////
  if ( this->fmk != NULL )
    delete this->fmk;
  
  if ( this->q != NULL )
    delete this->q;
  
  if ( this->pf != NULL )
    delete this->pf;  
  
  for ( uint i = 0 ; i < this->precomputedT.size(); i++ )
    delete [] ( this->precomputedT[i] );
  
  if ( this->ikmsum != NULL )
    delete this->ikmsum;  
  
  //////////////////////////////////////////////
  //           Iterative Linear Solver        //
  //////////////////////////////////////////////
  if ( this->linsolver != NULL )
    delete this->linsolver; 
  
  //////////////////////////////////////////////
  // likelihood computation related variables //
  //////////////////////////////////////////////   
  if ( this->eig != NULL )
    delete this->eig;    

  ////////////////////////////////////////////
  // variance computation related variables //
  ////////////////////////////////////////////  
  if ( this->precomputedTForVarEst != NULL )
    delete this->precomputedTForVarEst;


}

void FMKGPHyperparameterOptimization::initFromConfig ( const Config *_conf, 
                                                       const std::string & _confSection 
                                                     )
{ 
  ///////////////////////////////////
  // output/debug related settings //   
  ///////////////////////////////////  
  this->b_verbose = _conf->gB ( _confSection, "verbose", false );
  this->b_verboseTime = _conf->gB ( _confSection, "verboseTime", false );
  this->b_debug = _conf->gB ( _confSection, "debug", false );

  if ( this->b_verbose )
  {  
    std::cerr << "------------" << std::endl;
    std::cerr << "|  set-up  |" << std::endl;
    std::cerr << "------------" << std::endl;
  }
  
  
  //////////////////////////////////////
  // classification related variables //
  //////////////////////////////////////  
  this->b_performRegression = _conf->gB ( _confSection, "b_performRegression", false );
  
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
  
  this->d_parameterUpperBound = _conf->gD ( _confSection, "parameter_upper_bound", 2.5 );
  this->d_parameterLowerBound = _conf->gD ( _confSection, "parameter_lower_bound", 1.0 );
  
  std::string transform = _conf->gS( _confSection, "transform", "absexp" );
  
  if ( transform == "identity" )
  {
    this->pf = new NICE::PFIdentity( );
  }  
  else if ( transform == "absexp" )
  {
    this->pf = new NICE::PFAbsExp( 1.0, this->d_parameterLowerBound, this->d_parameterUpperBound );
  }
  else if ( transform == "exp" )
  {
    this->pf = new NICE::PFExp( 1.0, this->d_parameterLowerBound, this->d_parameterUpperBound );
  }
  else if ( transform == "MKL" )
  {
    //TODO generic, please :) load from a separate file or something like this!
    std::set<int> steps; steps.insert(4000); steps.insert(6000); //specific for VISAPP
    this->pf = new NICE::PFMKL( steps, this->d_parameterLowerBound, this->d_parameterUpperBound );
  }
  else if ( transform == "weightedDim" )
  {
    int pf_dim = _conf->gI ( _confSection, "pf_dim", 8 );
    this->pf = new NICE::PFWeightedDim( pf_dim, this->d_parameterLowerBound, this->d_parameterUpperBound );
  }
  else
  {
    fthrow(Exception, "Transformation type is unknown " << transform);
  }
  
  //////////////////////////////////////////////
  //           Iterative Linear Solver        //
  //////////////////////////////////////////////
  bool ils_verbose = _conf->gB ( _confSection, "ils_verbose", false );
  ils_max_iterations = _conf->gI ( _confSection, "ils_max_iterations", 1000 );
  if ( this->b_verbose )
    std::cerr << "FMKGPHyperparameterOptimization: maximum number of iterations is " << ils_max_iterations << std::endl;

  double ils_min_delta = _conf->gD ( _confSection, "ils_min_delta", 1e-7 );
  double ils_min_residual = _conf->gD ( _confSection, "ils_min_residual", 1e-7/*1e-2 */ );

  string ils_method = _conf->gS ( _confSection, "ils_method", "CG" );
  if ( ils_method.compare ( "CG" ) == 0 )
  {
    if ( this->b_verbose )
      std::cerr << "We use CG with " << ils_max_iterations << " iterations, " << ils_min_delta << " as min delta, and " << ils_min_residual << " as min res " << std::endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
    if ( this->b_verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients" << std::endl;
  }
  else if ( ils_method.compare ( "CGL" ) == 0 )
  {
    this->linsolver = new ILSConjugateGradientsLanczos ( ils_verbose , ils_max_iterations );
    if ( this->b_verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients (Lanczos)" << std::endl;
  }
  else if ( ils_method.compare ( "SYMMLQ" ) == 0 )
  {
    this->linsolver = new ILSSymmLqLanczos ( ils_verbose , ils_max_iterations );
    if ( this->b_verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS SYMMLQ" << std::endl;
  }
  else if ( ils_method.compare ( "MINRES" ) == 0 )
  {
    this->linsolver = new ILSMinResLanczos ( ils_verbose , ils_max_iterations );
    if ( this->b_verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS MINRES" << std::endl;
  }
  else
  {
    std::cerr << "FMKGPHyperparameterOptimization: " << _confSection << ":ils_method (" << ils_method << ") does not match any type (CG,CGL,SYMMLQ,MINRES), I will use CG" << std::endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
  }  
 
  
  /////////////////////////////////////
  // optimization related parameters //
  ///////////////////////////////////// 
  
  std::string optimizationMethod_s = _conf->gS ( _confSection, "optimization_method", "greedy" );
  if ( optimizationMethod_s == "greedy" )
    optimizationMethod = OPT_GREEDY;
  else if ( optimizationMethod_s == "downhillsimplex" )
    optimizationMethod = OPT_DOWNHILLSIMPLEX;
  else if ( optimizationMethod_s == "none" )
    optimizationMethod = OPT_NONE;
  else
    fthrow ( Exception, "Optimization method " << optimizationMethod_s << " is not known." );

  if ( this->b_verbose )
    std::cerr << "Using optimization method: " << optimizationMethod_s << std::endl;

  this->parameterStepSize = _conf->gD ( _confSection, "parameter_step_size", 0.1 );  
  
  this->optimizeNoise = _conf->gB ( _confSection, "optimize_noise", false );
  if ( this->b_verbose )
    std::cerr << "Optimize noise: " << ( optimizeNoise ? "on" : "off" ) << std::endl;    
  
  // if nothing is to be optimized and we have no other hyperparameters, then we could explicitly switch-off the optimization
  if ( !optimizeNoise && (transform == "identity") && (optimizationMethod != OPT_NONE) )
  {
    std::cerr << "FMKGPHyperparameterOptimization::initFromConfig No hyperparameter to optimize but optimization chosen... We ignore optimization. You might want to check this!" << std::endl;
    
    this->optimizationMethod = OPT_NONE;
  }
  

  downhillSimplexMaxIterations = _conf->gI ( _confSection, "downhillsimplex_max_iterations", 20 );
  // do not run longer than a day :)
  downhillSimplexTimeLimit     = _conf->gD ( _confSection, "downhillsimplex_time_limit", 24 * 60 * 60 );
  downhillSimplexParamTol      = _conf->gD ( _confSection, "downhillsimplex_delta", 0.01 );


  //////////////////////////////////////////////
  // likelihood computation related variables //
  ////////////////////////////////////////////// 

  this->verifyApproximation = _conf->gB ( _confSection, "verify_approximation", false );  
  
  // this->eig = new EigValuesTRLAN();
  // My time measurements show that both methods use equal time, a comparision
  // of their numerical performance has not been done yet  
  this->eig = new EVArnoldi ( _conf->gB ( _confSection, "eig_verbose", false ) /* verbose flag */, 10 );

  this->nrOfEigenvaluesToConsider = std::max ( 1, _conf->gI ( _confSection, "nrOfEigenvaluesToConsider", 1 ) );
  
  ////////////////////////////////////////////
  // variance computation related variables //
  ////////////////////////////////////////////  
  this->nrOfEigenvaluesToConsiderForVarApprox = std::max ( 1, _conf->gI ( _confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 ) );


  /////////////////////////////////////////////////////
  // online / incremental learning related variables //
  /////////////////////////////////////////////////////
  
  this->b_usePreviousAlphas = _conf->gB ( _confSection, "b_usePreviousAlphas", true );
  
  if ( this->b_verbose )
  {
    std::cerr << "------------" << std::endl;
    std::cerr << "|   start   |" << std::endl;
    std::cerr << "------------" << std::endl;
  }
}


///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// ///////////////////// 

void FMKGPHyperparameterOptimization::setParameterUpperBound ( const double & _parameterUpperBound )
{
  this->d_parameterUpperBound = _parameterUpperBound;
}
void FMKGPHyperparameterOptimization::setParameterLowerBound ( const double & _parameterLowerBound )
{
  this->d_parameterLowerBound = _parameterLowerBound;
}

std::set<uint> FMKGPHyperparameterOptimization::getKnownClassNumbers ( ) const
{
  return this->knownClasses;
}

void FMKGPHyperparameterOptimization::setPerformRegression ( const bool & _performRegression )
{
  //TODO check previously whether we already trained
  if ( false )
    throw NICE::Exception ( "FMPGKHyperparameterOptimization already initialized - switching between classification and regression not allowed!" );
  else
    this->b_performRegression = _performRegression;
}


void FMKGPHyperparameterOptimization::setFastMinKernel ( FastMinKernel * _fmk )
{
  //TODO check previously whether we already trained  
  if ( _fmk != NULL )
  {
    if ( this->fmk != NULL )
    {
      delete this->fmk;
      this->fmk = NULL;
    }    
    this->fmk = _fmk;
  }
  
  //
  if ( this->q != NULL )
  {  
    this->q->computeParametersFromData ( &(this->fmk->featureMatrix()) );
  }
}

void FMKGPHyperparameterOptimization::setNrOfEigenvaluesToConsiderForVarApprox ( const int & _nrOfEigenvaluesToConsiderForVarApprox )
{
  //TODO check previously whether we already trained
  this->nrOfEigenvaluesToConsiderForVarApprox = _nrOfEigenvaluesToConsiderForVarApprox;  
}


///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

inline void FMKGPHyperparameterOptimization::setupGPLikelihoodApprox ( GPLikelihoodApprox * & _gplike, 
                                                                       const std::map<uint, NICE::Vector> & _binaryLabels, 
                                                                       uint & _parameterVectorSize )
{
  _gplike = new GPLikelihoodApprox ( _binaryLabels, ikmsum, linsolver, eig, verifyApproximation, nrOfEigenvaluesToConsider );
  _gplike->setDebug( this->b_debug );
  _gplike->setVerbose( this->b_verbose );
  _parameterVectorSize = this->ikmsum->getNumParameters();
}

void FMKGPHyperparameterOptimization::updateEigenDecomposition( const int & _noEigenValues )
{
  //compute the largest eigenvalue of K + noise   
  
  try 
  {
    this->eig->getEigenvalues ( *ikmsum,  eigenMax, eigenMaxVectors, _noEigenValues  );
  }
  catch ( char const* exceptionMsg)
  {
    std::cerr << exceptionMsg << std::endl;
    throw("Problem in calculating Eigendecomposition of kernel matrix. Abort program...");
  }
  
  //NOTE EigenValue computation extracts EV and EW per default in decreasing order.
  
}

void FMKGPHyperparameterOptimization::performOptimization ( GPLikelihoodApprox & _gplike, 
                                                            const uint & _parameterVectorSize 
                                                          )
{
  if ( this->b_verbose )
    std::cerr << "perform optimization" << std::endl;
    
  if ( optimizationMethod == OPT_GREEDY )
  {
    if ( this->b_verbose )    
      std::cerr << "OPT_GREEDY!!! " << std::endl;
    
    // simple greedy strategy
    if ( ikmsum->getNumParameters() != 1 )
      fthrow ( Exception, "Reduce size of the parameter vector or use downhill simplex!" );

    NICE::Vector lB = ikmsum->getParameterLowerBounds();
    NICE::Vector uB = ikmsum->getParameterUpperBounds();
    
    if ( this->b_verbose )
      std::cerr << "lower bound " << lB << " upper bound " << uB << " parameterStepSize: " << parameterStepSize << std::endl;

    
    for ( double mypara = lB[0]; mypara <= uB[0]; mypara += this->parameterStepSize )
    {
      OPTIMIZATION::matrix_type hyperp ( 1, 1, mypara );
      _gplike.evaluate ( hyperp );
    }
  }
  else if ( optimizationMethod == OPT_DOWNHILLSIMPLEX )
  {
    //standard as before, normal optimization
    if ( this->b_verbose )    
        std::cerr << "DOWNHILLSIMPLEX!!! " << std::endl;

    // downhill simplex strategy
    OPTIMIZATION::DownhillSimplexOptimizer optimizer;

    OPTIMIZATION::matrix_type initialParams ( _parameterVectorSize, 1 );

    NICE::Vector currentParameters;
    ikmsum->getParameters ( currentParameters );

    for ( uint i = 0 ; i < _parameterVectorSize; i++ )
      initialParams(i,0) = currentParameters[ i ];

    if ( this->b_verbose )      
      std::cerr << "Initial parameters: " << initialParams << std::endl;

    //the scales object does not really matter in the actual implementation of Downhill Simplex
//     OPTIMIZATION::matrix_type scales ( _parameterVectorSize, 1);
//     scales.set(1.0);

    OPTIMIZATION::SimpleOptProblem optProblem ( &_gplike, initialParams, initialParams /* scales */ );

    optimizer.setMaxNumIter ( true, downhillSimplexMaxIterations );
    optimizer.setTimeLimit ( true, downhillSimplexTimeLimit );
    optimizer.setParamTol ( true, downhillSimplexParamTol );
    
    optimizer.optimizeProb ( optProblem );
  }
  else if ( optimizationMethod == OPT_NONE )
  {
    if ( this->b_verbose )
      std::cerr << "NO OPTIMIZATION!!! " << std::endl;

    // without optimization
    if ( optimizeNoise )
      fthrow ( Exception, "Deactivate optimize_noise!" );
    
    if ( this->b_verbose )
      std::cerr << "Optimization is deactivated!" << std::endl;
    
    double value (1.0);
    if ( this->d_parameterLowerBound == this->d_parameterUpperBound)
      value = this->d_parameterLowerBound;

    pf->setParameterLowerBounds ( NICE::Vector ( 1, value ) );
    pf->setParameterUpperBounds ( NICE::Vector ( 1, value ) );

    // we use the standard value
    OPTIMIZATION::matrix_type hyperp ( 1, 1, value );
    _gplike.setParameterLowerBound ( value );
    _gplike.setParameterUpperBound ( value );
    //we do not need to compute the likelihood here - we are only interested in directly obtaining alpha vectors
    _gplike.computeAlphaDirect( hyperp, eigenMax );
  }

  if ( this->b_verbose )
  {
    std::cerr << "Optimal hyperparameter was: " << _gplike.getBestParameters() << std::endl;
  }
}

void FMKGPHyperparameterOptimization::transformFeaturesWithOptimalParameters ( const GPLikelihoodApprox & _gplike, 
                                                                               const uint & parameterVectorSize 
                                                                             )
{
  // transform all features with the currently "optimal" parameter
  ikmsum->setParameters ( _gplike.getBestParameters() );    
}

void FMKGPHyperparameterOptimization::computeMatricesAndLUTs ( const GPLikelihoodApprox & _gplike )
{
  this->precomputedA.clear();
  this->precomputedB.clear();

  for ( std::map<uint, NICE::Vector>::const_iterator i = _gplike.getBestAlphas().begin(); i != _gplike.getBestAlphas().end(); i++ )
  {
    PrecomputedType A;
    PrecomputedType B;

    if ( this->b_debug &&  i->first == 1)
    {
        std::cerr << "Training for class " << i->first << endl;
        std::cerr << "  " << i->second << std::endl;
    }

    fmk->hik_prepare_alpha_multiplications ( i->second, A, B );
    A.setIoUntilEndOfFile ( false );
    B.setIoUntilEndOfFile ( false );

    this->precomputedA[ i->first ] = A;
    this->precomputedB[ i->first ] = B;

    if ( this->q != NULL )
    {
      double *T = fmk->hik_prepare_alpha_multiplications_fast ( A, B, this->q, this->pf );
      //just to be sure that we do not waste space here
      if ( precomputedT[ i->first ] != NULL )
        delete precomputedT[ i->first ];
      
      precomputedT[ i->first ] = T;
    }
  }
  
  if ( this->precomputedTForVarEst != NULL )
  {
    this->prepareVarianceApproximationRough();
  }
  else if ( this->nrOfEigenvaluesToConsiderForVarApprox > 0) 
  {
     this->prepareVarianceApproximationFine();
  }
  
  // in case that we should want to store the alpha vectors for incremental extensions
  if ( this->b_usePreviousAlphas )
    this->previousAlphas = _gplike.getBestAlphas();

}

#ifdef NICE_USELIB_MATIO
void FMKGPHyperparameterOptimization::optimizeBinary ( const sparse_t & _data, 
                                                       const NICE::Vector & _yl, 
                                                       const std::set<uint> & _positives, 
                                                       const std::set<uint> & _negatives, 
                                                       double _noise 
                                                     )
{
  std::map<uint, uint> examples;
  NICE::Vector y ( _yl.size() );
  uint ind = 0;
  for ( uint i = 0 ; i < _yl.size(); i++ )
  {
    if ( _positives.find ( i ) != _positives.end() ) {
      y[ examples.size() ] = 1.0;
      examples.insert ( pair<uint, uint> ( i, ind ) );
      ind++;
    } else if ( _negatives.find ( i ) != _negatives.end() ) {
      y[ examples.size() ] = -1.0;
      examples.insert ( pair<uint, uint> ( i, ind ) );
      ind++;
    }
  }
  y.resize ( examples.size() );
  std::cerr << "Examples: " << examples.size() << std::endl;

  optimize ( _data, y, examples, _noise );
}


void FMKGPHyperparameterOptimization::optimize ( const sparse_t & _data, 
                                                 const NICE::Vector & _y, 
                                                 const std::map<uint, uint> & _examples, 
                                                 double _noise 
                                               )
{
  NICE::Timer t;
  t.start();
  std::cerr << "Initializing data structure ..." << std::endl;
  if ( fmk != NULL ) delete fmk;
  fmk = new FastMinKernel ( _data, _noise, _examples );
  t.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for initializing the FastMinKernel structure: " << t.getLast() << std::endl;
  
  optimize ( _y );
}
#endif

uint FMKGPHyperparameterOptimization::prepareBinaryLabels ( std::map<uint, NICE::Vector> & _binaryLabels, 
                                                            const NICE::Vector & _y , 
                                                            std::set<uint> & _myClasses 
                                                          )
{
  _myClasses.clear();
  
  // determine which classes we have in our label vector
  // -> MATLAB: myClasses = unique(y);
  for ( NICE::Vector::const_iterator it = _y.begin(); it != _y.end(); it++ )
  {
    if ( _myClasses.find ( *it ) == _myClasses.end() )
    {
      _myClasses.insert ( *it );
    }
  }

  //count how many different classes appear in our data
  uint nrOfClasses ( _myClasses.size() );

  _binaryLabels.clear();
  //compute the corresponding binary label vectors
  if ( nrOfClasses > 2 )
  {
    //resize every labelVector and set all entries to -1.0
    for ( std::set<uint>::const_iterator k = _myClasses.begin(); k != _myClasses.end(); k++ )
    {
      _binaryLabels[ *k ].resize ( _y.size() );
      _binaryLabels[ *k ].set ( -1.0 );
    }

    // now look on every example and set the entry of its corresponding label vector to 1.0
    // proper existance should not be a problem
    for ( uint i = 0 ; i < _y.size(); i++ )
      _binaryLabels[ _y[i] ][i] = 1.0;
  }
  else if ( nrOfClasses == 2 )
  {
    //binary setting -- prepare a binary label vector
    NICE::Vector yb ( _y );

    this->i_binaryLabelNegative = *(_myClasses.begin());
    std::set<uint>::const_iterator classIt = _myClasses.begin(); classIt++;
    this->i_binaryLabelPositive = *classIt;
    
    if ( this->b_verbose )
    {
      std::cerr << "positiveClass : " << this->i_binaryLabelPositive << " negativeClass: " << this->i_binaryLabelNegative << std::endl;
      std::cerr << " all labels: " << _y << std::endl << std::endl;
    }

    for ( uint i = 0 ; i < yb.size() ; i++ )
      yb[i] = ( _y[i] == this->i_binaryLabelNegative ) ? -1.0 : 1.0;
    
    _binaryLabels[ this->i_binaryLabelPositive ] = yb;
        
    //we do NOT do real binary computation, but an implicite one with only a single object  
    nrOfClasses--;  
  }
  else //OCC setting
  {
    //we set the labels to 1, independent of the previously given class number
    //however, the original class numbers are stored and returned in classification
    NICE::Vector yOne ( _y.size(), 1 );
    
    _binaryLabels[ *(_myClasses.begin()) ] = yOne;
    
    //we have to indicate, that we are in an OCC setting
    nrOfClasses--;
  }

  return nrOfClasses;
}



void FMKGPHyperparameterOptimization::optimize ( const NICE::Vector & _y )
{
  if ( this->fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  this->labels  = _y;
  
  std::map< uint, NICE::Vector > binaryLabels;
  
  if ( this->b_performRegression )
  {
    // for regression, we are not interested in regression scores, rather than in any "label" 
    uint regressionLabel ( 1 );  
    binaryLabels.insert ( std::pair< uint, NICE::Vector> ( regressionLabel, _y ) );
    this->knownClasses.clear();
    this->knownClasses.insert ( regressionLabel );
  }
  else
  {
    this->prepareBinaryLabels ( binaryLabels, _y , knownClasses );    
  }
  
  //now call the main function :)
  this->optimize(binaryLabels);
}

  
void FMKGPHyperparameterOptimization::optimize ( std::map<uint, NICE::Vector> & _binaryLabels )
{
  Timer t;
  t.start();
  
  //how many different classes do we have right now?
  int nrOfClasses = _binaryLabels.size();
  
  if ( this->b_verbose )
  {
    std::cerr << "Initial noise level: " << this->fmk->getNoise() << std::endl;

    std::cerr << "Number of classes (=1 means we have a binary setting):" << nrOfClasses << std::endl;
    std::cerr << "Effective number of classes (neglecting classes without positive examples): " << this->knownClasses.size() << std::endl;
  }

  // combine standard model and noise model

  Timer t1;

  t1.start();
  //setup the kernel combination
  this->ikmsum = new IKMLinearCombination ();

  if ( this->b_verbose )
  {
    std::cerr << "_binaryLabels.size(): " << _binaryLabels.size() << std::endl;
  }

  //First model: noise
  this->ikmsum->addModel ( new IKMNoise ( this->fmk->get_n(), this->fmk->getNoise(), this->optimizeNoise ) );
  
  // set pretty low built-in noise, because we explicitely add the noise with the IKMNoise
  this->fmk->setNoise ( 0.0 );

  this->ikmsum->addModel ( new GMHIKernel ( this->fmk, this->pf, NULL /* no quantization */ ) );

  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the ikm-objects: " << t1.getLast() << std::endl;

  GPLikelihoodApprox * gplike;
  uint parameterVectorSize;

  t1.start();
  this->setupGPLikelihoodApprox ( gplike, _binaryLabels, parameterVectorSize );
  t1.stop();
    
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  if ( this->b_verbose )
  {
    std::cerr << "parameterVectorSize: " << parameterVectorSize << std::endl;
  }

  t1.start();
  // we compute all needed eigenvectors for standard classification and variance prediction at ones.
  // nrOfEigenvaluesToConsiderForVarApprox should NOT be larger than 1 if a method different than approximate_fine is used!
  
  this->updateEigenDecomposition(  std::max ( this->nrOfEigenvaluesToConsider, this->nrOfEigenvaluesToConsiderForVarApprox) );
    
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  if ( this->b_verbose )
    std::cerr << "resulting eigenvalues for first class: " << this->eigenMax[0] << std::endl;

  t1.start();
  this->performOptimization ( *gplike, parameterVectorSize );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

  if ( this->b_verbose )
    std::cerr << "Preparing classification ..." << std::endl;

  t1.start();
  this->transformFeaturesWithOptimalParameters ( *gplike, parameterVectorSize );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;

  t1.start();
  this->computeMatricesAndLUTs ( *gplike );
  t1.stop();
  if ( this->b_verboseTime )
    std::cerr << "Time used for setting up the A'nB -objects: " << t1.getLast() << std::endl;

  t.stop();

  ResourceStatistics rs;
  std::cerr << "Time used for learning: " << t.getLast() << std::endl;
  long maxMemory;
  rs.getMaximumMemory ( maxMemory );
  std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;

  //don't waste memory
  delete gplike;

}

void FMKGPHyperparameterOptimization::prepareVarianceApproximationRough()
{
  PrecomputedType AVar;
  this->fmk->hikPrepareKVNApproximation ( AVar );

  this->precomputedAForVarEst = AVar;
  this->precomputedAForVarEst.setIoUntilEndOfFile ( false );

  if ( this->q != NULL )
  {   
    double *T = this->fmk->hikPrepareLookupTableForKVNApproximation ( this->q, this->pf );
    this->precomputedTForVarEst = T;
  }
}

void FMKGPHyperparameterOptimization::prepareVarianceApproximationFine()
{
  if ( this->eigenMax.size() < (uint) this->nrOfEigenvaluesToConsiderForVarApprox ) 
  {
    std::cerr << "not enough eigenvectors computed for fine approximation of predictive variance. " <<std::endl;
    std::cerr << "Current number of EV: " <<  this->eigenMax.size() << " but required: " << (uint) this->nrOfEigenvaluesToConsiderForVarApprox << std::endl;
    this->updateEigenDecomposition(  this->nrOfEigenvaluesToConsiderForVarApprox ); 
  }
}

uint FMKGPHyperparameterOptimization::classify ( const NICE::SparseVector & _xstar, 
                                                 NICE::SparseVector & _scores 
                                               )  const
{
  // loop through all classes
  if ( this->precomputedA.size() == 0 )
  {
    fthrow ( Exception, "The precomputation vector is zero...have you trained this classifier?" );
  }

  uint maxClassNo = 0;
  for ( std::map<uint, PrecomputedType>::const_iterator i = this->precomputedA.begin() ; i != this->precomputedA.end(); i++ )
  {
    uint classno = i->first;
    maxClassNo = std::max ( maxClassNo, classno );
    double beta;

    if ( this->q != NULL ) {
      std::map<uint, double *>::const_iterator j = this->precomputedT.find ( classno );
      double *T = j->second;
      this->fmk->hik_kernel_sum_fast ( T, this->q, _xstar, beta );
    } else {
      const PrecomputedType & A = i->second;
      std::map<uint, PrecomputedType>::const_iterator j = this->precomputedB.find ( classno );
      const PrecomputedType & B = j->second;

      // fmk->hik_kernel_sum ( A, B, _xstar, beta ); if A, B are of type Matrix
      // Giving the transformation pf as an additional
      // argument is necessary due to the following reason:
      // FeatureMatrixT is sorted according to the original values, therefore,
      // searching for upper and lower bounds ( findFirst... functions ) require original feature
      // values as inputs. However, for calculation we need the transformed features values.

      this->fmk->hik_kernel_sum ( A, B, _xstar, beta, pf );
    }

    _scores[ classno ] = beta;
  }
  _scores.setDim ( maxClassNo + 1 );
  
  if ( this->precomputedA.size() > 1 )
  { // multi-class classification
    return _scores.maxElement();
  }
  else if ( this->knownClasses.size() == 2 ) // binary setting
  {      
    _scores[ this->i_binaryLabelNegative ] = -_scores[ this->i_binaryLabelPositive ];     
    return _scores[ this->i_binaryLabelPositive ] <= 0.0 ? this->i_binaryLabelNegative : this->i_binaryLabelPositive;
  }
  else //OCC or regression setting
  {
    return 1;
  }
}

uint FMKGPHyperparameterOptimization::classify ( const NICE::Vector & _xstar, 
                                                 NICE::SparseVector & _scores 
                                               ) const
{
  
  // loop through all classes
  if ( this->precomputedA.size() == 0 )
  {
    fthrow ( Exception, "The precomputation vector is zero...have you trained this classifier?" );
  }

  uint maxClassNo = 0;
  for ( std::map<uint, PrecomputedType>::const_iterator i = this->precomputedA.begin() ; i != this->precomputedA.end(); i++ )
  {
    uint classno = i->first;
    maxClassNo = std::max ( maxClassNo, classno );
    
    double beta;

    if ( this->q != NULL )
    {
      std::map<uint, double *>::const_iterator j = this->precomputedT.find ( classno );
      double *T = j->second;
      this->fmk->hik_kernel_sum_fast ( T, this->q, _xstar, beta );
    }
    else
    {
      const PrecomputedType & A = i->second;
      std::map<uint, PrecomputedType>::const_iterator j = this->precomputedB.find ( classno );
      const PrecomputedType & B = j->second;

      // fmk->hik_kernel_sum ( A, B, _xstar, beta ); if A, B are of type Matrix
      // Giving the transformation pf as an additional
      // argument is necessary due to the following reason:
      // FeatureMatrixT is sorted according to the original values, therefore,
      // searching for upper and lower bounds ( findFirst... functions ) require original feature
      // values as inputs. However, for calculation we need the transformed features values.

      this->fmk->hik_kernel_sum ( A, B, _xstar, beta, this->pf );
    }

    _scores[ classno ] = beta;
  }
  _scores.setDim ( maxClassNo + 1 );
  
  
  if ( this->precomputedA.size() > 1 )
  { // multi-class classification
    return _scores.maxElement();
  }
  else if ( this->knownClasses.size() == 2 ) // binary setting
  {      
    _scores[ this->i_binaryLabelNegative ] = -_scores[ this->i_binaryLabelPositive ];     
        
    return _scores[ this->i_binaryLabelPositive ] <= 0.0 ? this->i_binaryLabelNegative : this->i_binaryLabelPositive;
  }
  else //OCC or regression setting
  {
    return 1;
  }
}

    //////////////////////////////////////////
    // variance computation: sparse inputs
    //////////////////////////////////////////

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateRough ( const NICE::SparseVector & _x, 
                                                                                  double & _predVariance 
                                                                                ) const
{
  // security check!
  if ( this->pf == NULL )
   fthrow ( Exception, "pf is NULL...have you prepared the uncertainty prediction? Aborting..." );   
  
  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = _x.begin(); it != _x.end(); it++ )
  {
    kSelf += this->pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }
  
  // ---------------- compute the approximation of the second term --------------------
  double normKStar;

  if ( this->q != NULL )
  {
    if ( precomputedTForVarEst == NULL )
    {
      fthrow ( Exception, "The precomputed LUT for uncertainty prediction is NULL...have you prepared the uncertainty prediction? Aborting..." );
    }
    fmk->hikComputeKVNApproximationFast ( precomputedTForVarEst, this->q, _x, normKStar );
  }
  else
  {
    if ( precomputedAForVarEst.size () == 0 )
    {
      fthrow ( Exception, "The precomputedAForVarEst is empty...have you trained this classifer? Aborting..." );
    }
    fmk->hikComputeKVNApproximation ( precomputedAForVarEst, _x, normKStar, pf );
  }

  _predVariance = kSelf - ( 1.0 / eigenMax[0] )* normKStar;
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine ( const NICE::SparseVector & _x, 
                                                                                 double & _predVariance 
                                                                               ) const
{  
  if ( this->b_debug )
  {
    std::cerr << "FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine" << std::endl;
  }
  
  // security check!
  if ( this->eigenMaxVectors.rows() == 0 )
  {
      fthrow ( Exception, "eigenMaxVectors is empty...have you trained this classifer? Aborting..." );
  }    
  
  // ---------------- compute the first term --------------------
//   Timer t;
//   t.start();

  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = _x.begin(); it != _x.end(); it++ )
  {
    kSelf += this->pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }
  
  if ( this->b_debug )
  {
    std::cerr << "FMKGPHyp::VarApproxFine  -- kSelf: " << kSelf << std::endl;
  }
  
  // ---------------- compute the approximation of the second term --------------------
//    t.stop();  
//   std::cerr << "ApproxFine -- time for first term: "  << t.getLast()  << std::endl;

//   t.start();
  NICE::Vector kStar;
  this->fmk->hikComputeKernelVector ( _x, kStar );
  
  if ( this->b_debug )
  {
    std::cerr << "FMKGPHyp::VarApproxFine  -- kStar: " << kStar << std::endl;
    std::cerr << "nrOfEigenvaluesToConsiderForVarApprox: " << this->nrOfEigenvaluesToConsiderForVarApprox << std::endl;
  }  
  
/*  t.stop();
  std::cerr << "ApproxFine -- time for kernel vector: "  << t.getLast()  << std::endl;*/
    
//     NICE::Vector multiplicationResults; // will contain nrOfEigenvaluesToConsiderForVarApprox many entries
//     multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    NICE::Vector multiplicationResults( this->nrOfEigenvaluesToConsiderForVarApprox-1, 0.0 );
    //ok, there seems to be a nasty thing in computing multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    //wherefor it takes aeons...
    //so we compute it by ourselves
    
  if ( this->b_debug )
  {
    std::cerr << "FMKGPHyp::VarApproxFine  -- nrOfEigenvaluesToConsiderForVarApprox: " << this->nrOfEigenvaluesToConsiderForVarApprox << std::endl;
    std::cerr << "FMKGPHyp::VarApproxFine  -- initial multiplicationResults: " << multiplicationResults << std::endl;    
  }    
    
    
    
//     for ( uint tmpI = 0; tmpI < kStar.size(); tmpI++)
    NICE::Matrix::const_iterator eigenVecIt = this->eigenMaxVectors.begin();
//       double kStarI ( kStar[tmpI] );
    for ( int tmpJ = 0; tmpJ < this->nrOfEigenvaluesToConsiderForVarApprox-1; tmpJ++)
    {
      for ( NICE::Vector::const_iterator kStarIt = kStar.begin(); kStarIt != kStar.end(); kStarIt++,eigenVecIt++)
      {        
        multiplicationResults[tmpJ] += (*kStarIt) * (*eigenVecIt);//eigenMaxVectors(tmpI,tmpJ);
      }
    }
    
  if ( this->b_debug )
  {
    std::cerr << "FMKGPHyp::VarApproxFine  -- computed multiplicationResults: " << multiplicationResults << std::endl;    
  }      
    
    double projectionLength ( 0.0 );
    double currentSecondTerm ( 0.0 );
    double sumOfProjectionLengths ( 0.0 );
    int cnt ( 0 );
    NICE::Vector::const_iterator it = multiplicationResults.begin();

    while ( cnt < ( this->nrOfEigenvaluesToConsiderForVarApprox - 1 ) )
    {
      projectionLength = ( *it );
      currentSecondTerm += ( 1.0 / this->eigenMax[cnt] ) * pow ( projectionLength, 2 );
      sumOfProjectionLengths += pow ( projectionLength, 2 );
      
      it++;
      cnt++;      
    }
    
    
    double normKStar ( pow ( kStar.normL2 (), 2 ) );

    currentSecondTerm += ( 1.0 / this->eigenMax[this->nrOfEigenvaluesToConsiderForVarApprox-1] ) * ( normKStar - sumOfProjectionLengths );
    
    if ( ( normKStar - sumOfProjectionLengths ) < 0 )
    {
      std::cerr << "Attention: normKStar - sumOfProjectionLengths is smaller than zero -- strange!" << std::endl;
    }
    _predVariance = kSelf - currentSecondTerm; 
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::SparseVector & x, double & predVariance ) const
{
  // security check!  
  if ( this->ikmsum->getNumberOfModels() == 0 )
  {
    fthrow ( Exception, "ikmsum is empty... have you trained this classifer? Aborting..." );
  }  
  
    Timer t;
//   t.start();
  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = x.begin(); it != x.end(); it++ )
  {
    kSelf += this->pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }

  // ---------------- compute the second term --------------------
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );
  
  //now run the ILS method
  NICE::Vector diagonalElements;
  ikmsum->getDiagonalElements ( diagonalElements );

  // init simple jacobi pre-conditioning
  ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( linsolver );
  //TODO what to do for other solver techniques?


  //perform pre-conditioning
  if ( linsolver_cg != NULL )
    linsolver_cg->setJacobiPreconditioner ( diagonalElements );
  

  NICE::Vector beta;
  
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
  beta = (kStar * (1.0 / eigenMax[0]) );
  
  linsolver->solveLin ( *ikmsum, kStar, beta );

  beta *= kStar;
  
  double currentSecondTerm( beta.Sum() );
  predVariance = kSelf - currentSecondTerm;
}

    //////////////////////////////////////////
    // variance computation: non-sparse inputs
    //////////////////////////////////////////
    
void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateRough ( const NICE::Vector & x, double & predVariance ) const
{
  // security check!
  if ( pf == NULL )
   fthrow ( Exception, "pf is NULL...have you prepared the uncertainty prediction? Aborting..." ); 
  
  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  int dim ( 0 );
  for ( NICE::Vector::const_iterator it = x.begin(); it != x.end(); it++, dim++ )
  {
    kSelf += pf->f ( 0, *it );
    // if weighted dimensions:
    //kSelf += pf->f(dim,*it);
  }
  
  // ---------------- compute the approximation of the second term --------------------
  double normKStar;

  if ( this->q != NULL )
  {
    if ( precomputedTForVarEst == NULL )
    {
      fthrow ( Exception, "The precomputed LUT for uncertainty prediction is NULL...have you prepared the uncertainty prediction? Aborting..." );
    }
    fmk->hikComputeKVNApproximationFast ( precomputedTForVarEst, this->q, x, normKStar );
  }
  else
  {
    if ( precomputedAForVarEst.size () == 0 )
    {
      fthrow ( Exception, "The precomputedAForVarEst is empty...have you trained this classifer? Aborting..." );
    }    
    fmk->hikComputeKVNApproximation ( precomputedAForVarEst, x, normKStar, this->pf );
  }

  predVariance = kSelf - ( 1.0 / eigenMax[0] )* normKStar;
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine ( const NICE::Vector & _x, 
                                                                                 double & _predVariance 
                                                                               ) const
{
  // security check!
  if ( this->eigenMaxVectors.rows() == 0 )
  {
      fthrow ( Exception, "eigenMaxVectors is empty...have you trained this classifer? Aborting..." );
  }  
  
  // ---------------- compute the first term --------------------

  double kSelf ( 0.0 );
  uint dim ( 0 );
  for ( NICE::Vector::const_iterator it = _x.begin(); it != _x.end(); it++, dim++ )
  {
    kSelf += this->pf->f ( 0, *it );
    // if weighted dimensions:
    //kSelf += pf->f(dim,*it);
  }
  // ---------------- compute the approximation of the second term --------------------
  NICE::Vector kStar;
  this->fmk->hikComputeKernelVector ( _x, kStar );


    //ok, there seems to be a nasty thing in computing multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    //wherefor it takes aeons...
    //so we compute it by ourselves
//     NICE::Vector multiplicationResults; // will contain nrOfEigenvaluesToConsiderForVarApprox many entries
//     multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );

    NICE::Vector multiplicationResults(this-> nrOfEigenvaluesToConsiderForVarApprox-1, 0.0 );
    NICE::Matrix::const_iterator eigenVecIt = this->eigenMaxVectors.begin();
    for ( int tmpJ = 0; tmpJ < this->nrOfEigenvaluesToConsiderForVarApprox-1; tmpJ++)
    {
      for ( NICE::Vector::const_iterator kStarIt = kStar.begin(); kStarIt != kStar.end(); kStarIt++,eigenVecIt++)
      {        
        multiplicationResults[tmpJ] += (*kStarIt) * (*eigenVecIt);//eigenMaxVectors(tmpI,tmpJ);
      }
    }

    double projectionLength ( 0.0 );
    double currentSecondTerm ( 0.0 );
    double sumOfProjectionLengths ( 0.0 );
    int cnt ( 0 );
    NICE::Vector::const_iterator it = multiplicationResults.begin();

    while ( cnt < ( this->nrOfEigenvaluesToConsiderForVarApprox - 1 ) )
    {
      projectionLength = ( *it );
      currentSecondTerm += ( 1.0 / this->eigenMax[cnt] ) * pow ( projectionLength, 2 );
      sumOfProjectionLengths += pow ( projectionLength, 2 );
      
      it++;
      cnt++;      
    }
    
    
    double normKStar ( pow ( kStar.normL2 (), 2 ) );

    currentSecondTerm += ( 1.0 / this->eigenMax[nrOfEigenvaluesToConsiderForVarApprox-1] ) * ( normKStar - sumOfProjectionLengths );
    

    if ( ( normKStar - sumOfProjectionLengths ) < 0 )
    {
      std::cerr << "Attention: normKStar - sumOfProjectionLengths is smaller than zero -- strange!" << std::endl;
    }
    _predVariance = kSelf - currentSecondTerm; 
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::Vector & _x, 
                                                                       double & _predVariance 
                                                                     ) const
{
  if ( this->ikmsum->getNumberOfModels() == 0 )
  {
    fthrow ( Exception, "ikmsum is empty... have you trained this classifer? Aborting..." );
  }  

  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  uint dim ( 0 );
  for ( NICE::Vector::const_iterator it = _x.begin(); it != _x.end(); it++, dim++ )
  {
    kSelf += this->pf->f ( 0, *it );
    // if weighted dimensions:
    //kSelf += pf->f(dim,*it);
  }
  

  // ---------------- compute the second term --------------------
  NICE::Vector kStar;
  this->fmk->hikComputeKernelVector ( _x, kStar );
 
  //now run the ILS method
  NICE::Vector diagonalElements;
  this->ikmsum->getDiagonalElements ( diagonalElements );

  // init simple jacobi pre-conditioning
  ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( this->linsolver );


  //perform pre-conditioning
  if ( linsolver_cg != NULL )
    linsolver_cg->setJacobiPreconditioner ( diagonalElements );
  

  NICE::Vector beta;
  
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
  beta = (kStar * (1.0 / this->eigenMax[0]) );
  this->linsolver->solveLin ( *ikmsum, kStar, beta );

  beta *= kStar;
  
  double currentSecondTerm( beta.Sum() );
  
  _predVariance = kSelf - currentSecondTerm;
}    

///////////////////// INTERFACE PERSISTENT /////////////////////
// interface specific methods for store and restore
///////////////////// INTERFACE PERSISTENT ///////////////////// 

void FMKGPHyperparameterOptimization::restore ( std::istream & _is, 
                                                int _format 
                                              )
{
  bool b_restoreVerbose ( false );

#ifdef B_RESTOREVERBOSE
  b_restoreVerbose = true;
#endif

  if ( _is.good() )
  {
    if ( b_restoreVerbose ) 
      std::cerr << " in FMKGPHyperparameterOptimization restore" << std::endl;
    
    std::string tmp;
    _is >> tmp; //class name 
    
    if ( ! this->isStartTag( tmp, "FMKGPHyperparameterOptimization" ) )
    {
        std::cerr << " WARNING - attempt to restore FMKGPHyperparameterOptimization, but start flag " << tmp << " does not match! Aborting... " << std::endl;
      throw;
    } 

    if (fmk != NULL)
    {
      delete fmk;
      fmk = NULL;
    }
    
    if ( ikmsum != NULL )
    {
      delete ikmsum;
    }
    ikmsum = new IKMLinearCombination (); 
    if ( b_restoreVerbose ) 
      std::cerr << "ikmsum object created" << std::endl;
    
    
    _is.precision ( numeric_limits<double>::digits10 + 1 );
       
    
    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      _is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "FMKGPHyperparameterOptimization" ) )
      {
        b_endOfBlock = true;
        continue;
      }      
      
      tmp = this->removeStartTag ( tmp );
      
      if ( b_restoreVerbose )
        std::cerr << " currently restore section " << tmp << " in FMKGPHyperparameterOptimization" << std::endl;
      
      ///////////////////////////////////
      // output/debug related settings //   
      ///////////////////////////////////
      if ( tmp.compare("verbose") == 0 )
      {
        _is >> this->b_verbose;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("verboseTime") == 0 )
      {
        _is >> this->b_verboseTime;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("debug") == 0 )
      {
        _is >> this->b_debug;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      } 
      //////////////////////////////////////
      // classification related variables //
      //////////////////////////////////////
      else if ( tmp.compare("b_performRegression") == 0 )
      {
        _is >> this->b_performRegression;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }     
      else if ( tmp.compare("fmk") == 0 )
      {
        if ( this->fmk != NULL )
          delete this->fmk;        
        this->fmk = new FastMinKernel();
        this->fmk->restore( _is, _format );

        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("q") == 0 )
      {
        std::string isNull;
        _is >> isNull; // NOTNULL or NULL
        if (isNull.compare("NOTNULL") == 0)
        {   
          if ( this->q != NULL )
            delete this->q;
          
          std::string s_quantType;
           _is >> s_quantType;
           s_quantType = this->removeStartTag ( s_quantType );
          
          if ( s_quantType == "Quantization1DAequiDist0To1" )
          {
            this->q = new NICE::Quantization1DAequiDist0To1();
          }
          else if ( s_quantType == "Quantization1DAequiDist0ToMax" )
          {           
            this->q = new NICE::Quantization1DAequiDist0ToMax ( );
          }
          else if ( s_quantType == "QuantizationNDAequiDist0ToMax" )
          {
            this->q = new NICE::QuantizationNDAequiDist0ToMax ( );
          }
          else
          {
            fthrow(Exception, "Quantization type is unknown " << s_quantType);
          }
               
          this->q->restore ( _is, _format );
        }
        else
        {
          if ( this->q != NULL )
            delete this->q;
          this->q = NULL;
        }
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );        
      }
      else if  ( tmp.compare("parameterUpperBound") == 0 )
      {
        _is >> this->d_parameterUpperBound;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("parameterLowerBound") == 0 )
      {
        _is >> this->d_parameterLowerBound;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("pf") == 0 )
      {
      
        _is >> tmp; // start of block 
        if ( this->isEndTag( tmp, "pf" ) )
        {
          std::cerr << " ParameterizedFunction object can not be restored. Aborting..." << std::endl;
          throw;
        } 
        
        std::string transform ( this->removeStartTag( tmp ) );

        if ( transform == "PFAbsExp" )
        {
          this->pf = new NICE::PFAbsExp ();
        } else if ( transform == "PFExp" ) {
          this->pf = new NICE::PFExp ();
        }
        else if ( transform == "PFIdentity" )
        {
          this->pf = new NICE::PFIdentity( );          
        } else {
          fthrow(Exception, "Transformation type is unknown " << transform);
        }
        
        this->pf->restore( _is, _format);
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("precomputedA") == 0 )
      {
        _is >> tmp; // size
        uint preCompSize ( 0 );
        _is >> preCompSize;
        this->precomputedA.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedA with size: " << preCompSize << std::endl;
        for ( int i = 0; i < preCompSize; i++ )
        {
          uint nr;
          _is >> nr;
          PrecomputedType pct;
          pct.setIoUntilEndOfFile ( false );
          pct.restore ( _is, _format );
          this->precomputedA.insert ( std::pair<uint, PrecomputedType> ( nr, pct ) );
        }
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }        
      else if ( tmp.compare("precomputedB") == 0 )
      {
        _is >> tmp; // size
        uint preCompSize ( 0 );
        _is >> preCompSize;
        this->precomputedB.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedB with size: " << preCompSize << std::endl;
        for ( int i = 0; i < preCompSize; i++ )
        {
          uint nr;
          _is >> nr;
          PrecomputedType pct;
          pct.setIoUntilEndOfFile ( false );
          pct.restore ( _is, _format );
          this->precomputedB.insert ( std::pair<uint, PrecomputedType> ( nr, pct ) );
        }    
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }       
      else if ( tmp.compare("precomputedT") == 0 )
      {
        _is >> tmp; // size
        uint precomputedTSize ( 0 );
        _is >> precomputedTSize;

        this->precomputedT.clear();
        
        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedT with size: " << precomputedTSize << std::endl;

        if ( precomputedTSize > 0 )
        {
          if ( b_restoreVerbose ) 
            std::cerr << " restore precomputedT" << std::endl;
          _is >> tmp;
          int sizeOfLUT;
          _is >> sizeOfLUT;    
          
          for (int i = 0; i < precomputedTSize; i++)
          {
            _is >> tmp;
            uint index;
            _is >> index;        
            double * array = new double [ sizeOfLUT];
            for ( int i = 0; i < sizeOfLUT; i++ )
            {
              _is >> array[i];
            }
            this->precomputedT.insert ( std::pair<uint, double*> ( index, array ) );
          }
        } 
        else
        {
          if ( b_restoreVerbose ) 
            std::cerr << " skip restoring precomputedT" << std::endl;
        }
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("labels") == 0 )
      {
        _is >> this->labels;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("binaryLabelPositive") == 0 )
      {
        _is >> this->i_binaryLabelPositive;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("binaryLabelNegative") == 0 )
      {
        _is >> this->i_binaryLabelNegative;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("knownClasses") == 0 )
      {
        _is >> tmp; // size
        uint knownClassesSize ( 0 );
        _is >> knownClassesSize;

        this->knownClasses.clear();
        
        if ( knownClassesSize > 0 )
        {          
          for (uint i = 0; i < knownClassesSize; i++)
          {
            uint classNo;
            _is >> classNo;        
            this->knownClasses.insert ( classNo );
          }
        } 
        else
        {
          //nothing to do
        }        
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );        
      }
      else if ( tmp.compare("ikmsum") == 0 )
      {
        bool b_endOfBlock ( false ) ;
        
        while ( !b_endOfBlock )
        {
          _is >> tmp; // start of block 
          
          if ( this->isEndTag( tmp, "ikmsum" ) )
          {
            b_endOfBlock = true;
            continue;
          }      
          
          tmp = this->removeStartTag ( tmp );        
          if ( tmp.compare("IKMNoise") == 0 )
          {
            IKMNoise * ikmnoise = new IKMNoise ();
            ikmnoise->restore ( _is, _format );
            
            if ( b_restoreVerbose ) 
              std::cerr << " add ikmnoise to ikmsum object " << std::endl;
            ikmsum->addModel ( ikmnoise );        
          }
          else
          { 
            std::cerr << "WARNING -- unexpected ikmsum object -- " << tmp << " -- for restoration... aborting" << std::endl;
            throw;
          }         
        }
      }
      //////////////////////////////////////////////
      //           Iterative Linear Solver        //
      //////////////////////////////////////////////
      
      else if  ( tmp.compare("linsolver") == 0 )
      {
        //TODO linsolver  
        // current solution: hard coded with default values, since LinearSolver does not offer Persistent functionalities
        this->linsolver = new ILSConjugateGradients ( false , 1000, 1e-7, 1e-7 );        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if  ( tmp.compare("ils_max_iterations") == 0 )
      {
        _is >> ils_max_iterations;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      /////////////////////////////////////
      // optimization related parameters //
      ///////////////////////////////////// 
      
      else if  ( tmp.compare("optimizationMethod") == 0 )
      {
        unsigned int ui_optimizationMethod;        
        _is >> ui_optimizationMethod;        
        optimizationMethod = static_cast<OPTIMIZATIONTECHNIQUE> ( ui_optimizationMethod ) ;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if  ( tmp.compare("optimizeNoise") == 0 )
      {
        _is >> optimizeNoise;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }        
      else if  ( tmp.compare("parameterStepSize") == 0 )
      {
        _is >> parameterStepSize;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("downhillSimplexMaxIterations") == 0 )
      {
        _is >> downhillSimplexMaxIterations;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("downhillSimplexTimeLimit") == 0 )
      {
        _is >> downhillSimplexTimeLimit;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("downhillSimplexParamTol") == 0 )
      {
        _is >> downhillSimplexParamTol;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      //////////////////////////////////////////////
      // likelihood computation related variables //
      //////////////////////////////////////////////
      else if ( tmp.compare("verifyApproximation") == 0 )
      {
        _is >> verifyApproximation;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("eig") == 0 )
      {
        //TODO eig
        // currently hard coded, since EV does not offer Persistent functionalities and 
        // in addition, we currently have no other choice for EV then EVArnoldi
        this->eig = new EVArnoldi ( false /*eig_verbose */, 10 );        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }     
      else if ( tmp.compare("nrOfEigenvaluesToConsider") == 0 )
      {
        _is >> nrOfEigenvaluesToConsider;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("eigenMax") == 0 )
      {
        _is >> eigenMax;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("eigenMaxVectors") == 0 )
      {
        _is >> eigenMaxVectors;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      ////////////////////////////////////////////
      // variance computation related variables //
      ////////////////////////////////////////////
      else if ( tmp.compare("nrOfEigenvaluesToConsiderForVarApprox") == 0 )
      {
        _is >> nrOfEigenvaluesToConsiderForVarApprox;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("precomputedAForVarEst") == 0 )
      {
        int sizeOfAForVarEst;
        _is >> sizeOfAForVarEst;
        
        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedAForVarEst with size: " << sizeOfAForVarEst << std::endl;
        
        if (sizeOfAForVarEst > 0)
        {
          precomputedAForVarEst.clear();
          
          precomputedAForVarEst.setIoUntilEndOfFile ( false );
          precomputedAForVarEst.restore ( _is, _format );
        }
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }        
      else if ( tmp.compare("precomputedTForVarEst") == 0 )
      {
        std::string isNull;
        _is >> isNull; // NOTNULL or NULL
        if ( b_restoreVerbose ) 
          std::cerr << "content of isNull: " << isNull << std::endl;    
        if (isNull.compare("NOTNULL") == 0)
        {
          if ( b_restoreVerbose ) 
            std::cerr << "restore precomputedTForVarEst" << std::endl;
          
          int sizeOfLUT;
          _is >> sizeOfLUT;      
          precomputedTForVarEst = new double [ sizeOfLUT ];
          for ( int i = 0; i < sizeOfLUT; i++ )
          {
            _is >> precomputedTForVarEst[i];
          }      
        }
        else
        {
          if ( b_restoreVerbose ) 
            std::cerr << "skip restoring of precomputedTForVarEst" << std::endl;
          if (precomputedTForVarEst != NULL)
            delete precomputedTForVarEst;
        }
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }  
      /////////////////////////////////////////////////////
      // online / incremental learning related variables //
      /////////////////////////////////////////////////////
      else if  ( tmp.compare("b_usePreviousAlphas") == 0 )
      {
        _is >> b_usePreviousAlphas;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("previousAlphas") == 0 )
      {        
        _is >> tmp; // size
        uint sizeOfPreviousAlphas ( 0 );
        _is >> sizeOfPreviousAlphas;
        this->previousAlphas.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore previousAlphas with size: " << sizeOfPreviousAlphas << std::endl;
        for ( int i = 0; i < sizeOfPreviousAlphas; i++ )
        {
          uint classNo;
          _is >> classNo;
          NICE::Vector classAlpha;
          _is >> classAlpha;
          this->previousAlphas.insert ( std::pair< uint, NICE::Vector > ( classNo, classAlpha ) );
        }
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else
      {
        std::cerr << "WARNING -- unexpected FMKGPHyper object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;
      }
      
 
    }

    


    //NOTE are there any more models you added? then add them here respectively in the correct order
    //.....  

    //the last one is the GHIK - which we do not have to restore, but simply reset it
    if ( b_restoreVerbose ) 
      std::cerr << " add GMHIKernel" << std::endl;
    ikmsum->addModel ( new GMHIKernel ( fmk, this->pf, this->q ) );    
    
    if ( b_restoreVerbose ) 
      std::cerr << " restore positive and negative label" << std::endl;

      


    this->knownClasses.clear();
    
    if ( b_restoreVerbose ) 
      std::cerr << " fill known classes object " << std::endl;
    
    if ( this->precomputedA.size() == 1)
    {
      this->knownClasses.insert( this->i_binaryLabelPositive );
      this->knownClasses.insert( this->i_binaryLabelNegative );
      if ( b_restoreVerbose ) 
        std::cerr << " binary setting - added corresp. two class numbers" << std::endl;
    }
    else
    {
      for ( std::map<uint, PrecomputedType>::const_iterator itA = this->precomputedA.begin(); itA != this->precomputedA.end(); itA++)
          knownClasses.insert ( itA->first );
      if ( b_restoreVerbose ) 
        std::cerr << " multi class setting - added corresp. multiple class numbers" << std::endl;
    }
  }
  else
  {
    std::cerr << "InStream not initialized - restoring not possible!" << std::endl;
    throw;
  }
}

void FMKGPHyperparameterOptimization::store ( std::ostream & _os, 
                                              int _format 
                                            ) const
{
  if ( _os.good() )
  {
    // show starting point
    _os << this->createStartTag( "FMKGPHyperparameterOptimization" ) << std::endl;

//     _os.precision ( numeric_limits<double>::digits10 + 1 );
      
    
    ///////////////////////////////////
    // output/debug related settings //   
    ///////////////////////////////////
    
    _os << this->createStartTag( "verbose" ) << std::endl;
    _os << this->b_verbose << std::endl;
    _os << this->createEndTag( "verbose" ) << std::endl;
    
    _os << this->createStartTag( "verboseTime" ) << std::endl;
    _os << this->b_verboseTime << std::endl;
    _os << this->createEndTag( "verboseTime" ) << std::endl;
    
    _os << this->createStartTag( "debug" ) << std::endl;
    _os << this->b_debug << std::endl;
    _os << this->createEndTag( "debug" ) << std::endl;
    
    //////////////////////////////////////
    // classification related variables //
    //////////////////////////////////////    
    
    _os << this->createStartTag( "b_performRegression" ) << std::endl;
    _os << b_performRegression << std::endl;
    _os << this->createEndTag( "b_performRegression" ) << std::endl;
    
    _os << this->createStartTag( "fmk" ) << std::endl;
    this->fmk->store ( _os, _format );
    _os << this->createEndTag( "fmk" ) << std::endl;
    
    _os << this->createStartTag( "q" ) << std::endl;
    if ( q != NULL )
    {
      _os << "NOTNULL" << std::endl;
      this->q->store ( _os, _format );
    }
    else
    {
      _os << "NULL" << std::endl;
    }
    _os << this->createEndTag( "q" ) << std::endl;
    
    _os << this->createStartTag( "parameterUpperBound" ) << std::endl;
    _os << this->d_parameterUpperBound << std::endl;
    _os << this->createEndTag( "parameterUpperBound" ) << std::endl;
    
    
    _os << this->createStartTag( "parameterLowerBound" ) << std::endl;
    _os << this->d_parameterLowerBound << std::endl;
    _os << this->createEndTag( "parameterLowerBound" ) << std::endl;    
    
    _os << this->createStartTag( "pf" ) << std::endl;
    this->pf->store(_os, _format);
    _os << this->createEndTag( "pf" ) << std::endl;     
    
    _os << this->createStartTag( "precomputedA" ) << std::endl;
    _os << "size: " << this->precomputedA.size() << std::endl;
    std::map< uint, PrecomputedType >::const_iterator preCompIt = this->precomputedA.begin();
    for ( uint i = 0; i < this->precomputedA.size(); i++ )
    {
      _os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( _os, _format );
      preCompIt++;
    }    
    _os << this->createEndTag( "precomputedA" ) << std::endl;  
    
    
    _os << this->createStartTag( "precomputedB" ) << std::endl;
    _os << "size: " << this->precomputedB.size() << std::endl;
    preCompIt = this->precomputedB.begin();
    for ( uint i = 0; i < this->precomputedB.size(); i++ )
    {
      _os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( _os, _format );
      preCompIt++;
    }    
    _os << this->createEndTag( "precomputedB" ) << std::endl; 
      
    
    
    _os << this->createStartTag( "precomputedT" ) << std::endl;
    _os << "size: " << this->precomputedT.size() << std::endl;
    if ( this->precomputedT.size() > 0 )
    {
      int sizeOfLUT ( 0 );
      if ( q != NULL )
        sizeOfLUT = q->getNumberOfBins() * this->fmk->get_d();
      _os << "SizeOfLUTs: " << sizeOfLUT << std::endl;      
      for ( std::map< uint, double * >::const_iterator it = this->precomputedT.begin(); it != this->precomputedT.end(); it++ )
      {
        _os << "index: " << it->first << std::endl;
        for ( int i = 0; i < sizeOfLUT; i++ )
        {
          _os << ( it->second ) [i] << " ";
        }
        _os << std::endl;
      }
    } 
    _os << this->createEndTag( "precomputedT" ) << std::endl;
    
    
    _os << this->createStartTag( "labels" ) << std::endl;
    _os << this->labels << std::endl;
    _os << this->createEndTag( "labels" ) << std::endl;  
    
    //store the class numbers for binary settings (if mc-settings, these values will be negative by default)
    _os << this->createStartTag( "binaryLabelPositive" ) << std::endl;
    _os << this->i_binaryLabelPositive << std::endl;
    _os << this->createEndTag( "binaryLabelPositive" ) << std::endl; 
    
    _os << this->createStartTag( "binaryLabelNegative" ) << std::endl;
    _os << this->i_binaryLabelNegative << std::endl;
    _os << this->createEndTag( "binaryLabelNegative" ) << std::endl;
    
    _os << this->createStartTag( "knownClasses" ) << std::endl;
    _os << "size: " << this->knownClasses.size() << std::endl;

    for ( std::set< uint >::const_iterator itKnownClasses = this->knownClasses.begin();
          itKnownClasses != this->knownClasses.end();
          itKnownClasses++
        )
    {
      _os << *itKnownClasses << " " << std::endl;
    }   
    _os << this->createEndTag( "knownClasses" ) << std::endl;    
    
    _os << this->createStartTag( "ikmsum" ) << std::endl;
    for ( int j = 0; j < ikmsum->getNumberOfModels() - 1; j++ )
    {
      ( ikmsum->getModel ( j ) )->store ( _os, _format );
    }
    _os << this->createEndTag( "ikmsum" ) << std::endl;    
    
    //////////////////////////////////////////////
    //           Iterative Linear Solver        //
    //////////////////////////////////////////////     
    
    _os << this->createStartTag( "linsolver" ) << std::endl;
    //TODO linsolver 
    _os << this->createEndTag( "linsolver" ) << std::endl;      
       

    _os << this->createStartTag( "ils_max_iterations" ) << std::endl;
    _os << this->ils_max_iterations << std::endl;
    _os << this->createEndTag( "ils_max_iterations" ) << std::endl;  
    
    /////////////////////////////////////
    // optimization related parameters //
    /////////////////////////////////////    
    
    _os << this->createStartTag( "optimizationMethod" ) << std::endl;
    _os << this->optimizationMethod << std::endl;
    _os << this->createEndTag( "optimizationMethod" ) << std::endl; 
    
    _os << this->createStartTag( "optimizeNoise" ) << std::endl;
    _os << this->optimizeNoise << std::endl;
    _os << this->createEndTag( "optimizeNoise" ) << std::endl;
    
    _os << this->createStartTag( "parameterStepSize" ) << std::endl;
    _os << this->parameterStepSize << std::endl;
    _os << this->createEndTag( "parameterStepSize" ) << std::endl;
    
    _os << this->createStartTag( "downhillSimplexMaxIterations" ) << std::endl;
    _os << this->downhillSimplexMaxIterations << std::endl;
    _os << this->createEndTag( "downhillSimplexMaxIterations" ) << std::endl;
    
    
    _os << this->createStartTag( "downhillSimplexTimeLimit" ) << std::endl;
    _os << this->downhillSimplexTimeLimit << std::endl;
    _os << this->createEndTag( "downhillSimplexTimeLimit" ) << std::endl;
    
    
    _os << this->createStartTag( "downhillSimplexParamTol" ) << std::endl;
    _os << this->downhillSimplexParamTol << std::endl;
    _os << this->createEndTag( "downhillSimplexParamTol" ) << std::endl;
    
    //////////////////////////////////////////////
    // likelihood computation related variables //
    //////////////////////////////////////////////     
    
    _os << this->createStartTag( "verifyApproximation" ) << std::endl;
    _os << this->verifyApproximation << std::endl;
    _os << this->createEndTag( "verifyApproximation" ) << std::endl;    
    
    _os << this->createStartTag( "eig" ) << std::endl;
    //TODO eig 
    _os << this->createEndTag( "eig" ) << std::endl;    
       
    
    _os << this->createStartTag( "nrOfEigenvaluesToConsider" ) << std::endl;
    _os << this->nrOfEigenvaluesToConsider << std::endl;
    _os << this->createEndTag( "nrOfEigenvaluesToConsider" ) << std::endl;
    
    _os << this->createStartTag( "eigenMax" ) << std::endl;
    _os << this->eigenMax << std::endl;
    _os << this->createEndTag( "eigenMax" ) << std::endl;

    _os << this->createStartTag( "eigenMaxVectors" ) << std::endl;
    _os << this->eigenMaxVectors << std::endl;
    _os << this->createEndTag( "eigenMaxVectors" ) << std::endl;
    
    
    ////////////////////////////////////////////
    // variance computation related variables //
    ////////////////////////////////////////////    
    
    _os << this->createStartTag( "nrOfEigenvaluesToConsiderForVarApprox" ) << std::endl;
    _os << this->nrOfEigenvaluesToConsiderForVarApprox << std::endl;
    _os << this->createEndTag( "nrOfEigenvaluesToConsiderForVarApprox" ) << std::endl;
    
    _os << this->createStartTag( "precomputedAForVarEst" ) << std::endl;
    _os << precomputedAForVarEst.size() << std::endl;
    
    if ( this->precomputedAForVarEst.size() > 0)
    {
      this->precomputedAForVarEst.store ( _os, _format );
      _os << std::endl; 
    }
    _os << this->createEndTag( "precomputedAForVarEst" ) << std::endl;
    
    
    _os << this->createStartTag( "precomputedTForVarEst" ) << std::endl;
    if ( this->precomputedTForVarEst != NULL )
    {
      _os << "NOTNULL" << std::endl;
      int sizeOfLUT ( 0 );
      if ( q != NULL )
        sizeOfLUT = q->getNumberOfBins() * this->fmk->get_d();
      
      _os << sizeOfLUT << std::endl;
      for ( int i = 0; i < sizeOfLUT; i++ )
      {
        _os << this->precomputedTForVarEst[i] << " ";
      }
      _os << std::endl;
    }
    else
    {
      _os << "NULL" << std::endl;
    }
    _os << this->createEndTag( "precomputedTForVarEst" ) << std::endl;    
    
    /////////////////////////////////////////////////////
    // online / incremental learning related variables //
    /////////////////////////////////////////////////////    
    _os << this->createStartTag( "b_usePreviousAlphas" ) << std::endl;
    _os << this->b_usePreviousAlphas << std::endl;
    _os << this->createEndTag( "b_usePreviousAlphas" ) << std::endl;    
    
    _os << this->createStartTag( "previousAlphas" ) << std::endl;
    _os << "size: " << this->previousAlphas.size() << std::endl;
    std::map< uint, NICE::Vector >::const_iterator prevAlphaIt = this->previousAlphas.begin();
    for ( uint i = 0; i < this->previousAlphas.size(); i++ )
    {
      _os << prevAlphaIt->first << std::endl;
      _os << prevAlphaIt->second << std::endl;
      prevAlphaIt++;
    }
    _os << this->createEndTag( "previousAlphas" ) << std::endl;

    
    
    // done
    _os << this->createEndTag( "FMKGPHyperparameterOptimization" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void FMKGPHyperparameterOptimization::clear ( ) {};

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void FMKGPHyperparameterOptimization::addExample( const NICE::SparseVector * example, 
                                                  const double & label, 
                                                  const bool & performOptimizationAfterIncrement
                                                )
{
  if ( this->b_verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addExample --- " << std::endl;  
  
  NICE::Timer t;
  t.start();  

  std::set< uint > newClasses;
  
  this->labels.append ( label );
  //have we seen this class already?
  if ( !this->b_performRegression && ( this->knownClasses.find( label ) == this->knownClasses.end() ) )
  {
    this->knownClasses.insert( label );
    newClasses.insert( label );
  }
  
  // If we currently have been in a binary setting, we now have to take care
  // that we also compute an alpha vector for the second class, which previously 
  // could be dealt with implicitely.
  // Therefore, we insert its label here...
  if ( (newClasses.size() > 0 ) && ( (this->knownClasses.size() - newClasses.size() ) == 2 ) )
    newClasses.insert( this->i_binaryLabelNegative );    

  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  NICE::Timer tFmk;
  tFmk.start();
  this->fmk->addExample ( example, pf );
  tFmk.stop();
  if ( this->b_verboseTime)
    std::cerr << "Time used for adding the data to the fmk object: " << tFmk.getLast() << std::endl;

  
  // add examples to all implicite kernel matrices we currently use
  this->ikmsum->addExample ( example, label, performOptimizationAfterIncrement );
  
  
  // update the corresponding matrices A, B and lookup tables T  
  // optional: do the optimization again using the previously known solutions as initialization
  this->updateAfterIncrement ( newClasses, performOptimizationAfterIncrement );
  
  //clean up
  newClasses.clear();
  
  t.stop();  

  NICE::ResourceStatistics rs;
  
  std::cerr << "Time used for re-learning: " << t.getLast() << std::endl;
  
  long maxMemory;
  rs.getMaximumMemory ( maxMemory );
  
  if ( this->b_verbose )
    std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;   
  
  if ( this->b_verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addExample done --- " << std::endl;   
}

void FMKGPHyperparameterOptimization::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
                                                           const NICE::Vector & newLabels,
                                                           const bool & performOptimizationAfterIncrement
                                                         )
{
  if ( this->b_verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addMultipleExamples --- " << std::endl;  
  
  NICE::Timer t;
  t.start();  

  std::set< uint > newClasses;
  
  this->labels.append ( newLabels );
  //have we seen this class already?
  if ( !this->b_performRegression)
  {
    for ( NICE::Vector::const_iterator vecIt = newLabels.begin(); 
          vecIt != newLabels.end();
          vecIt++
      )
    {  
      if ( this->knownClasses.find( *vecIt ) == this->knownClasses.end() )
      {
        this->knownClasses.insert( *vecIt );
        newClasses.insert( *vecIt );
      } 
    }

    // If we currently have been in a OCC setting, and only add a single new class
    // we have to take care that are still efficient, i.e., that we solve for alpha
    // only ones, since scores are symmetric in binary cases
    // Therefore, we remove the label of the secodn class from newClasses, to skip
    // alpha computations for this class lateron...
    // 
    // Therefore, we insert its label here...
    if ( (newClasses.size() == 1 ) && ( (this->knownClasses.size() - newClasses.size() ) == 1 ) )
      newClasses.clear();

    // If we currently have been in a binary setting, we now have to take care
    // that we also compute an alpha vector for the second class, which previously 
    // could be dealt with implicitely.
    // Therefore, we insert its label here...
    if ( (newClasses.size() > 0 ) && ( (this->knownClasses.size() - newClasses.size() ) == 2 ) )
      newClasses.insert( this->i_binaryLabelNegative );      
      
  }
  // in a regression setting, we do not have to remember any "class labels"
  else{}
  
  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  NICE::Timer tFmk;
  tFmk.start();
  this->fmk->addMultipleExamples ( newExamples, pf );
  tFmk.stop();
  if ( this->b_verboseTime)
    std::cerr << "Time used for adding the data to the fmk object: " << tFmk.getLast() << std::endl;
  
  // add examples to all implicite kernel matrices we currently use
  this->ikmsum->addMultipleExamples ( newExamples, newLabels, performOptimizationAfterIncrement );
  
  // update the corresponding matrices A, B and lookup tables T  
  // optional: do the optimization again using the previously known solutions as initialization
  this->updateAfterIncrement ( newClasses, performOptimizationAfterIncrement );

  //clean up
  newClasses.clear();
  
  t.stop();  

  NICE::ResourceStatistics rs;
  
  std::cerr << "Time used for re-learning: " << t.getLast() << std::endl;
  
  long maxMemory;
  rs.getMaximumMemory ( maxMemory );
  
  if ( this->b_verbose )
    std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;  
  
  if ( this->b_verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addMultipleExamples done --- " << std::endl;    
}
