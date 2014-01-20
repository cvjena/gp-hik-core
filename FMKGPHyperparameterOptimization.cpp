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
#include <core/vector/Eigen.h>
// 
#include <core/optimization/blackbox/DownhillSimplexOptimizer.h>


// gp-hik-core includes
#include "FMKGPHyperparameterOptimization.h"
#include "FastMinKernel.h"
#include "GMHIKernel.h"
#include "IKMNoise.h"



using namespace NICE;
using namespace std;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

void FMKGPHyperparameterOptimization::updateAfterIncrement ( 
      const std::set < int > newClasses,
      const bool & performOptimizationAfterIncrement )
{
  if ( this->fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  std::map<int, NICE::Vector> binaryLabels;
  std::set<int> classesToUse;
  //TODO this could be made faster when storing the previous binary label vectors...
  
  if ( this->b_performRegression )
  {
    // for regression, we are not interested in regression scores, rather than in any "label" 
    int regressionLabel ( 1 );  
    binaryLabels.insert ( std::pair< int, NICE::Vector> ( regressionLabel, this->labels ) );
  }
  else
    this->prepareBinaryLabels ( binaryLabels, this->labels , classesToUse );
  
  if ( this->verbose )
    std::cerr << "labels.size() after increment: " << this->labels.size() << std::endl;

  NICE::Timer t1;

  NICE::GPLikelihoodApprox * gplike;
  uint parameterVectorSize;

  t1.start();
  this->setupGPLikelihoodApprox ( gplike, binaryLabels, parameterVectorSize );
  t1.stop();
  if ( this->verboseTime )
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  t1.start();
  if ( this->b_usePreviousAlphas && ( this->previousAlphas.size() > 0) )
  {
    //We initialize it with the same values as we use in GPLikelihoodApprox in batch training
    //default in GPLikelihoodApprox for the first time:
    // alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );
    double factor ( 1.0 / this->eigenMax[0] );
    
    std::map<int, NICE::Vector>::const_iterator binaryLabelsIt = binaryLabels.begin();
    
    for ( std::map<int, NICE::Vector>::iterator prevAlphaIt = this->previousAlphas.begin();
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
    for (std::set<int>::const_iterator newClIt =  newClasses.begin(); newClIt != newClasses.end(); newClIt++)
    {      
      NICE::Vector alphaVec = (binaryLabels[*newClIt] * factor ); //see GPLikelihoodApprox for an explanation
      previousAlphas.insert( std::pair<int, NICE::Vector>(*newClIt, alphaVec) );
    }      

    gplike->setInitialAlphaGuess ( &previousAlphas );
  }
  else
  {
    //if we do not use previous alphas, we do not have to set up anything here
    gplike->setInitialAlphaGuess ( NULL );
  }
    
  t1.stop();
  if ( this->verboseTime )
    std::cerr << "Time used for setting up the alpha-objects: " << t1.getLast() << std::endl;

  if ( this->verbose ) 
    std::cerr << "update Eigendecomposition " << std::endl;
  
  t1.start();
  // we compute all needed eigenvectors for standard classification and variance prediction at ones.
  // nrOfEigenvaluesToConsiderForVarApprox should NOT be larger than 1 if a method different than approximate_fine is used!
  this->updateEigenDecomposition(  std::max ( this->nrOfEigenvaluesToConsider, this->nrOfEigenvaluesToConsiderForVarApprox) );
  t1.stop();
  if ( this->verboseTime )
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  
  //////////////////////  //////////////////////
  //   RE-RUN THE OPTIMIZATION, IF DESIRED    //
  //////////////////////  //////////////////////    
    
  if ( this->verbose )
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
    
  if ( this->verbose )
    std::cerr << "perform optimization after increment " << std::endl;
   
  int optimizationMethodTmpCopy;
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
  if ( this->verboseTime )
    std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

  if ( this->verbose )
    std::cerr << "Preparing after retraining for classification ..." << std::endl;

  t1.start();
  this->transformFeaturesWithOptimalParameters ( *gplike, parameterVectorSize );
  t1.stop();
  if ( this->verboseTime)
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
  if ( this->verboseTime )
    std::cerr << "Time used for setting up the A'nB -objects: " << t1.getLast() << std::endl;

  //don't waste memory
  delete gplike;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization( const bool & b_performRegression )
{
  // initialize pointer variables
  pf = NULL;
  eig = NULL;
  linsolver = NULL;
  fmk = NULL;
  q = NULL;
  precomputedTForVarEst = NULL;
  ikmsum  = NULL;
  
  // initialize boolean flags
  verbose = false;
  verboseTime = false;
  debug = false;
  
  //stupid unneeded default values
  binaryLabelPositive = -1;
  binaryLabelNegative = -2;
  
  this->b_usePreviousAlphas = false;
  this->b_performRegression = b_performRegression;
}

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization ( const Config *_conf, ParameterizedFunction *_pf, FastMinKernel *_fmk, const string & _confSection )
{
   // initialize pointer variables
  pf = NULL;
  eig = NULL;
  linsolver = NULL;
  fmk = NULL;
  q = NULL;
  precomputedTForVarEst = NULL;
  ikmsum = NULL;
  
  //stupid unneeded default values
  binaryLabelPositive = -1;
  binaryLabelNegative = -2;  
  knownClasses.clear();
  
  if ( _fmk == NULL )
    this->initialize ( _conf, _pf ); //then the confSection is also the default value
  else
    this->initialize ( _conf, _pf, _fmk, _confSection );
}

FMKGPHyperparameterOptimization::~FMKGPHyperparameterOptimization()
{
  //pf will delete from outer program
  if ( this->eig != NULL )
    delete this->eig;
  if ( this->linsolver != NULL )
    delete this->linsolver;
  if ( this->fmk != NULL )
    delete this->fmk;
  if ( this->q != NULL )
    delete this->q;

  for ( uint i = 0 ; i < precomputedT.size(); i++ )
    delete [] ( precomputedT[i] );

  if ( precomputedTForVarEst != NULL )
    delete precomputedTForVarEst;

  if ( ikmsum != NULL )
    delete ikmsum;
}

void FMKGPHyperparameterOptimization::initialize ( const Config *_conf, ParameterizedFunction *_pf, FastMinKernel *_fmk, const std::string & _confSection )
{

  if ( _fmk != NULL )
  {
    if ( this->fmk != NULL )
    {
      delete this->fmk;
      fmk = NULL;
    }    
    this->fmk = _fmk;
  }
  
  this->pf = _pf;
 
  
  this->verbose = _conf->gB ( _confSection, "verbose", false );
  this->verboseTime = _conf->gB ( _confSection, "verboseTime", false );
  this->debug = _conf->gB ( _confSection, "debug", false );

  if ( verbose )
  {  
    std::cerr << "------------" << std::endl;
    std::cerr << "|  set-up  |" << std::endl;
    std::cerr << "------------" << std::endl;
  }
  
  this->b_performRegression = _conf->gB ( _confSection, "b_performRegression", false );


  // this->eig = new EigValuesTRLAN();
  // My time measurements show that both methods use equal time, a comparision
  // of their numerical performance has not been done yet  
  this->eig = new EVArnoldi ( _conf->gB ( _confSection, "eig_verbose", false ) /* verbose flag */, 10 );


  this->parameterUpperBound = _conf->gD ( _confSection, "parameter_upper_bound", 2.5 );
  this->parameterLowerBound = _conf->gD ( _confSection, "parameter_lower_bound", 1.0 );
  this->parameterStepSize = _conf->gD ( _confSection, "parameter_step_size", 0.1 );

  this->verifyApproximation = _conf->gB ( _confSection, "verify_approximation", false );
  this->nrOfEigenvaluesToConsider = _conf->gI ( _confSection, "nrOfEigenvaluesToConsider", 1 );
  this->nrOfEigenvaluesToConsiderForVarApprox = _conf->gI ( _confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 );


  bool useQuantization = _conf->gB ( _confSection, "use_quantization", false );
  
  if ( verbose ) 
  {
    std::cerr << "_confSection: " << _confSection << std::endl;
    std::cerr << "use_quantization: " << useQuantization << std::endl;
  }
  
  if ( _conf->gB ( _confSection, "use_quantization", false ) ) {
    int numBins = _conf->gI ( _confSection, "num_bins", 100 );
    if ( verbose )
      std::cerr << "FMKGPHyperparameterOptimization: quantization initialized with " << numBins << " bins." << std::endl;
    this->q = new Quantization ( numBins );
  } else {
    this->q = NULL;
  }

  bool ils_verbose = _conf->gB ( _confSection, "ils_verbose", false );
  ils_max_iterations = _conf->gI ( _confSection, "ils_max_iterations", 1000 );
  if ( verbose )
    std::cerr << "FMKGPHyperparameterOptimization: maximum number of iterations is " << ils_max_iterations << std::endl;

  double ils_min_delta = _conf->gD ( _confSection, "ils_min_delta", 1e-7 );
  double ils_min_residual = _conf->gD ( _confSection, "ils_min_residual", 1e-7/*1e-2 */ );

  string ils_method = _conf->gS ( _confSection, "ils_method", "CG" );
  if ( ils_method.compare ( "CG" ) == 0 )
  {
    if ( verbose )
      std::cerr << "We use CG with " << ils_max_iterations << " iterations, " << ils_min_delta << " as min delta, and " << ils_min_residual << " as min res " << std::endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
    if ( verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients" << std::endl;
  }
  else if ( ils_method.compare ( "CGL" ) == 0 )
  {
    this->linsolver = new ILSConjugateGradientsLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients (Lanczos)" << std::endl;
  }
  else if ( ils_method.compare ( "SYMMLQ" ) == 0 )
  {
    this->linsolver = new ILSSymmLqLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS SYMMLQ" << std::endl;
  }
  else if ( ils_method.compare ( "MINRES" ) == 0 )
  {
    this->linsolver = new ILSMinResLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      std::cerr << "FMKGPHyperparameterOptimization: using ILS MINRES" << std::endl;
  }
  else
  {
    std::cerr << "FMKGPHyperparameterOptimization: " << _confSection << ":ils_method (" << ils_method << ") does not match any type (CG,CGL,SYMMLQ,MINRES), I will use CG" << std::endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
  }
  
  string optimizationMethod_s = _conf->gS ( _confSection, "optimization_method", "greedy" );
  if ( optimizationMethod_s == "greedy" )
    optimizationMethod = OPT_GREEDY;
  else if ( optimizationMethod_s == "downhillsimplex" )
    optimizationMethod = OPT_DOWNHILLSIMPLEX;
  else if ( optimizationMethod_s == "none" )
    optimizationMethod = OPT_NONE;
  else
    fthrow ( Exception, "Optimization method " << optimizationMethod_s << " is not known." );

  if ( verbose )
    std::cerr << "Using optimization method: " << optimizationMethod_s << std::endl;

  downhillSimplexMaxIterations = _conf->gI ( _confSection, "downhillsimplex_max_iterations", 20 );
  // do not run longer than a day :)
  downhillSimplexTimeLimit = _conf->gD ( _confSection, "downhillsimplex_time_limit", 24 * 60 * 60 );
  downhillSimplexParamTol = _conf->gD ( _confSection, "downhillsimplex_delta", 0.01 );

  optimizeNoise = _conf->gB ( _confSection, "optimize_noise", false );
  if ( verbose )
    std::cerr << "Optimize noise: " << ( optimizeNoise ? "on" : "off" ) << std::endl;
  
  this->b_usePreviousAlphas = _conf->gB ( _confSection, "b_usePreviousAlphas", true );
  
  if ( verbose )
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
  parameterUpperBound = _parameterUpperBound;
}
void FMKGPHyperparameterOptimization::setParameterLowerBound ( const double & _parameterLowerBound )
{
  parameterLowerBound = _parameterLowerBound;
}

std::set<int> FMKGPHyperparameterOptimization::getKnownClassNumbers ( ) const
{
  return this->knownClasses;
}



///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

inline void FMKGPHyperparameterOptimization::setupGPLikelihoodApprox ( GPLikelihoodApprox * & gplike, const std::map<int, NICE::Vector> & binaryLabels, uint & parameterVectorSize )
{
  gplike = new GPLikelihoodApprox ( binaryLabels, ikmsum, linsolver, eig, verifyApproximation, nrOfEigenvaluesToConsider );
  gplike->setDebug( debug );
  gplike->setVerbose( verbose );
  parameterVectorSize = ikmsum->getNumParameters();
}

void FMKGPHyperparameterOptimization::updateEigenDecomposition( const int & i_noEigenValues )
{
  //compute the largest eigenvalue of K + noise   
  try 
  {
    eig->getEigenvalues ( *ikmsum,  eigenMax, eigenMaxVectors, i_noEigenValues  );
  }
  catch ( char const* exceptionMsg)
  {
    std::cerr << exceptionMsg << std::endl;
    throw("Problem in calculating Eigendecomposition of kernel matrix. Abort program...");
  }
  
  //NOTE EigenValue computation extracts EV and EW per default in decreasing order.
  
}

void FMKGPHyperparameterOptimization::performOptimization ( GPLikelihoodApprox & gplike, const uint & parameterVectorSize )
{
  if (verbose)
    std::cerr << "perform optimization" << std::endl;
    
  if ( optimizationMethod == OPT_GREEDY )
  {
    if ( verbose )    
      std::cerr << "OPT_GREEDY!!! " << std::endl;
    
    // simple greedy strategy
    if ( ikmsum->getNumParameters() != 1 )
      fthrow ( Exception, "Reduce size of the parameter vector or use downhill simplex!" );

    NICE::Vector lB = ikmsum->getParameterLowerBounds();
    NICE::Vector uB = ikmsum->getParameterUpperBounds();
    
    if ( verbose )
      std::cerr << "lower bound " << lB << " upper bound " << uB << " parameterStepSize: " << parameterStepSize << std::endl;

    
    for ( double mypara = lB[0]; mypara <= uB[0]; mypara += this->parameterStepSize )
    {
      OPTIMIZATION::matrix_type hyperp ( 1, 1, mypara );
      gplike.evaluate ( hyperp );
    }
  }
  else if ( optimizationMethod == OPT_DOWNHILLSIMPLEX )
  {
    //standard as before, normal optimization
    if ( verbose )    
        std::cerr << "DOWNHILLSIMPLEX!!! " << std::endl;

    // downhill simplex strategy
    OPTIMIZATION::DownhillSimplexOptimizer optimizer;

    OPTIMIZATION::matrix_type initialParams ( parameterVectorSize, 1 );

    NICE::Vector currentParameters;
    ikmsum->getParameters ( currentParameters );

    for ( uint i = 0 ; i < parameterVectorSize; i++ )
      initialParams(i,0) = currentParameters[ i ];

    if ( verbose )      
      std::cerr << "Initial parameters: " << initialParams << std::endl;

    //the scales object does not really matter in the actual implementation of Downhill Simplex
//     OPTIMIZATION::matrix_type scales ( parameterVectorSize, 1);
//     scales.set(1.0);

    OPTIMIZATION::SimpleOptProblem optProblem ( &gplike, initialParams, initialParams /* scales */ );

    optimizer.setMaxNumIter ( true, downhillSimplexMaxIterations );
    optimizer.setTimeLimit ( true, downhillSimplexTimeLimit );
    optimizer.setParamTol ( true, downhillSimplexParamTol );
    
    optimizer.optimizeProb ( optProblem );
  }
  else if ( optimizationMethod == OPT_NONE )
  {
    if ( verbose )
      std::cerr << "NO OPTIMIZATION!!! " << std::endl;

    // without optimization
    if ( optimizeNoise )
      fthrow ( Exception, "Deactivate optimize_noise!" );
    
    if ( verbose )
      std::cerr << "Optimization is deactivated!" << std::endl;
    
    double value (1.0);
    if ( this->parameterLowerBound == this->parameterUpperBound)
      value = this->parameterLowerBound;

    pf->setParameterLowerBounds ( NICE::Vector ( 1, value ) );
    pf->setParameterUpperBounds ( NICE::Vector ( 1, value ) );

    // we use the standard value
    OPTIMIZATION::matrix_type hyperp ( 1, 1, value );
    gplike.setParameterLowerBound ( value );
    gplike.setParameterUpperBound ( value );
    //we do not need to compute the likelihood here - we are only interested in directly obtaining alpha vectors
    gplike.computeAlphaDirect( hyperp, eigenMax );
  }

  if ( verbose )
  {
    std::cerr << "Optimal hyperparameter was: " << gplike.getBestParameters() << std::endl;
  }
}

void FMKGPHyperparameterOptimization::transformFeaturesWithOptimalParameters ( const GPLikelihoodApprox & gplike, const uint & parameterVectorSize )
{
  // transform all features with the currently "optimal" parameter
  ikmsum->setParameters ( gplike.getBestParameters() );    
}

void FMKGPHyperparameterOptimization::computeMatricesAndLUTs ( const GPLikelihoodApprox & gplike )
{
  precomputedA.clear();
  precomputedB.clear();

  for ( std::map<int, NICE::Vector>::const_iterator i = gplike.getBestAlphas().begin(); i != gplike.getBestAlphas().end(); i++ )
  {
    PrecomputedType A;
    PrecomputedType B;

    fmk->hik_prepare_alpha_multiplications ( i->second, A, B );
    A.setIoUntilEndOfFile ( false );
    B.setIoUntilEndOfFile ( false );

    precomputedA[ i->first ] = A;
    precomputedB[ i->first ] = B;

    if ( q != NULL )
    {
      double *T = fmk->hik_prepare_alpha_multiplications_fast ( A, B, *q, pf );
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
    this->previousAlphas = gplike.getBestAlphas();

}

#ifdef NICE_USELIB_MATIO
void FMKGPHyperparameterOptimization::optimizeBinary ( const sparse_t & data, const NICE::Vector & yl, const std::set<int> & positives, const std::set<int> & negatives, double noise )
{
  std::map<int, int> examples;
  NICE::Vector y ( yl.size() );
  int ind = 0;
  for ( uint i = 0 ; i < yl.size(); i++ )
  {
    if ( positives.find ( i ) != positives.end() ) {
      y[ examples.size() ] = 1.0;
      examples.insert ( pair<int, int> ( i, ind ) );
      ind++;
    } else if ( negatives.find ( i ) != negatives.end() ) {
      y[ examples.size() ] = -1.0;
      examples.insert ( pair<int, int> ( i, ind ) );
      ind++;
    }
  }
  y.resize ( examples.size() );
  std::cerr << "Examples: " << examples.size() << std::endl;

  optimize ( data, y, examples, noise );
}


void FMKGPHyperparameterOptimization::optimize ( const sparse_t & data, const NICE::Vector & y, const std::map<int, int> & examples, double noise )
{
  NICE::Timer t;
  t.start();
  std::cerr << "Initializing data structure ..." << std::endl;
  if ( fmk != NULL ) delete fmk;
  fmk = new FastMinKernel ( data, noise, examples );
  t.stop();
  if (verboseTime)
    std::cerr << "Time used for initializing the FastMinKernel structure: " << t.getLast() << std::endl;
  
  optimize ( y );
}
#endif

int FMKGPHyperparameterOptimization::prepareBinaryLabels ( std::map<int, NICE::Vector> & binaryLabels, const NICE::Vector & y , std::set<int> & myClasses )
{
  myClasses.clear();
  
  // determine which classes we have in our label vector
  // -> MATLAB: myClasses = unique(y);
  for ( NICE::Vector::const_iterator it = y.begin(); it != y.end(); it++ )
  {
    if ( myClasses.find ( *it ) == myClasses.end() )
    {
      myClasses.insert ( *it );
    }
  }

  //count how many different classes appear in our data
  int nrOfClasses = myClasses.size();

  binaryLabels.clear();
  //compute the corresponding binary label vectors
  if ( nrOfClasses > 2 )
  {
    //resize every labelVector and set all entries to -1.0
    for ( set<int>::const_iterator k = myClasses.begin(); k != myClasses.end(); k++ )
    {
      binaryLabels[ *k ].resize ( y.size() );
      binaryLabels[ *k ].set ( -1.0 );
    }

    // now look on every example and set the entry of its corresponding label vector to 1.0
    // proper existance should not be a problem
    for ( int i = 0 ; i < ( int ) y.size(); i++ )
      binaryLabels[ y[i] ][i] = 1.0;
  }
  else if ( nrOfClasses == 2 )
  {
    //binary setting -- prepare two binary label vectors with opposite signs
    NICE::Vector yb ( y );

    binaryLabelNegative = *(myClasses.begin());
    std::set<int>::const_iterator classIt = myClasses.begin(); classIt++;
    binaryLabelPositive = *classIt;
    
    if ( verbose )
      std::cerr << "positiveClass : " << binaryLabelPositive << " negativeClass: " << binaryLabelNegative << std::endl;

    for ( uint i = 0 ; i < yb.size() ; i++ )
      yb[i] = ( y[i] == binaryLabelNegative ) ? -1.0 : 1.0;
    
    binaryLabels[ binaryLabelPositive ] = yb;
        
    //we do NOT do real binary computation, but an implicite one with only a single object  
    nrOfClasses--;  
  }
  else //OCC setting
  {
    //we set the labels to 1, independent of the previously given class number
    //however, the original class numbers are stored and returned in classification
    NICE::Vector yOne ( y.size(), 1 );
    
    binaryLabels[ *(myClasses.begin()) ] = yOne;
    
    //we have to indicate, that we are in an OCC setting
    nrOfClasses--;
  }

  return nrOfClasses;
}



void FMKGPHyperparameterOptimization::optimize ( const NICE::Vector & y )
{
  if ( fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  this->labels  = y;
  
  std::map<int, NICE::Vector> binaryLabels;
  
  if ( this->b_performRegression )
  {
    // for regression, we are not interested in regression scores, rather than in any "label" 
    int regressionLabel ( 1 );  
    binaryLabels.insert ( std::pair< int, NICE::Vector> ( regressionLabel, y ) );
    this->knownClasses.clear();
    this->knownClasses.insert ( regressionLabel );
  }
  else
  {
    this->prepareBinaryLabels ( binaryLabels, y , knownClasses );    
  }
  
  //now call the main function :)
  this->optimize(binaryLabels);
}

  
void FMKGPHyperparameterOptimization::optimize ( std::map<int, NICE::Vector> & binaryLabels )
{
  Timer t;
  t.start();
  
  //how many different classes do we have right now?
  int nrOfClasses = binaryLabels.size();
  
  if (verbose)
  {
    std::cerr << "Initial noise level: " << fmk->getNoise() << std::endl;

    std::cerr << "Number of classes (=1 means we have a binary setting):" << nrOfClasses << std::endl;
    std::cerr << "Effective number of classes (neglecting classes without positive examples): " << knownClasses.size() << std::endl;
  }

  // combine standard model and noise model

  Timer t1;

  t1.start();
  //setup the kernel combination
  ikmsum = new IKMLinearCombination ();

  if ( verbose )
  {
    std::cerr << "binaryLabels.size(): " << binaryLabels.size() << std::endl;
  }

  //First model: noise
  ikmsum->addModel ( new IKMNoise ( fmk->get_n(), fmk->getNoise(), optimizeNoise ) );
  
  // set pretty low built-in noise, because we explicitely add the noise with the IKMNoise
  fmk->setNoise ( 0.0 );

  ikmsum->addModel ( new GMHIKernel ( fmk, pf, NULL /* no quantization */ ) );

  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the ikm-objects: " << t1.getLast() << std::endl;

  GPLikelihoodApprox * gplike;
  uint parameterVectorSize;

  t1.start();
  this->setupGPLikelihoodApprox ( gplike, binaryLabels, parameterVectorSize );
  t1.stop();
    
  if (verboseTime)
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  if (verbose)
  {
    std::cerr << "parameterVectorSize: " << parameterVectorSize << std::endl;
  }

  t1.start();
  // we compute all needed eigenvectors for standard classification and variance prediction at ones.
  // nrOfEigenvaluesToConsiderForVarApprox should NOT be larger than 1 if a method different than approximate_fine is used!
  this->updateEigenDecomposition(  std::max ( this->nrOfEigenvaluesToConsider, this->nrOfEigenvaluesToConsiderForVarApprox) );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  if ( verbose )
    std::cerr << "resulting eigenvalues for first class: " << eigenMax[0] << std::endl;

  t1.start();
  this->performOptimization ( *gplike, parameterVectorSize );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

  if ( verbose )
    std::cerr << "Preparing classification ..." << std::endl;

  t1.start();
  this->transformFeaturesWithOptimalParameters ( *gplike, parameterVectorSize );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;

  t1.start();
  this->computeMatricesAndLUTs ( *gplike );
  t1.stop();
  if (verboseTime)
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
  fmk->hikPrepareKVNApproximation ( AVar );

  precomputedAForVarEst = AVar;
  precomputedAForVarEst.setIoUntilEndOfFile ( false );

  if ( q != NULL )
  {   
    double *T = fmk->hikPrepareLookupTableForKVNApproximation ( *q, pf );
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

int FMKGPHyperparameterOptimization::classify ( const NICE::SparseVector & xstar, NICE::SparseVector & scores ) const
{
  // loop through all classes
  if ( precomputedA.size() == 0 )
  {
    fthrow ( Exception, "The precomputation vector is zero...have you trained this classifier?" );
  }

  uint maxClassNo = 0;
  for ( std::map<int, PrecomputedType>::const_iterator i = precomputedA.begin() ; i != precomputedA.end(); i++ )
  {
    uint classno = i->first;
    maxClassNo = std::max ( maxClassNo, classno );
    double beta;

    if ( q != NULL ) {
      std::map<int, double *>::const_iterator j = precomputedT.find ( classno );
      double *T = j->second;
      fmk->hik_kernel_sum_fast ( T, *q, xstar, beta );
    } else {
      const PrecomputedType & A = i->second;
      std::map<int, PrecomputedType>::const_iterator j = precomputedB.find ( classno );
      const PrecomputedType & B = j->second;

      // fmk->hik_kernel_sum ( A, B, xstar, beta ); if A, B are of type Matrix
      // Giving the transformation pf as an additional
      // argument is necessary due to the following reason:
      // FeatureMatrixT is sorted according to the original values, therefore,
      // searching for upper and lower bounds ( findFirst... functions ) require original feature
      // values as inputs. However, for calculation we need the transformed features values.

      fmk->hik_kernel_sum ( A, B, xstar, beta, pf );
    }

    scores[ classno ] = beta;
  }
  scores.setDim ( maxClassNo + 1 );
  
  if ( precomputedA.size() > 1 )
  { // multi-class classification
    return scores.maxElement();
  }
  else if ( this->knownClasses.size() == 2 ) // binary setting
  {      
    scores[binaryLabelNegative] = -scores[binaryLabelPositive];     
    return scores[ binaryLabelPositive ] <= 0.0 ? binaryLabelNegative : binaryLabelPositive;
  }
  else //OCC or regression setting
  {
    return 1;
  }
}

int FMKGPHyperparameterOptimization::classify ( const NICE::Vector & xstar, NICE::SparseVector & scores ) const
{
  // loop through all classes
  if ( precomputedA.size() == 0 )
  {
    fthrow ( Exception, "The precomputation vector is zero...have you trained this classifier?" );
  }

  uint maxClassNo = 0;
  for ( std::map<int, PrecomputedType>::const_iterator i = precomputedA.begin() ; i != precomputedA.end(); i++ )
  {
    uint classno = i->first;
    maxClassNo = std::max ( maxClassNo, classno );
    double beta;

    if ( q != NULL ) {
      std::map<int, double *>::const_iterator j = precomputedT.find ( classno );
      double *T = j->second;
      fmk->hik_kernel_sum_fast ( T, *q, xstar, beta );
    } else {
      const PrecomputedType & A = i->second;
      std::map<int, PrecomputedType>::const_iterator j = precomputedB.find ( classno );
      const PrecomputedType & B = j->second;

      // fmk->hik_kernel_sum ( A, B, xstar, beta ); if A, B are of type Matrix
      // Giving the transformation pf as an additional
      // argument is necessary due to the following reason:
      // FeatureMatrixT is sorted according to the original values, therefore,
      // searching for upper and lower bounds ( findFirst... functions ) require original feature
      // values as inputs. However, for calculation we need the transformed features values.

      fmk->hik_kernel_sum ( A, B, xstar, beta, pf );
    }

    scores[ classno ] = beta;
  }
  scores.setDim ( maxClassNo + 1 );
  
  if ( precomputedA.size() > 1 )
  { // multi-class classification
    return scores.maxElement();
  }
  else if ( this->knownClasses.size() == 2 ) // binary setting
  {      
    scores[binaryLabelNegative] = -scores[binaryLabelPositive];     
    return scores[ binaryLabelPositive ] <= 0.0 ? binaryLabelNegative : binaryLabelPositive;
  }
  else //OCC or regression setting
  {
    return 1;
  }
}

    //////////////////////////////////////////
    // variance computation: sparse inputs
    //////////////////////////////////////////

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateRough ( const NICE::SparseVector & x, double & predVariance ) const
{
  // security check!
  if ( pf == NULL )
   fthrow ( Exception, "pf is NULL...have you prepared the uncertainty prediction? Aborting..." );   
  
  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = x.begin(); it != x.end(); it++ )
  {
    kSelf += pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }
  
  // ---------------- compute the approximation of the second term --------------------
  double normKStar;

  if ( q != NULL )
  {
    if ( precomputedTForVarEst == NULL )
    {
      fthrow ( Exception, "The precomputed LUT for uncertainty prediction is NULL...have you prepared the uncertainty prediction? Aborting..." );
    }
    fmk->hikComputeKVNApproximationFast ( precomputedTForVarEst, *q, x, normKStar );
  }
  else
  {
    if ( precomputedAForVarEst.size () == 0 )
    {
      fthrow ( Exception, "The precomputedAForVarEst is empty...have you trained this classifer? Aborting..." );
    }
    fmk->hikComputeKVNApproximation ( precomputedAForVarEst, x, normKStar, pf );
  }

  predVariance = kSelf - ( 1.0 / eigenMax[0] )* normKStar;
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine ( const NICE::SparseVector & x, double & predVariance ) const
{
  // security check!
  if ( eigenMaxVectors.rows() == 0 )
  {
      fthrow ( Exception, "eigenMaxVectors is empty...have you trained this classifer? Aborting..." );
  }    
  
  // ---------------- compute the first term --------------------
//   Timer t;
//   t.start();

  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = x.begin(); it != x.end(); it++ )
  {
    kSelf += pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }
  // ---------------- compute the approximation of the second term --------------------
//    t.stop();  
//   std::cerr << "ApproxFine -- time for first term: "  << t.getLast()  << std::endl;

//   t.start();
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );
/*  t.stop();
  std::cerr << "ApproxFine -- time for kernel vector: "  << t.getLast()  << std::endl;*/
    
//     NICE::Vector multiplicationResults; // will contain nrOfEigenvaluesToConsiderForVarApprox many entries
//     multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    NICE::Vector multiplicationResults( nrOfEigenvaluesToConsiderForVarApprox-1, 0.0 );
    //ok, there seems to be a nasty thing in computing multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    //wherefor it takes aeons...
    //so we compute it by ourselves
    
    
//     for ( uint tmpI = 0; tmpI < kStar.size(); tmpI++)
    NICE::Matrix::const_iterator eigenVecIt = eigenMaxVectors.begin();
//       double kStarI ( kStar[tmpI] );
    for ( int tmpJ = 0; tmpJ < nrOfEigenvaluesToConsiderForVarApprox-1; tmpJ++)
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

    while ( cnt < ( nrOfEigenvaluesToConsiderForVarApprox - 1 ) )
    {
      projectionLength = ( *it );
      currentSecondTerm += ( 1.0 / eigenMax[cnt] ) * pow ( projectionLength, 2 );
      sumOfProjectionLengths += pow ( projectionLength, 2 );
      
      it++;
      cnt++;      
    }
    
    
    double normKStar ( pow ( kStar.normL2 (), 2 ) );

    currentSecondTerm += ( 1.0 / eigenMax[nrOfEigenvaluesToConsiderForVarApprox-1] ) * ( normKStar - sumOfProjectionLengths );
    
    if ( ( normKStar - sumOfProjectionLengths ) < 0 )
    {
      std::cerr << "Attention: normKStar - sumOfProjectionLengths is smaller than zero -- strange!" << std::endl;
    }
    predVariance = kSelf - currentSecondTerm; 
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::SparseVector & x, double & predVariance ) const
{
  // security check!  
  if ( ikmsum->getNumberOfModels() == 0 )
  {
    fthrow ( Exception, "ikmsum is empty... have you trained this classifer? Aborting..." );
  }  
  
    Timer t;
//   t.start();
  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = x.begin(); it != x.end(); it++ )
  {
    kSelf += pf->f ( 0, it->second );
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

  if ( q != NULL )
  {
    if ( precomputedTForVarEst == NULL )
    {
      fthrow ( Exception, "The precomputed LUT for uncertainty prediction is NULL...have you prepared the uncertainty prediction? Aborting..." );
    }
    fmk->hikComputeKVNApproximationFast ( precomputedTForVarEst, *q, x, normKStar );
  }
  else
  {
    if ( precomputedAForVarEst.size () == 0 )
    {
      fthrow ( Exception, "The precomputedAForVarEst is empty...have you trained this classifer? Aborting..." );
    }    
    fmk->hikComputeKVNApproximation ( precomputedAForVarEst, x, normKStar, pf );
  }

  predVariance = kSelf - ( 1.0 / eigenMax[0] )* normKStar;
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine ( const NICE::Vector & x, double & predVariance ) const
{
  // security check!
  if ( eigenMaxVectors.rows() == 0 )
  {
      fthrow ( Exception, "eigenMaxVectors is empty...have you trained this classifer? Aborting..." );
  }  
  
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
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );


    //ok, there seems to be a nasty thing in computing multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    //wherefor it takes aeons...
    //so we compute it by ourselves
//     NICE::Vector multiplicationResults; // will contain nrOfEigenvaluesToConsiderForVarApprox many entries
//     multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );

    NICE::Vector multiplicationResults( nrOfEigenvaluesToConsiderForVarApprox-1, 0.0 );
    NICE::Matrix::const_iterator eigenVecIt = eigenMaxVectors.begin();
    for ( int tmpJ = 0; tmpJ < nrOfEigenvaluesToConsiderForVarApprox-1; tmpJ++)
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

    while ( cnt < ( nrOfEigenvaluesToConsiderForVarApprox - 1 ) )
    {
      projectionLength = ( *it );
      currentSecondTerm += ( 1.0 / eigenMax[cnt] ) * pow ( projectionLength, 2 );
      sumOfProjectionLengths += pow ( projectionLength, 2 );
      
      it++;
      cnt++;      
    }
    
    
    double normKStar ( pow ( kStar.normL2 (), 2 ) );

    currentSecondTerm += ( 1.0 / eigenMax[nrOfEigenvaluesToConsiderForVarApprox-1] ) * ( normKStar - sumOfProjectionLengths );
    

    if ( ( normKStar - sumOfProjectionLengths ) < 0 )
    {
      std::cerr << "Attention: normKStar - sumOfProjectionLengths is smaller than zero -- strange!" << std::endl;
    }
    predVariance = kSelf - currentSecondTerm; 
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::Vector & x, double & predVariance ) const
{
  if ( ikmsum->getNumberOfModels() == 0 )
  {
    fthrow ( Exception, "ikmsum is empty... have you trained this classifer? Aborting..." );
  }  

  // ---------------- compute the first term --------------------
  double kSelf ( 0.0 );
  int dim ( 0 );
  for ( NICE::Vector::const_iterator it = x.begin(); it != x.end(); it++, dim++ )
  {
    kSelf += pf->f ( 0, *it );
    // if weighted dimensions:
    //kSelf += pf->f(dim,*it);
  }

  // ---------------- compute the second term --------------------
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );

  //now run the ILS method
  NICE::Vector diagonalElements;
  ikmsum->getDiagonalElements ( diagonalElements );

  // init simple jacobi pre-conditioning
  ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( linsolver );


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

///////////////////// INTERFACE PERSISTENT /////////////////////
// interface specific methods for store and restore
///////////////////// INTERFACE PERSISTENT ///////////////////// 

void FMKGPHyperparameterOptimization::restore ( std::istream & is, int format )
{
  bool b_restoreVerbose ( false );

#ifdef B_RESTOREVERBOSE
  b_restoreVerbose = true;
#endif

  if ( is.good() )
  {
    if ( b_restoreVerbose ) 
      std::cerr << " in FMKGP restore" << std::endl;
    
    std::string tmp;
    is >> tmp; //class name 
    
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
    
    
    is.precision ( numeric_limits<double>::digits10 + 1 );
       
    
    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "FMKGPHyperparameterOptimization" ) )
      {
        b_endOfBlock = true;
        continue;
      }      
      
      tmp = this->removeStartTag ( tmp );
      
      if ( b_restoreVerbose )
        std::cerr << " currently restore section " << tmp << " in FMKGPHyperparameterOptimization" << std::endl;
      
      if ( tmp.compare("fmk") == 0 )
      {
        fmk = new FastMinKernel;
        fmk->restore( is, format );

        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("precomputedA") == 0 )
      {
        is >> tmp; // size
        int preCompSize ( 0 );
        is >> preCompSize;
        precomputedA.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedA with size: " << preCompSize << std::endl;
        for ( int i = 0; i < preCompSize; i++ )
        {
          int nr;
          is >> nr;
          PrecomputedType pct;
          pct.setIoUntilEndOfFile ( false );
          pct.restore ( is, format );
          precomputedA.insert ( std::pair<int, PrecomputedType> ( nr, pct ) );
        }
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }        
      else if ( tmp.compare("precomputedB") == 0 )
      {
        is >> tmp; // size
        int preCompSize ( 0 );
        is >> preCompSize;
        precomputedB.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedB with size: " << preCompSize << std::endl;
        for ( int i = 0; i < preCompSize; i++ )
        {
          int nr;
          is >> nr;
          PrecomputedType pct;
          pct.setIoUntilEndOfFile ( false );
          pct.restore ( is, format );
          precomputedB.insert ( std::pair<int, PrecomputedType> ( nr, pct ) );
        }    
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }       
      else if ( tmp.compare("precomputedT") == 0 )
      {
        is >> tmp; // size
        int precomputedTSize ( 0 );
        is >> precomputedTSize;

        precomputedT.clear();
        
        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedT with size: " << precomputedTSize << std::endl;

        if ( precomputedTSize > 0 )
        {
          if ( b_restoreVerbose ) 
            std::cerr << " restore precomputedT" << std::endl;
          is >> tmp;
          int sizeOfLUT;
          is >> sizeOfLUT;    
          
          for (int i = 0; i < precomputedTSize; i++)
          {
            is >> tmp;
            int index;
            is >> index;        
            double * array = new double [ sizeOfLUT];
            for ( int i = 0; i < sizeOfLUT; i++ )
            {
              is >> array[i];
            }
            precomputedT.insert ( std::pair<int, double*> ( index, array ) );
          }
        } 
        else
        {
          if ( b_restoreVerbose ) 
            std::cerr << " skip restoring precomputedT" << std::endl;
        }
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("precomputedAForVarEst") == 0 )
      {
        int sizeOfAForVarEst;
        is >> sizeOfAForVarEst;
        
        if ( b_restoreVerbose ) 
          std::cerr << "restore precomputedAForVarEst with size: " << sizeOfAForVarEst << std::endl;
        
        if (sizeOfAForVarEst > 0)
        {
          precomputedAForVarEst.clear();
          
          precomputedAForVarEst.setIoUntilEndOfFile ( false );
          precomputedAForVarEst.restore ( is, format );
        }
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }        
      else if ( tmp.compare("precomputedTForVarEst") == 0 )
      {
        std::string isNull;
        is >> isNull; // NOTNULL or NULL
        if ( b_restoreVerbose ) 
          std::cerr << "content of isNull: " << isNull << std::endl;    
        if (isNull.compare("NOTNULL") == 0)
        {
          if ( b_restoreVerbose ) 
            std::cerr << "restore precomputedTForVarEst" << std::endl;
          
          int sizeOfLUT;
          is >> sizeOfLUT;      
          precomputedTForVarEst = new double [ sizeOfLUT ];
          for ( int i = 0; i < sizeOfLUT; i++ )
          {
            is >> precomputedTForVarEst[i];
          }      
        }
        else
        {
          if ( b_restoreVerbose ) 
            std::cerr << "skip restoring of precomputedTForVarEst" << std::endl;
          if (precomputedTForVarEst != NULL)
            delete precomputedTForVarEst;
        }
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }       
      else if ( tmp.compare("eigenMax") == 0 )
      {
        is >> eigenMax;
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }    
      else if ( tmp.compare("eigenMaxVectors") == 0 )
      {
        is >> eigenMaxVectors;
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }    
      else if ( tmp.compare("ikmsum") == 0 )
      {
        bool b_endOfBlock ( false ) ;
        
        while ( !b_endOfBlock )
        {
          is >> tmp; // start of block 
          
          if ( this->isEndTag( tmp, "ikmsum" ) )
          {
            b_endOfBlock = true;
            continue;
          }      
          
          tmp = this->removeStartTag ( tmp );        
          if ( tmp.compare("IKMNoise") == 0 )
          {
            IKMNoise * ikmnoise = new IKMNoise ();
            ikmnoise->restore ( is, format );
            
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
      else if ( tmp.compare("binaryLabelPositive") == 0 )
      {
        is >> binaryLabelPositive;
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("binaryLabelNegative") == 0 )
      {
        is >> binaryLabelNegative;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("labels") == 0 )
      {
        is >> labels;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("b_usePreviousAlphas") == 0 )
      {
        is >> b_usePreviousAlphas;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if  ( tmp.compare("previousAlphas") == 0 )
      {        
        is >> tmp; // size
        int sizeOfPreviousAlphas ( 0 );
        is >> sizeOfPreviousAlphas;
        previousAlphas.clear();

        if ( b_restoreVerbose ) 
          std::cerr << "restore previousAlphas with size: " << sizeOfPreviousAlphas << std::endl;
        for ( int i = 0; i < sizeOfPreviousAlphas; i++ )
        {
          int classNo;
          is >> classNo;
          NICE::Vector classAlpha;
          is >> classAlpha;
          previousAlphas.insert ( std::pair< int, NICE::Vector > ( classNo, classAlpha ) );
        }
        
        is >> tmp; // end of block 
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

      


    knownClasses.clear();
    
    if ( b_restoreVerbose ) 
      std::cerr << " fill known classes object " << std::endl;
    
    if ( precomputedA.size() == 1)
    {
      knownClasses.insert( binaryLabelPositive );
      knownClasses.insert( binaryLabelNegative );
      if ( b_restoreVerbose ) 
        std::cerr << " binary setting - added corresp. two class numbers" << std::endl;
    }
    else
    {
      for ( std::map<int, PrecomputedType>::const_iterator itA = precomputedA.begin(); itA != precomputedA.end(); itA++)
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

void FMKGPHyperparameterOptimization::store ( std::ostream & os, int format ) const
{
  if ( os.good() )
  {
    // show starting point
    os << this->createStartTag( "FMKGPHyperparameterOptimization" ) << std::endl;
    
    os << this->createStartTag( "fmk" ) << std::endl;
    fmk->store ( os, format );
    os << this->createEndTag( "fmk" ) << std::endl;

    os.precision ( numeric_limits<double>::digits10 + 1 );



    //we only have to store the things we computed, since the remaining settings come with the config file afterwards
    
    os << this->createStartTag( "precomputedA" ) << std::endl;
    os << "size: " << precomputedA.size() << std::endl;
    std::map< int, PrecomputedType >::const_iterator preCompIt = precomputedA.begin();
    for ( uint i = 0; i < precomputedA.size(); i++ )
    {
      os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( os, format );
      preCompIt++;
    }    
    os << this->createEndTag( "precomputedA" ) << std::endl;  
    
    
    os << this->createStartTag( "precomputedB" ) << std::endl;
    os << "size: " << precomputedB.size() << std::endl;
    preCompIt = precomputedB.begin();
    for ( uint i = 0; i < precomputedB.size(); i++ )
    {
      os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( os, format );
      preCompIt++;
    }    
    os << this->createEndTag( "precomputedB" ) << std::endl; 
      
    
    
    os << this->createStartTag( "precomputedT" ) << std::endl;
    os << "size: " << precomputedT.size() << std::endl;
    if ( precomputedT.size() > 0 )
    {
      int sizeOfLUT ( 0 );
      if ( q != NULL )
        sizeOfLUT = q->size() * this->fmk->get_d();
      os << "SizeOfLUTs: " << sizeOfLUT << std::endl;      
      for ( std::map< int, double * >::const_iterator it = precomputedT.begin(); it != precomputedT.end(); it++ )
      {
        os << "index: " << it->first << std::endl;
        for ( int i = 0; i < sizeOfLUT; i++ )
        {
          os << ( it->second ) [i] << " ";
        }
        os << std::endl;
      }
    } 
    os << this->createEndTag( "precomputedT" ) << std::endl; 

    //now store the things needed for the variance estimation
    
    os << this->createStartTag( "precomputedAForVarEst" ) << std::endl;
    os << precomputedAForVarEst.size() << std::endl;
    
    if (precomputedAForVarEst.size() > 0)
    {
      precomputedAForVarEst.store ( os, format );
      os << std::endl; 
    }
    os << this->createEndTag( "precomputedAForVarEst" ) << std::endl;
    
    
    os << this->createStartTag( "precomputedTForVarEst" ) << std::endl;
    if ( precomputedTForVarEst != NULL )
    {
      os << "NOTNULL" << std::endl;
      int sizeOfLUT ( 0 );
      if ( q != NULL )
        sizeOfLUT = q->size() * this->fmk->get_d();
      
      os << sizeOfLUT << std::endl;
      for ( int i = 0; i < sizeOfLUT; i++ )
      {
        os << precomputedTForVarEst[i] << " ";
      }
      os << std::endl;
    }
    else
    {
      os << "NULL" << std::endl;
    }
    os << this->createEndTag( "precomputedTForVarEst" ) << std::endl;
    
    //store the eigenvalues and eigenvectors
    os << this->createStartTag( "eigenMax" ) << std::endl;
    os << eigenMax << std::endl;
    os << this->createEndTag( "eigenMax" ) << std::endl;   

    os << this->createStartTag( "eigenMaxVectors" ) << std::endl;
    os << eigenMaxVectors << std::endl;
    os << this->createEndTag( "eigenMaxVectors" ) << std::endl;       


    os << this->createStartTag( "ikmsum" ) << std::endl;
    for ( int j = 0; j < ikmsum->getNumberOfModels() - 1; j++ )
    {
      ( ikmsum->getModel ( j ) )->store ( os, format );
    }
    os << this->createEndTag( "ikmsum" ) << std::endl;

    
    //store the class numbers for binary settings (if mc-settings, these values will be negative by default)
    os << this->createStartTag( "binaryLabelPositive" ) << std::endl;
    os << binaryLabelPositive << std::endl;
    os << this->createEndTag( "binaryLabelPositive" ) << std::endl; 
    
    os << this->createStartTag( "binaryLabelNegative" ) << std::endl;
    os << binaryLabelNegative << std::endl;
    os << this->createEndTag( "binaryLabelNegative" ) << std::endl; 
    
    
    os << this->createStartTag( "labels" ) << std::endl;
    os << labels << std::endl;
    os << this->createEndTag( "labels" ) << std::endl;  
    
    
    os << this->createStartTag( "b_usePreviousAlphas" ) << std::endl;
    os << b_usePreviousAlphas << std::endl;
    os << this->createEndTag( "b_usePreviousAlphas" ) << std::endl;
    
    os << this->createStartTag( "previousAlphas" ) << std::endl;
    os << "size: " << previousAlphas.size() << std::endl;
    std::map< int, NICE::Vector >::const_iterator prevAlphaIt = previousAlphas.begin();
    for ( uint i = 0; i < previousAlphas.size(); i++ )
    {
      os << prevAlphaIt->first << std::endl;
      os << prevAlphaIt->second << std::endl;
      prevAlphaIt++;
    }
    os << this->createEndTag( "previousAlphas" ) << std::endl;       
    
    
    // done
    os << this->createEndTag( "FMKGPHyperparameterOptimization" ) << std::endl;    
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
  if ( this->verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addExample --- " << std::endl;  
  
  NICE::Timer t;
  t.start();  

  std::set< int > newClasses;
  
  this->labels.append ( label );
  //have we seen this class already?
  if ( !this->b_performRegression && ( this->knownClasses.find( label ) == this->knownClasses.end() ) )
  {
    this->knownClasses.insert( label );
    newClasses.insert( label );
  }    

  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  NICE::Timer tFmk;
  tFmk.start();
  this->fmk->addExample ( example, pf );
  tFmk.stop();
  if ( this->verboseTime)
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
  
  if ( this->verbose )
    std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;   
  
  if ( this->verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addExample done --- " << std::endl;   
}

void FMKGPHyperparameterOptimization::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
                                                           const NICE::Vector & newLabels,
                                                           const bool & performOptimizationAfterIncrement
                                                         )
{
  if ( this->verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addMultipleExamples --- " << std::endl;  
  
  NICE::Timer t;
  t.start();  

  std::set< int > newClasses;
  
  this->labels.append ( newLabels );
  //have we seen this class already?
  if ( !this->b_performRegression)
  {
    for ( NICE::Vector::const_iterator vecIt = newLabels.begin(); 
	vecIt != newLabels.end(); vecIt++
	)
    {  
	if ( this->knownClasses.find( *vecIt ) == this->knownClasses.end() )
      {
	this->knownClasses.insert( *vecIt );
	newClasses.insert( *vecIt );
      } 
    }
  }
  // in a regression setting, we do not have to remember any "class labels"
  else{}
  
  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  NICE::Timer tFmk;
  tFmk.start();
  this->fmk->addMultipleExamples ( newExamples, pf );
  tFmk.stop();
  if ( this->verboseTime)
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
  
  if ( this->verbose )
    std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;  
  
  if ( this->verbose )
    std::cerr << " --- FMKGPHyperparameterOptimization::addMultipleExamples done --- " << std::endl;    
}
