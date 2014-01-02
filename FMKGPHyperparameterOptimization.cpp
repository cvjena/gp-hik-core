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

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization()
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

  //TODO 
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


  this->eig = new EVArnoldi ( _conf->gB ( _confSection, "eig_verbose", false ) /* verbose flag */, 10 );
  // this->eig = new EigValuesTRLAN();
  // My time measurements show that both methods use equal time, a comparision
  // of their numerical performance has not been done yet  


  this->parameterUpperBound = _conf->gD ( _confSection, "parameter_upper_bound", 2.5 );
  this->parameterLowerBound = _conf->gD ( _confSection, "parameter_lower_bound", 1.0 );
  this->parameterStepSize = _conf->gD ( _confSection, "parameter_step_size", 0.1 );

  this->verifyApproximation = _conf->gB ( _confSection, "verify_approximation", false );
  this->nrOfEigenvaluesToConsider = _conf->gI ( _confSection, "nrOfEigenvaluesToConsider", 1 );
  this->nrOfEigenvaluesToConsiderForVarApprox = _conf->gI ( _confSection, "nrOfEigenvaluesToConsiderForVarApprox", 2 );



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
  
  if ( verbose )
  {
    std::cerr << "------------" << std::endl;
    std::cerr << "|   start   |" << std::endl;
    std::cerr << "------------" << std::endl;
  }
}

// get and set methods

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

 //high level methods

void FMKGPHyperparameterOptimization::setupGPLikelihoodApprox ( GPLikelihoodApprox * & gplike, const std::map<int, NICE::Vector> & binaryLabels, uint & parameterVectorSize )
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
  
  // EigenValue computation does not necessarily extract them in decreasing order.
  // Therefore: sort eigenvalues decreasingly!
  NICE::VectorT< int > ewPermutation;
  eigenMax.sortDescend ( ewPermutation );

  
  
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

    
    NICE::Vector tmp = gplike.getBestParameters(  );
    
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
        std::cerr << "DOWNHILLSIMPLEX WITHOUT BALANCED LEARNING!!! " << std::endl;

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
    //OPTIMIZATION::matrix_type scales ( parameterVectorSize, 1);
    //cales.Set(1.0);
    
    OPTIMIZATION::SimpleOptProblem optProblem ( &gplike, initialParams, initialParams /* scales*/ );

    //     std::cerr << "OPT: " << mypara << " " << nlikelihood << " " << logdet << " " << dataterm << std::endl;
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
  
    std::map<int, NICE::Vector> bestAlphas = gplike.getBestAlphas();
    std::cerr << "length of alpha vectors: " << bestAlphas.size() << std::endl;
    std::cerr << "alpha vector for first class: " << bestAlphas.begin()->second << std::endl;
  }
}

void FMKGPHyperparameterOptimization::transformFeaturesWithOptimalParameters ( const GPLikelihoodApprox & gplike, const uint & parameterVectorSize )
{
  // transform all features with the "optimal" parameter
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
    
    //TODO update the variance-related matrices as well here - currently it is done before in the outer method!!!
  }

}

#ifdef NICE_USELIB_MATIO
void FMKGPHyperparameterOptimization::optimizeBinary ( const sparse_t & data, const NICE::Vector & yl, const std::set<int> & positives, const std::set<int> & negatives, double noise )
{
  map<int, int> examples;
  Vector y ( yl.size() );
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
  Timer t;
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
    
    //NOTE 
    //uncomment the following, if you want to perform real binary computations with 2 classes
// 	  //we only need one vector, which already contains +1 and -1, so we need only one computation too
//     binaryLabels[ negativeClass ] = yb;
//     binaryLabels[ negativeClass ] *= -1.0;  
    
//     std::cerr << "binaryLabels.size(): " << binaryLabels.size() << std::endl;
    
//     binaryLabels[ 0 ] = yb;
//     binaryLabels[ 0 ] *= -1.0;
    
    
    //comment the following, if you want to do a real binary computation. It should be senseless, but let's see...
    
    //we do NOT do real binary computation, but an implicite one with only a single object  
    nrOfClasses--;
    std::set<int>::iterator it = myClasses.begin(); it++;
//     myClasses.erase(it);    
  }
  else //OCC setting
  {
    //we set the labels to 1, independent of the previously given class number
    //however, the original class numbers are stored and returned in classification
    Vector yNew ( y.size(), 1 );
    myClasses.clear();
    myClasses.insert ( 1 );
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
  prepareBinaryLabels ( binaryLabels, y , knownClasses );
  
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

  //NOTE The GMHIKernel is always the last model which is added (this is necessary for easy store and restore functionality)
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
  this->updateEigenDecomposition(  this->nrOfEigenvaluesToConsider );
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
    precomputedTForVarEst = T;
  }
}

void FMKGPHyperparameterOptimization::prepareVarianceApproximationFine()
{
  this->updateEigenDecomposition(  this->nrOfEigenvaluesToConsiderForVarApprox ); 
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
      map<int, double *>::const_iterator j = precomputedT.find ( classno );
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
  else
  {  // binary setting    
    scores[binaryLabelNegative] = -scores[binaryLabelPositive];     
    return scores[ binaryLabelPositive ] <= 0.0 ? binaryLabelNegative : binaryLabelPositive;
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
  else 
  { // binary setting
   
    scores[binaryLabelNegative] = -scores[binaryLabelPositive];     
    return scores[ binaryLabelPositive ] <= 0.0 ? binaryLabelNegative : binaryLabelPositive;
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
//     t.stop();  
//   std::cerr << "ApproxExact -- time for first term: "  << t.getLast()  << std::endl;

//   t.start();  
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );
//  t.stop();
//   std::cerr << "ApproxExact -- time for kernel vector: "  << t.getLast()  << std::endl;
//   

  
  //now run the ILS method
  NICE::Vector diagonalElements;
  ikmsum->getDiagonalElements ( diagonalElements );

//     t.start();
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
/*    t.stop();
std::cerr << "ApproxExact -- time for preconditioning etc: "  << t.getLast()  << std::endl;    
  
t.start();*/
  //   t.start();
  linsolver->solveLin ( *ikmsum, kStar, beta );
  //   t.stop();
//     t.stop();
//         t.stop();
//   std::cerr << "ApproxExact -- time for lin solve: "  << t.getLast()  << std::endl;

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
//   Timer t;
//   t.start();

  double kSelf ( 0.0 );
  int dim ( 0 );
  for ( NICE::Vector::const_iterator it = x.begin(); it != x.end(); it++, dim++ )
  {
    kSelf += pf->f ( 0, *it );
    // if weighted dimensions:
    //kSelf += pf->f(dim,*it);
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

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::Vector & x, double & predVariance ) const
{
  if ( ikmsum->getNumberOfModels() == 0 )
  {
    fthrow ( Exception, "ikmsum is empty... have you trained this classifer? Aborting..." );
  }  
  
    Timer t;
//   t.start();
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
//     t.stop();  
//   std::cerr << "ApproxExact -- time for first term: "  << t.getLast()  << std::endl;

//   t.start();  
  NICE::Vector kStar;
  fmk->hikComputeKernelVector ( x, kStar );
//  t.stop();
//   std::cerr << "ApproxExact -- time for kernel vector: "  << t.getLast()  << std::endl;
//   

  //now run the ILS method
  NICE::Vector diagonalElements;
  ikmsum->getDiagonalElements ( diagonalElements );

//     t.start();
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
/*    t.stop();
std::cerr << "ApproxExact -- time for preconditioning etc: "  << t.getLast()  << std::endl;    
  
t.start();*/
  //   t.start();
  linsolver->solveLin ( *ikmsum, kStar, beta );
  //   t.stop();
//     t.stop();
//         t.stop();
//   std::cerr << "ApproxExact -- time for lin solve: "  << t.getLast()  << std::endl;

  beta *= kStar;
  
  double currentSecondTerm( beta.Sum() );
  predVariance = kSelf - currentSecondTerm;
}    

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

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

    
    // done
    os << this->createEndTag( "FMKGPHyperparameterOptimization" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void FMKGPHyperparameterOptimization::clear ( ) {};
