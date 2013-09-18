/**
* @file FMKGPHyperparameterOptimization.cpp
* @brief Heart of the framework to set up everything, perform optimization, incremental updates, classification, variance prediction (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/02/2012

*/
#include <iostream>
#include <map>

#include <core/algebra/ILSConjugateGradients.h>
#include <core/algebra/ILSConjugateGradientsLanczos.h>
#include <core/algebra/ILSSymmLqLanczos.h>
#include <core/algebra/ILSMinResLanczos.h>
#include <core/algebra/ILSPlainGradient.h>
#include <core/algebra/EigValuesTRLAN.h>
#include <core/algebra/CholeskyRobust.h>
#include <core/vector/Algorithms.h>
#include <core/vector/Eigen.h>
#include <core/basics/Timer.h>
#include <core/basics/ResourceStatistics.h>

#include "core/optimization/blackbox/DownhillSimplexOptimizer.h"

#include "FMKGPHyperparameterOptimization.h"
#include "FastMinKernel.h"
#include "GMHIKernel.h"
#include "IKMNoise.h"
#include "../core/basics/Exception.h"


using namespace NICE;
using namespace std;

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization()
{
  pf = NULL;
  eig = NULL;
  linsolver = NULL;
  fmk = NULL;
  q = NULL;
  precomputedTForVarEst = NULL;
  verbose = false;
  verboseTime = false;
  debug = false;
  
  //stupid unneeded default values
  binaryLabelPositive = -1;
  binaryLabelNegative = -2;
}

FMKGPHyperparameterOptimization::FMKGPHyperparameterOptimization ( const Config *_conf, ParameterizedFunction *_pf, FastMinKernel *_fmk, const string & _confSection )
{
  //default settings, may become overwritten lateron
  pf = NULL;
  eig = NULL;
  linsolver = NULL;
  fmk = NULL;
  q = NULL;
  precomputedTForVarEst = NULL;
  
  //stupid unneeded default values
  binaryLabelPositive = -1;
  binaryLabelNegative = -2;  
  knownClasses.clear();

  if ( _fmk == NULL )
    this->initialize ( _conf, _pf ); //then the confSection is also the default value
  //TODO not needed anymore, only for backword compatibility
//   else if ( _confSection.compare ( "HIKGP" ) == 0 )
//     this->initialize ( _conf, _pf, _fmk );
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

  for ( std::map<int, IKMLinearCombination * >::iterator it =  ikmsums.begin(); it != ikmsums.end(); it++ )
    delete it->second;
}

void FMKGPHyperparameterOptimization::initialize ( const Config *_conf, ParameterizedFunction *_pf, FastMinKernel *_fmk, const std::string & _confSection )
{
  if ( this->fmk != NULL )
    delete this->fmk;
  if ( _fmk != NULL )
    this->fmk = _fmk;
  this->pf = _pf;
  
  
  std::cerr << "------------" << std::endl;
  std::cerr << "|  set-up  |" << std::endl;
  std::cerr << "------------" << std::endl;


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

  this->verbose = _conf->gB ( _confSection, "verbose", false );
  this->verboseTime = _conf->gB ( _confSection, "verboseTime", false );
  this->debug = _conf->gB ( _confSection, "debug", false );

  bool useQuantization = _conf->gB ( _confSection, "use_quantization", false );
  std::cerr << "_confSection: " << _confSection << std::endl;
  std::cerr << "use_quantization: " << useQuantization << std::endl;
  if ( _conf->gB ( _confSection, "use_quantization", false ) ) {
    int numBins = _conf->gI ( _confSection, "num_bins", 100 );
    if ( verbose )
      cerr << "FMKGPHyperparameterOptimization: quantization initialized with " << numBins << " bins." << endl;
    this->q = new Quantization ( numBins );
  } else {
    this->q = NULL;
  }

  bool ils_verbose = _conf->gB ( _confSection, "ils_verbose", false );
  ils_max_iterations = _conf->gI ( _confSection, "ils_max_iterations", 1000 );
  if ( verbose )
    cerr << "FMKGPHyperparameterOptimization: maximum number of iterations is " << ils_max_iterations << endl;

  double ils_min_delta = _conf->gD ( _confSection, "ils_min_delta", 1e-7 );
  double ils_min_residual = _conf->gD ( _confSection, "ils_min_residual", 1e-7/*1e-2 */ );

  string ils_method = _conf->gS ( _confSection, "ils_method", "CG" );
  if ( ils_method.compare ( "CG" ) == 0 )
  {
    if ( verbose )
      std::cerr << "We use CG with " << ils_max_iterations << " iterations, " << ils_min_delta << " as min delta, and " << ils_min_residual << " as min res " << std::endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
    if ( verbose )
      cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients" << endl;
  }
  else if ( ils_method.compare ( "CGL" ) == 0 )
  {
    this->linsolver = new ILSConjugateGradientsLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      cerr << "FMKGPHyperparameterOptimization: using ILS ConjugateGradients (Lanczos)" << endl;
  }
  else if ( ils_method.compare ( "SYMMLQ" ) == 0 )
  {
    this->linsolver = new ILSSymmLqLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      cerr << "FMKGPHyperparameterOptimization: using ILS SYMMLQ" << endl;
  }
  else if ( ils_method.compare ( "MINRES" ) == 0 )
  {
    this->linsolver = new ILSMinResLanczos ( ils_verbose , ils_max_iterations );
    if ( verbose )
      cerr << "FMKGPHyperparameterOptimization: using ILS MINRES" << endl;
  }
  else
  {
    cerr << "FMKGPHyperparameterOptimization: " << _confSection << ":ils_method (" << ils_method << ") does not match any type (CG,CGL,SYMMLQ,MINRES), I will use CG" << endl;
    this->linsolver = new ILSConjugateGradients ( ils_verbose , ils_max_iterations, ils_min_delta, ils_min_residual );
  }
  
  this->usePreviousAlphas = _conf->gB (_confSection, "usePreviousAlphas", true );

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
    cerr << "Using optimization method: " << optimizationMethod_s << endl;

  downhillSimplexMaxIterations = _conf->gI ( _confSection, "downhillsimplex_max_iterations", 20 );
  // do not run longer than a day :)
  downhillSimplexTimeLimit = _conf->gD ( _confSection, "downhillsimplex_time_limit", 24 * 60 * 60 );
  downhillSimplexParamTol = _conf->gD ( _confSection, "downhillsimplex_delta", 0.01 );

  learnBalanced = _conf->gB ( _confSection, "learn_balanced", false );
  std::cerr << "balanced learning: " << learnBalanced << std::endl;

  optimizeNoise = _conf->gB ( _confSection, "optimize_noise", false );
  if ( verbose )
    cerr << "Optimize noise: " << ( optimizeNoise ? "on" : "off" ) << endl;
  
  std::cerr << "------------" << std::endl;
  std::cerr << "|   start   |" << std::endl;
  std::cerr << "------------" << std::endl;  
}

void FMKGPHyperparameterOptimization::setParameterUpperBound ( const double & _parameterUpperBound )
{
  parameterUpperBound = _parameterUpperBound;
}
void FMKGPHyperparameterOptimization::setParameterLowerBound ( const double & _parameterLowerBound )
{
  parameterLowerBound = _parameterLowerBound;
}

void FMKGPHyperparameterOptimization::setupGPLikelihoodApprox ( std::map<int, GPLikelihoodApprox * > & gplikes, const std::map<int, NICE::Vector> & binaryLabels, std::map<int, uint> & parameterVectorSizes )
{
  if ( learnBalanced )
  {
    if ( verbose )
    {
      std::cerr << "FMKGPHyperparameterOptimization::setupGPLikelihoodApprox -- balanced setting" << std::endl;
      std::cerr << "number of ikmsum-objects: " << ikmsums.size() << std::endl;
    }
    
    for ( std::map<int, IKMLinearCombination*>::const_iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
    {
      map<int, NICE::Vector> binaryLabelsSingle;
      binaryLabelsSingle.insert ( *binaryLabels.find ( it->first ) );
      GPLikelihoodApprox *gplike = new GPLikelihoodApprox ( binaryLabelsSingle, it->second, linsolver, eig, verifyApproximation, nrOfEigenvaluesToConsider );
      gplike->setUsePreviousAlphas( usePreviousAlphas );
      gplike->setDebug( debug );
      gplike->setVerbose( verbose );
      gplikes.insert ( std::pair<int, GPLikelihoodApprox * > ( it->first, gplike ) );
      parameterVectorSizes.insert ( std::pair<int, uint> ( it->first, it->second->getNumParameters() ) );
    }
    if ( verbose )
      std::cerr << "resulting number of gplike-objects: " << gplikes.size() << std::endl;
  }
  else
  {
    GPLikelihoodApprox *gplike = new GPLikelihoodApprox ( binaryLabels, ikmsums.begin()->second, linsolver, eig, verifyApproximation, nrOfEigenvaluesToConsider );
    gplike->setUsePreviousAlphas( usePreviousAlphas );
    gplike->setDebug( debug );
    gplike->setVerbose( verbose );
    gplikes.insert ( std::pair<int, GPLikelihoodApprox * > ( 0, gplike ) );
    parameterVectorSizes.insert ( std::pair<int, uint> ( 0, ikmsums.begin()->second->getNumParameters() ) );
  }
}

void FMKGPHyperparameterOptimization::updateEigenVectors()
{
  if ( verbose )
  {
    std::cerr << "FMKGPHyperparameterOptimization::updateEigenVectors -- size of ikmsums: " << ikmsums.size() << std::endl;
  }
  
  if ( learnBalanced )
  {
    //simply use the first kernel matrix to compute the eigenvalues and eigenvectors for the fine approximation of predictive uncertainties
    std::map<int, IKMLinearCombination * >::iterator ikmsumsIt;
    eigenMax.resize(ikmsums.size());
    eigenMaxVectors.resize(ikmsums.size());
    
    int classCnt(0);
    for ( ikmsumsIt = ikmsums.begin(); ikmsumsIt != ikmsums.end(); ikmsumsIt++, classCnt++ )
    {
      
      eig->getEigenvalues ( * ikmsumsIt->second, eigenMax[classCnt], eigenMaxVectors[classCnt], nrOfEigenvaluesToConsiderForVarApprox );
    }
  }
  else
  {
    //compute the largest eigenvalue of K + noise
    eigenMax.resize(1);
    eigenMaxVectors.resize(1);    
    
    //TODO check why we are only interested in the largest EW!
    eig->getEigenvalues ( * ( ikmsums.begin()->second ),  eigenMax[0], eigenMaxVectors[0], 1 /* we are only interested in the largest eigenvalue here*/ );
  }
}

void FMKGPHyperparameterOptimization::performOptimization ( std::map<int, GPLikelihoodApprox * > & gplikes, const std::map<int, uint> & parameterVectorSizes, const bool & roughOptimization )
{
  if (verbose)
    std::cerr << "perform optimization" << std::endl;
  
  if ( optimizationMethod == OPT_GREEDY )
  {
    if ( verbose )    
      std::cerr << "OPT_GREEDY!!! " << std::endl;
    
    // simple greedy strategy
    if ( ikmsums.begin()->second->getNumParameters() != 1 )
      fthrow ( Exception, "Reduce size of the parameter vector or use downhill simplex!" );

    Vector lB = ikmsums.begin()->second->getParameterLowerBounds();
    Vector uB = ikmsums.begin()->second->getParameterUpperBounds();
    
    if ( verbose )
      cerr << "lower bound " << lB << " upper bound " << uB << endl;

    if ( learnBalanced )
    {
      if ( lB[0] == uB[0] ) //do we already know a specific parameter?
      {
        for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
        {
          if ( verbose )
            std::cerr << "Optimizing class " << gpLikeIt->first << std::endl;

          OPTIMIZATION::matrix_type hyperp ( 1, 1, lB[0] );
          gpLikeIt->second->evaluate ( hyperp );
        }
      }
      else
      {
        fthrow ( Exception, "HYPERPARAMETER OPTIMZIATION SHOULD NOT BE USED TOGETHER WITH BALANCED LEARNING IN THIS FRAMEWORK!!!" );
      }
    }
    else
    {
      for ( double mypara = lB[0]; mypara <= uB[0]; mypara += this->parameterStepSize )
      {
        OPTIMIZATION::matrix_type hyperp ( 1, 1, mypara );
        gplikes.begin()->second->evaluate ( hyperp );
      }
    }
  }
  else if ( optimizationMethod == OPT_DOWNHILLSIMPLEX )
  {

    if ( learnBalanced )
    {
      if ( verbose )
        std::cerr << "DOWNHILLSIMPLEX WITH BALANCED LEARNING!!! " << std::endl;
      fthrow ( Exception, "HYPERPARAMETER OPTIMZIATION SHOULD NOT BE USED TOGETHER WITH BALANCED LEARNING IN THIS FRAMEWORK!!!" );

      //unfortunately, we suffer from the fact that we do only have a single fmk-object
      //therefore, we should either copy the fmk-object as often as we have classes or do some averaging or whatsoever
    }
    else
    { //standard as before, normal optimization
      if ( verbose )    
        std::cerr << "DOWNHILLSIMPLEX WITHOUT BALANCED LEARNING!!! " << std::endl;

      // downhill simplex strategy
      OPTIMIZATION::DownhillSimplexOptimizer optimizer;

      OPTIMIZATION::matrix_type initialParams ( parameterVectorSizes.begin()->second, 1 );

      Vector currentParameters;
      ikmsums.begin()->second->getParameters ( currentParameters );

      for ( uint i = 0 ; i < parameterVectorSizes.begin()->second; i++ )
        initialParams(i,0) = currentParameters[ i ];

      if ( verbose )      
        std::cerr << "Initial parameters: " << initialParams << std::endl;

//       OPTIMIZATION::matrix_type scales ( parameterVectorSizes.begin()->second, 1);

      if ( roughOptimization ) //should be used when we perform the optimziation for the first time
      {
//         scales.Set(1.0);
      }
      else  //should be used, when we perform the optimization in an incremental learning scenario, so that we already have a good guess
      {
//         scales.Set(1.0);
//         for ( uint i = 0 ; i < parameterVectorSizes.begin()->second; i++ )
//           scales[i][0] = currentParameters[ i ];
        optimizer.setDownhillParams ( 0.2 /* default: 1.0 */, 0.1 /* default: 0.5 */, 0.2 /* default: 1.0 */ );
      }

      //the scales object does not really matter in the actual implementation of Downhill Simplex
      OPTIMIZATION::SimpleOptProblem optProblem ( gplikes.begin()->second, initialParams, initialParams /* scales*/ );

      //     cerr << "OPT: " << mypara << " " << nlikelihood << " " << logdet << " " << dataterm << endl;
      optimizer.setMaxNumIter ( true, downhillSimplexMaxIterations );
      optimizer.setTimeLimit ( true, downhillSimplexTimeLimit );
      optimizer.setParamTol ( true, downhillSimplexParamTol );
      optimizer.optimizeProb ( optProblem );

    }
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
    if ( learnBalanced )
    {
      for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
      {
        OPTIMIZATION::matrix_type hyperp ( 1, 1, value);    
        gpLikeIt->second->setParameterLowerBound ( value );
        gpLikeIt->second->setParameterUpperBound ( value );   
        gpLikeIt->second->evaluate ( hyperp );
      }
    }
    else
    {
      OPTIMIZATION::matrix_type hyperp ( 1, 1, value );
      gplikes.begin()->second->setParameterLowerBound ( value );
      gplikes.begin()->second->setParameterUpperBound ( value );
      //we do not need to compute the likelihood here - we are only interested in directly obtaining alpha vectors
      gplikes.begin()->second->computeAlphaDirect( hyperp );
    }
  }

  if ( learnBalanced )
  {
    lastAlphas.clear();
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      if (verbose)
        std::cerr << "Optimal hyperparameter for class " << gpLikeIt->first << " was: " << gpLikeIt->second->getBestParameters() << std::endl;
      
      lastAlphas = gplikes.begin()->second->getBestAlphas();
    }
  }
  else
  {
    if ( verbose )
      std::cerr << "Optimal hyperparameter was: " << gplikes.begin()->second->getBestParameters() << std::endl;
    lastAlphas.clear();
    lastAlphas = gplikes.begin()->second->getBestAlphas();
  }
}

void FMKGPHyperparameterOptimization::transformFeaturesWithOptimalParameters ( const std::map<int, GPLikelihoodApprox * > & gplikes, const std::map<int, uint> & parameterVectorSizes )
{
  if ( verbose )
    std::cerr << "FMKGPHyperparameterOptimization::transformFeaturesWithOptimalParameters" << std::endl;
  
  // transform all features with the "optimal" parameter
  if ( learnBalanced )
  {
    if ( verbose )
      std::cerr << "learn Balanced" << std::endl;
    
    double meanValue ( 0.0 );
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      meanValue += gpLikeIt->second->getBestParameters() [0];
    }
    meanValue /= gplikes.size();
    NICE::Vector averagedParams ( parameterVectorSizes.begin()->second, meanValue );
    
    if ( verbose)
      std::cerr << "averaged Params: " << averagedParams << std::endl;

    //since we only have a single fmk-object, we only have to modify our data for a single time
    ikmsums.begin()->second->setParameters ( averagedParams );
  }
  else
  {
    if ( verbose )
    {
      std::cerr << "learn not Balanced" << std::endl;
      std::cerr << "previous best parameters. " << gplikes.begin()->second->getBestParameters() << std::endl;
//     std::cerr << "previous best alphas: " << gplikes.begin()->second->getBestAlphas() << std::endl;
    }
    
    ikmsums.begin()->second->setParameters ( gplikes.begin()->second->getBestParameters() );
  }
}

void FMKGPHyperparameterOptimization::computeMatricesAndLUTs ( const std::map<int, GPLikelihoodApprox * > & gplikes )
{
  precomputedA.clear();
  precomputedB.clear();

  if ( learnBalanced )
  {
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      std::map<int, Vector>::const_iterator i = gpLikeIt->second->getBestAlphas().begin();

      PrecomputedType A;
      PrecomputedType B;

//       std::cerr << "computeMatricesAndLUTs -- alpha: " << i->second << std::endl;

      fmk->hik_prepare_alpha_multiplications ( i->second, A, B );
      A.setIoUntilEndOfFile ( false );
      B.setIoUntilEndOfFile ( false );
      precomputedA[ gpLikeIt->first ] = A;
      precomputedB[ gpLikeIt->first ] = B;

      if ( q != NULL )
      {
        double *T = fmk->hik_prepare_alpha_multiplications_fast ( A, B, *q, pf );
        //just to be sure that we do not waste space here
        if ( precomputedT[ gpLikeIt->first ] != NULL )
          delete precomputedT[ gpLikeIt->first ];
        
        precomputedT[ gpLikeIt->first ] = T;
      }
    }
  }
  else
  { //no GP rebalancing

    for ( std::map<int, Vector>::const_iterator i = gplikes.begin()->second->getBestAlphas().begin(); i != gplikes.begin()->second->getBestAlphas().end(); i++ )
    {
      PrecomputedType A;
      PrecomputedType B;

//       std::cerr << "computeMatricesAndLUTs -- alpha: " << i->second << std::endl;

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
  cerr << "Examples: " << examples.size() << endl;

  optimize ( data, y, examples, noise );
}


void FMKGPHyperparameterOptimization::optimize ( const sparse_t & data, const NICE::Vector & y, const std::map<int, int> & examples, double noise )
{
  Timer t;
  t.start();
  cerr << "Initializing data structure ..." << std::endl;
  if ( fmk != NULL ) delete fmk;
  fmk = new FastMinKernel ( data, noise, examples );
  t.stop();
  if (verboseTime)
    std::cerr << "Time used for initializing the FastMinKernel structure: " << t.getLast() << std::endl;
  
  optimize ( y );
}
#endif

int FMKGPHyperparameterOptimization::prepareBinaryLabels ( map<int, NICE::Vector> & binaryLabels, const NICE::Vector & y , std::set<int> & myClasses )
{
  myClasses.clear();
  for ( NICE::Vector::const_iterator it = y.begin(); it != y.end(); it++ )
    if ( myClasses.find ( *it ) == myClasses.end() )
    {
      myClasses.insert ( *it );
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
//     std::cerr << "binary setting -- prepare two binary label vectors with opposite signs" << std::endl;
    Vector yb ( y );

    binaryLabelNegative = *(myClasses.begin());
    std::set<int>::const_iterator classIt = myClasses.begin(); classIt++;
    binaryLabelPositive = *classIt;
    
//     std::cerr << "positiveClass : " << binaryLabelPositive << " negativeClass: " << binaryLabelNegative << std::endl;

    for ( uint i = 0 ; i < yb.size() ; i++ )
      yb[i] = ( y[i] == binaryLabelNegative ) ? -1.0 : 1.0;
    
    binaryLabels[ binaryLabelPositive ] = yb;
	  //binaryLabels[ 1 ] = yb;
    
    //uncomment the following, if you want to perform real binary computations with 2 classes
// 	  //we only need one vector, which already contains +1 and -1, so we need only one computation too
//     binaryLabels[ negativeClass ] = yb;
//     binaryLabels[ negativeClass ] *= -1.0;  
    
//     std::cerr << "binaryLabels.size(): " << binaryLabels.size() << std::endl;
    
//     binaryLabels[ 0 ] = yb;
//     binaryLabels[ 0 ] *= -1.0;
    
    
    //comment the following, if you want to do a real binary computation. It should be senseless, but let's see...
    
    //we do no real binary computation, but an implicite one with only a single object   
    nrOfClasses--;
    std::set<int>::iterator it = myClasses.begin(); it++;
    myClasses.erase(it);    
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
//   std::set<int> classesToUse;
//   classesToUse.clear();
//   
//   for (std::map<int, NICE::Vector>::const_iterator clIt = binaryLabels.begin(); clIt != binaryLabels.end(); clIt++)
//   {
//     classesToUse.insert(clIt->first);
//   }
  
  if (verbose)
  {
    std::cerr << "Initial noise level: " << fmk->getNoise() << endl;

    std::cerr << "Number of classes (=1 means we have a binary setting):" << nrOfClasses << std::endl;
    std::cerr << "Effective number of classes (neglecting classes without positive examples): " << knownClasses.size() << std::endl;
  }

  // combine standard model and noise model
  ikmsums.clear();

  Timer t1;

  t1.start();
  //setup the kernel combination
  if ( learnBalanced )
  {
    for ( std::set<int>::const_iterator clIt = knownClasses.begin(); clIt != knownClasses.end(); clIt++ )
    {
      IKMLinearCombination *ikmsum = new IKMLinearCombination ();
      ikmsums.insert ( std::pair<int, IKMLinearCombination*> ( *clIt, ikmsum ) );
    }
  }
  else
  {
    IKMLinearCombination *ikmsum = new IKMLinearCombination ();
    ikmsums.insert ( std::pair<int, IKMLinearCombination*> ( 0, ikmsum ) );
  }

  if ( verbose )
  {
    std::cerr << "ikmsums.size(): " << ikmsums.size() << std::endl;
    std::cerr << "binaryLabels.size(): " << binaryLabels.size() << std::endl;
  }

//   First model: noise
  if ( learnBalanced )
  {
    int cnt ( 0 );
    for ( std::set<int>::const_iterator clIt = knownClasses.begin(); clIt != knownClasses.end(); clIt++, cnt++ )
    {
      ikmsums.find ( *clIt )->second->addModel ( new IKMNoise ( binaryLabels[*clIt], fmk->getNoise(), optimizeNoise ) );
    }
  }
  else
  {
    ikmsums.find ( 0 )->second->addModel ( new IKMNoise ( fmk->get_n(), fmk->getNoise(), optimizeNoise ) );
  }
  
  // set pretty low built-in noise, because we explicitely add the noise with the IKMNoise
  fmk->setNoise ( 0.0 );

  //NOTE The GMHIKernel is always the last model which is added (this is necessary for easy store and restore functionality)
  for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
  {
    it->second->addModel ( new GMHIKernel ( fmk, pf, NULL /* no quantization */ ) );
  }
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the ikm-objects: " << t1.getLast() << std::endl;

  std::map<int, GPLikelihoodApprox * > gplikes;
  std::map<int, uint> parameterVectorSizes;

  t1.start();
  this->setupGPLikelihoodApprox ( gplikes, binaryLabels, parameterVectorSizes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  if (verbose)
  {
    std::cerr << "parameterVectorSizes: " << std::endl;
    for ( std::map<int, uint>::const_iterator pvsIt = parameterVectorSizes.begin(); pvsIt != parameterVectorSizes.end(); pvsIt++ )
    {
      std::cerr << pvsIt->first << " " << pvsIt->second << " ";
    }
    std::cerr << std::endl;
  }

  t1.start();
  this->updateEigenVectors();
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  if ( verbose )
    std::cerr << "resulting eigenvalues for first class: " << eigenMax[0] << std::endl;

  t1.start();
  this->performOptimization ( gplikes, parameterVectorSizes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

  if ( verbose )
    cerr << "Preparing classification ..." << endl;

  t1.start();
  this->transformFeaturesWithOptimalParameters ( gplikes, parameterVectorSizes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;

  t1.start();
  this->computeMatricesAndLUTs ( gplikes );
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
  if ( learnBalanced )
  {
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      delete gpLikeIt->second;
    }
  }
  else
  {
    delete gplikes.begin()->second;
  }
}

void FMKGPHyperparameterOptimization::updateAfterSingleIncrement ( const NICE::SparseVector & x, const bool & performOptimizationAfterIncrement )
{
  Timer t;
  t.start();
  if ( fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  std::map<int, NICE::Vector> binaryLabels;
  std::set<int> classesToUse;
  prepareBinaryLabels ( binaryLabels, labels , classesToUse );
  if ( verbose )
    std::cerr << "labels.size() after increment: " << labels.size() << std::endl;

  Timer t1;
  t1.start();
  //update the kernel combinations
  std::map<int, NICE::Vector>::const_iterator labelIt = binaryLabels.begin();
  // note, that if we only have a single ikmsum-object, than the labelvector will not be used at all in the internal objects (only relevant in ikmnoise)

  for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
  {
    it->second->addExample ( x, labelIt->second );
    labelIt++;
  }

  //we have to reset the fmk explicitely
  for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
  {
    ( ( GMHIKernel* ) it->second->getModel ( it->second->getNumberOfModels() - 1 ) )->setFastMinKernel ( this->fmk );
  }

  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the ikm-objects: " << t1.getLast() << std::endl;

  std::map<int, GPLikelihoodApprox * > gplikes;
  std::map<int, uint> parameterVectorSizes;

  t1.start();
  this->setupGPLikelihoodApprox ( gplikes, binaryLabels, parameterVectorSizes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  if ( verbose )
  {
    std::cerr << "parameterVectorSizes: " << std::endl;
    for ( std::map<int, uint>::const_iterator pvsIt = parameterVectorSizes.begin(); pvsIt != parameterVectorSizes.end(); pvsIt++ )
    {
      std::cerr << pvsIt->first << " " << pvsIt->second << " ";
    }
    std::cerr << std::endl;
  }

  t1.start();
  if ( usePreviousAlphas )
  {
    std::map<int, NICE::Vector>::const_iterator binaryLabelsIt = binaryLabels.begin();
    std::vector<NICE::Vector>::const_iterator eigenMaxIt = eigenMax.begin(); 
    for ( std::map<int, NICE::Vector>::iterator lastAlphaIt = lastAlphas.begin() ;lastAlphaIt != lastAlphas.end(); lastAlphaIt++ )
    {
      int oldSize ( lastAlphaIt->second.size() );
      lastAlphaIt->second.resize ( oldSize + 1 );

      //We initialize it with the same values as we use in GPLikelihoodApprox in batch training
      //default in GPLikelihoodApprox for the first time:
      // alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );

      double maxEigenValue ( 1.0 );
      if ( (*eigenMaxIt).size() > 0 )
        maxEigenValue = (*eigenMaxIt)[0];
      double factor ( 1.0 / maxEigenValue );    

      if ( binaryLabelsIt->second[oldSize] > 0 ) //we only have +1 and -1, so this might be benefitial in terms of speed
        lastAlphaIt->second[oldSize] = factor;
      else
        lastAlphaIt->second[oldSize] = -factor; //we follow the initialization as done in previous steps
        //lastAlphaIt->second[oldSize] = 0.0; // following the suggestion of Yeh and Darrell

      binaryLabelsIt++;
      
      if (learnBalanced)
      {
        eigenMaxIt++;
      }
    }

    for ( std::map<int, GPLikelihoodApprox * >::iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      gpLikeIt->second->setLastAlphas ( &lastAlphas );
    }
  }
  //if we do not use previous alphas, we do not have to set up anything here  
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the alpha-objects: " << t1.getLast() << std::endl;

  t1.start();
  this->updateEigenVectors();
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  if ( verbose )
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
  //NOTE we could skip this, if we do not want to change our parameters given new examples
  if ( performOptimizationAfterIncrement )
  {
    t1.start();
    this->performOptimization ( gplikes, parameterVectorSizes, false /* initialize not with default values but using the last solution */ );
    t1.stop();
    if (verboseTime)
      std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

    if ( verbose )
      cerr << "Preparing after retraining for classification ..." << endl;

    t1.start();
    this->transformFeaturesWithOptimalParameters ( gplikes, parameterVectorSizes );
    t1.stop();
    if (verboseTime)
      std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;
  }
  else
  {
    t1.start();
    t1.stop();
    std::cerr << "skip optimization" << std::endl;
    if (verboseTime)
      std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

    std::cerr << "skip feature transformation" << std::endl;
    if (verboseTime)
      std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;
  }
  
  //NOTE unfortunately, the whole vector alpha differs, and not only its last entry.
  // If we knew any method, which could update this efficiently, we could also compute A and B more efficiently by updating them.
  // Since we are not aware of any such method, we have to compute them completely new
  // :/
  t1.start();
  this->computeMatricesAndLUTs ( gplikes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the A'nB -objects: " << t1.getLast() << std::endl;

  t.stop();  

  ResourceStatistics rs;
  std::cerr << "Time used for re-learning: " << t.getLast() << std::endl;
  long maxMemory;
  rs.getMaximumMemory ( maxMemory );
  std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;

  //don't waste memory
  if ( learnBalanced )
  {
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      delete gpLikeIt->second;
    }
  }
  else
  {
    delete gplikes.begin()->second;
  }
}

void FMKGPHyperparameterOptimization::updateAfterMultipleIncrements ( const std::vector<const NICE::SparseVector*> & x, const bool & performOptimizationAfterIncrement )
{
  Timer t;
  t.start();
  if ( fmk == NULL )
    fthrow ( Exception, "FastMinKernel object was not initialized!" );

  std::map<int, NICE::Vector> binaryLabels;
  std::set<int> classesToUse;
  this->prepareBinaryLabels ( binaryLabels, labels , classesToUse );
  //actually, this is not needed, since we have the global set knownClasses
  classesToUse.clear();
  
  std::map<int, NICE::Vector> newBinaryLabels;
  if ( newClasses.size() > 0)
  {
    for (std::set<int>::const_iterator newClIt = newClasses.begin(); newClIt != newClasses.end(); newClIt++)
    {
      std::map<int, NICE::Vector>::iterator binLabelIt = binaryLabels.find(*newClIt);
      newBinaryLabels.insert(*binLabelIt);
    }
  }
  
  if ( verbose )
    std::cerr << "labels.size() after increment: " << labels.size() << std::endl;
 
  
  // ************************************************************
  //   include the information for classes we know so far    
  // ************************************************************
  if (verbose)
    std::cerr <<  "include the information for classes we know so far " << std::endl;
  
  Timer t1;
  t1.start();
  //update the kernel combinations
  std::map<int, NICE::Vector>::const_iterator labelIt = binaryLabels.begin();
  // note, that if we only have a single ikmsum-object, than the labelvector will not be used at all in the internal objects (only relevant in ikmnoise)

  if ( verbose )
  {
    if ( newClasses.size() > 0)
    {
      std::cerr << "new classes: ";
      for (std::set<int>::const_iterator newClIt = newClasses.begin(); newClIt != newClasses.end(); newClIt++)
      {
        std::cerr << *newClIt << " ";
      }
      std::cerr << std::endl;
    }
    else
      std::cerr << "no new classes" << std::endl;
  }
    
  for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
  {
    //make sure that we only work on currently known classes in this loop
    while ( ( newClasses.size() > 0) && (newClasses.find( labelIt->first ) != newClasses.end()) )
    {
      labelIt++;
    }
    for ( std::vector<const NICE::SparseVector*>::const_iterator exampleIt = x.begin(); exampleIt != x.end(); exampleIt++ )
    {
      it->second->addExample ( **exampleIt, labelIt->second );
    }
    labelIt++;
  }
  
  //we have to reset the fmk explicitely
  for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
  {
    if ( newClasses.find( it->first ) != newClasses.end() )
      continue;
    else
      ( ( GMHIKernel* ) it->second->getModel ( it->second->getNumberOfModels() - 1 ) )->setFastMinKernel ( this->fmk );
  }

  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the ikm-objects for known classes: " << t1.getLast() << std::endl;
  
  // *********************************************
  //          work on the new classes
  // *********************************************
    
  if (verbose)    
    std::cerr << "work on the new classes " << std::endl;
  
  double tmpNoise;  
  (ikmsums.begin()->second->getModel( 0 ))->getFirstDiagonalElement(tmpNoise);
    
  if ( newClasses.size() > 0)
  {
    //setup the new kernel combinations
    if ( learnBalanced )
    {
      for ( std::set<int>::const_iterator clIt = newClasses.begin(); clIt != newClasses.end(); clIt++ )
      {
        IKMLinearCombination *ikmsum = new IKMLinearCombination ();
        ikmsums.insert ( std::pair<int, IKMLinearCombination*> ( *clIt, ikmsum ) );
      }
    }
    else
    {
      //nothing to do, we already have the single ikmsum-object
    } 
    
  //   First model: noise
    if ( learnBalanced )
    {
      for ( std::set<int>::const_iterator clIt = newClasses.begin(); clIt != newClasses.end(); clIt++ )
      {
        ikmsums.find ( *clIt )->second->addModel ( new IKMNoise ( newBinaryLabels[*clIt], tmpNoise, optimizeNoise ) );
      }
    }
    else
    {
      //nothing to do, we already have the single ikmsum-object
    }
    
    if ( learnBalanced )
    {    
      //NOTE The GMHIKernel is always the last model which is added (this is necessary for easy store and restore functionality)
      std::map<int, IKMLinearCombination * >::iterator ikmSumIt = ikmsums.begin();
      for ( std::set<int>::const_iterator clIt = newClasses.begin(); clIt != newClasses.end(); clIt++ )
      {
        while ( ikmSumIt->first != *clIt)
        {
          ikmSumIt++;
        }
        ikmSumIt->second->addModel ( new GMHIKernel ( this->fmk, pf, NULL /* no quantization */ ) );
      }  
    }
    else{
      //nothing to do, we already have the single ikmsum-object
    }
  } // if ( newClasses.size() > 0)  
  
  // ******************************************************************************************
  //       now do everything which is independent of the number of new classes
  // ******************************************************************************************  

  if (verbose)
    std::cerr << "now do everything which is independent of the number of new classes" << std::endl;

  std::map<int, GPLikelihoodApprox * > gplikes;
  std::map<int, uint> parameterVectorSizes;

  t1.start();
  this->setupGPLikelihoodApprox ( gplikes, binaryLabels, parameterVectorSizes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the gplike-objects: " << t1.getLast() << std::endl;

  if ( verbose )
  {
    std::cerr << "parameterVectorSizes: " << std::endl;
    for ( std::map<int, uint>::const_iterator pvsIt = parameterVectorSizes.begin(); pvsIt != parameterVectorSizes.end(); pvsIt++ )
    {
      std::cerr << pvsIt->first << " " << pvsIt->second << " ";
    }
    std::cerr << std::endl;
  }

  t1.start();
  this->updateEigenVectors();
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the eigenvectors-objects: " << t1.getLast() << std::endl;

  t1.start();
  if ( usePreviousAlphas )
  {
    std::map<int, NICE::Vector>::const_iterator binaryLabelsIt = binaryLabels.begin();
    std::vector<NICE::Vector>::const_iterator eigenMaxIt = eigenMax.begin();
    
    for ( std::map<int, NICE::Vector>::iterator lastAlphaIt = lastAlphas.begin() ;lastAlphaIt != lastAlphas.end(); lastAlphaIt++ )
    {
      //make sure that we only work on currently known classes in this loop
      while ( newClasses.find( labelIt->first ) != newClasses.end())
      {
        labelIt++;
        //since we already updated the eigenvalues, they contain the eigenvalues for the new classes as well.
        if (learnBalanced)
        {
          eigenMaxIt++;
        }
      }      
      int oldSize ( lastAlphaIt->second.size() );
      lastAlphaIt->second.resize ( oldSize + x.size() );

      //We initialize it with the same values as we use in GPLikelihoodApprox in batch training
      //default in GPLikelihoodApprox for the first time:
      // alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );

      double maxEigenValue ( 1.0 );
      if ( (*eigenMaxIt).size() > 0 )
        maxEigenValue = (*eigenMaxIt)[0];
      double factor ( 1.0 / maxEigenValue );    

      for ( uint i = 0; i < x.size(); i++ )
      {
        if ( binaryLabelsIt->second[oldSize+i] > 0 ) //we only have +1 and -1, so this might be benefitial in terms of speed
          lastAlphaIt->second[oldSize+i] = factor;
        else
          lastAlphaIt->second[oldSize+i] = -factor; //we follow the initialization as done in previous steps
          //lastAlphaIt->second[oldSize+i] = 0.0; // following the suggestion of Yeh and Darrell
      }

      binaryLabelsIt++;
      
      if (learnBalanced)
      {
        eigenMaxIt++;
      }
    }

    //compute unaffected alpha-vectors for the new classes
    eigenMaxIt = eigenMax.begin();
    std::set<int>::const_iterator clIt = knownClasses.begin();

    for (std::set<int>::const_iterator newClIt =  newClasses.begin(); newClIt != newClasses.end(); newClIt++)
    {
      if (learnBalanced)
      {
        //go to the position of the new class
        while (*clIt < *newClIt)
        {
          eigenMaxIt++;
          clIt++;
        }
      }
      
      double maxEigenValue ( 1.0 );
      if ( (*eigenMaxIt).size() > 0 )
        maxEigenValue = (*eigenMaxIt)[0];
      
      NICE::Vector alphaVec = (binaryLabels[*newClIt] * (1.0 / maxEigenValue) ); //see GPLikelihoodApprox for an explanation
      lastAlphas.insert( std::pair<int, NICE::Vector>(*newClIt, alphaVec) );
    }      

    for ( std::map<int, GPLikelihoodApprox * >::iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      gpLikeIt->second->setLastAlphas ( &lastAlphas );
    }
  }
  
  //if we do not use previous alphas, we do not have to set up anything here  
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the alpha-objects: " << t1.getLast() << std::endl;  

  if ( verbose )
    std::cerr << "resulting eigenvalues of first class: " << eigenMax[0] << std::endl;

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
  //NOTE we can skip this, if we do not want to change our parameters given new examples
  if ( performOptimizationAfterIncrement )
  {
    t1.start();
    this->performOptimization ( gplikes, parameterVectorSizes, false /* initialize not with default values but using the last solution */ );
    t1.stop();
    if (verboseTime)
      std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;
    
    t1.start();
    this->transformFeaturesWithOptimalParameters ( gplikes, parameterVectorSizes );
    t1.stop();
    if (verboseTime)
      std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;
  }
  else
  {
    //deactivate the optimization method;
    int originalOptimizationMethod = optimizationMethod;
    this->optimizationMethod = OPT_NONE;
    //and deactive the noise-optimization as well
    if (optimizeNoise) this->optimizeNoise = false;
    
    t1.start();
    //this is needed to compute the alpha vectors for the standard parameter settings
    this->performOptimization ( gplikes, parameterVectorSizes, false /* initialize not with default values but using the last solution */ );
    t1.stop();
    std::cerr << "skip optimization after increment" << std::endl;
    if (verboseTime)
      std::cerr << "Time used for performing the optimization: " << t1.getLast() << std::endl;

    std::cerr << "skip feature transformation" << std::endl;
    if (verboseTime)
      std::cerr << "Time used for transforming features with optimal parameters: " << t1.getLast() << std::endl;
    
    //re-activate the optimization method
    this->optimizationMethod = originalOptimizationMethod;    
  }

  if ( verbose )
    cerr << "Preparing after retraining for classification ..." << endl;


  //NOTE unfortunately, the whole vector alpha differs, and not only its last entry.
  // If we knew any method, which could update this efficiently, we could also compute A and B more efficiently by updating them.
  // Since we are not aware of any such method, we have to compute them completely new
  // :/
  t1.start();
  this->computeMatricesAndLUTs ( gplikes );
  t1.stop();
  if (verboseTime)
    std::cerr << "Time used for setting up the A'nB -objects: " << t1.getLast() << std::endl;

  t.stop();

  ResourceStatistics rs;
  std::cerr << "Time used for re-learning: " << t.getLast() << std::endl;
  long maxMemory;
  rs.getMaximumMemory ( maxMemory );
  std::cerr << "Maximum memory used: " << maxMemory << " KB" << std::endl;

  //don't waste memory
  if ( learnBalanced )
  {
    for ( std::map<int, GPLikelihoodApprox*>::const_iterator gpLikeIt = gplikes.begin(); gpLikeIt != gplikes.end(); gpLikeIt++ )
    {
      delete gpLikeIt->second;
    }
  }
  else
  {
    delete gplikes.begin()->second;
  }
  gplikes.clear();//TODO check whether this is useful or not
}

void FMKGPHyperparameterOptimization::prepareVarianceApproximation()
{
  PrecomputedType AVar;
  fmk->hikPrepareKVNApproximation ( AVar );

  precomputedAForVarEst = AVar;
  precomputedAForVarEst.setIoUntilEndOfFile ( false );

  if ( q != NULL )
  {
    //do we have results from previous runs but called this method nonetheless?
    //then delete it and compute it again
    if (precomputedTForVarEst != NULL)
      delete precomputedTForVarEst;
    
    double *T = fmk->hikPrepareLookupTableForKVNApproximation ( *q, pf );
    precomputedTForVarEst = T;
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
  for ( map<int, PrecomputedType>::const_iterator i = precomputedA.begin() ; i != precomputedA.end(); i++ )
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
      map<int, PrecomputedType>::const_iterator j = precomputedB.find ( classno );
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
  
  if ( precomputedA.size() > 1 ) {
    // multi-class classification
    return scores.maxElement();
  } else {
    // binary setting
    // FIXME: not really flexible for every situation
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
  for ( map<int, PrecomputedType>::const_iterator i = precomputedA.begin() ; i != precomputedA.end(); i++ )
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
      map<int, PrecomputedType>::const_iterator j = precomputedB.find ( classno );
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
  
  if ( precomputedA.size() > 1 ) {
    // multi-class classification
    return scores.maxElement();
  } else {
    // binary setting
    // FIXME: not really flexible for every situation
    scores[binaryLabelNegative] = -scores[binaryLabelPositive]; 
    
    return scores[ binaryLabelPositive ] <= 0.0 ? binaryLabelNegative : binaryLabelPositive;
  }
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateRough ( const NICE::SparseVector & x, NICE::Vector & predVariances ) const
{
  double kSelf ( 0.0 );
  for ( NICE::SparseVector::const_iterator it = x.begin(); it != x.end(); it++ )
  {
    kSelf += pf->f ( 0, it->second );
    // if weighted dimensions:
    //kSelf += pf->f(it->first,it->second);
  }

  double normKStar;

  if ( q != NULL )
  {
    if ( precomputedTForVarEst == NULL )
    {
      fthrow ( Exception, "The precomputed LUT for uncertainty prediction is NULL...have you prepared the uncertainty prediction?" );
    }
    fmk->hikComputeKVNApproximationFast ( precomputedTForVarEst, *q, x, normKStar );
  }
  else
  {
    fmk->hikComputeKVNApproximation ( precomputedAForVarEst, x, normKStar, pf );
  }

  predVariances.clear();
  predVariances.resize( eigenMax.size() );
  
  // for balanced setting, we get approximations for every binary task
  int cnt( 0 );
  for (std::vector<NICE::Vector>::const_iterator eigenMaxIt = eigenMax.begin(); eigenMaxIt != eigenMax.end(); eigenMaxIt++, cnt++)
  {
    predVariances[cnt] = kSelf - ( 1.0 / (*eigenMaxIt)[0] )* normKStar;
  }
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceApproximateFine ( const NICE::SparseVector & x, NICE::Vector & predVariances ) const
{
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
  
  std::vector<NICE::Vector>::const_iterator eigenMaxIt = eigenMax.begin();
  
  predVariances.clear();
  predVariances.resize( eigenMax.size() );  

  int classIdx( 0 );
  // for balanced setting, we get approximations for every binary task
  for (std::vector< NICE::Matrix>::const_iterator eigenMaxVectorIt = eigenMaxVectors.begin(); eigenMaxVectorIt != eigenMaxVectors.end(); eigenMaxVectorIt++, eigenMaxIt++, classIdx++)
  {
    
    double currentSecondTerm ( 0.0 );
    double sumOfProjectionLengths ( 0.0 );

    if ( ( kStar.size() != (*eigenMaxVectorIt).rows() ) || ( kStar.size() <= 0 ) )
    {
      //NOTE output?
    }

//     NICE::Vector multiplicationResults; // will contain nrOfEigenvaluesToConsiderForVarApprox many entries
//     multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    NICE::Vector multiplicationResults( nrOfEigenvaluesToConsiderForVarApprox, 0.0 );
    //ok, there seems to be a nasty thing in computing multiplicationResults.multiply ( *eigenMaxVectorIt, kStar, true/* transpose */ );
    //wherefor it takes aeons...
    //so we compute it by ourselves
    for ( uint tmpI = 0; tmpI < kStar.size(); tmpI++)
    {
      double kStarI ( kStar[tmpI] );
      for ( int tmpJ = 0; tmpJ < nrOfEigenvaluesToConsiderForVarApprox; tmpJ++)
      {
        multiplicationResults[tmpJ] += kStarI * (*eigenMaxVectorIt)(tmpI,tmpJ);
      }
    }

    double projectionLength ( 0.0 );
    int cnt ( 0 );
    NICE::Vector::const_iterator it = multiplicationResults.begin();

    while ( cnt < ( nrOfEigenvaluesToConsiderForVarApprox - 1 ) )
    {
      projectionLength = ( *it );
      currentSecondTerm += ( 1.0 / (*eigenMaxIt)[cnt] ) * pow ( projectionLength, 2 );
      sumOfProjectionLengths += pow ( projectionLength, 2 );
      it++;
      cnt++;
    }

    double normKStar ( pow ( kStar.normL2 (), 2 ) );

    currentSecondTerm += ( 1.0 / (*eigenMaxIt)[nrOfEigenvaluesToConsiderForVarApprox-1] ) * ( normKStar - sumOfProjectionLengths );

    if ( ( normKStar - sumOfProjectionLengths ) < 0 )
    {
  //     std::cerr << "Attention: normKStar - sumOfProjectionLengths is smaller than zero -- strange!" << std::endl;
    }
    predVariances[classIdx] = kSelf - currentSecondTerm; 
  }
}

void FMKGPHyperparameterOptimization::computePredictiveVarianceExact ( const NICE::SparseVector & x, NICE::Vector & predVariances ) const
{
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

  // for balanced setting, we get uncertainties for every binary task 
  std::vector<NICE::Vector>::const_iterator eigenMaxIt = eigenMax.begin();
  
  predVariances.clear();
  predVariances.resize( eigenMax.size() );  

  int cnt( 0 );
  for (std::map<int, IKMLinearCombination * >::const_iterator ikmSumIt = ikmsums.begin(); ikmSumIt != ikmsums.end(); ikmSumIt++, eigenMaxIt++, cnt++ )
  {  
    //now run the ILS method
    NICE::Vector diagonalElements;
    ikmSumIt->second->getDiagonalElements ( diagonalElements );

//     t.start();
    // init simple jacobi pre-conditioning
    ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( linsolver );
  

    //perform pre-conditioning
    if ( linsolver_cg != NULL )
      linsolver_cg->setJacobiPreconditioner ( diagonalElements );
   

    Vector beta;
    
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
    beta = (kStar * (1.0 / (*eigenMaxIt)[0]) );
/*    t.stop();
  std::cerr << "ApproxExact -- time for preconditioning etc: "  << t.getLast()  << std::endl;    
    
  t.start();*/
    //   t.start();
    linsolver->solveLin ( * ( ikmSumIt->second ), kStar, beta );
    //   t.stop();
//     t.stop();
//         t.stop();
//   std::cerr << "ApproxExact -- time for lin solve: "  << t.getLast()  << std::endl;

    beta *= kStar;
    
    double currentSecondTerm( beta.Sum() );
    predVariances[cnt] = kSelf - currentSecondTerm;
  }
}

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void FMKGPHyperparameterOptimization::restore ( std::istream & is, int format )
{
  if ( is.good() )
  {
    //load the underlying data
    if (fmk != NULL)
      delete fmk;
    fmk = new FastMinKernel;
    fmk->restore(is,format);    
    
    //now set up the GHIK-things in ikmsums
    for ( std::map<int, IKMLinearCombination * >::iterator it = ikmsums.begin(); it != ikmsums.end(); it++ )
    {
      it->second->addModel ( new GMHIKernel ( fmk, this->pf, this->q ) );
    }    
    
    is.precision ( numeric_limits<double>::digits10 + 1 );

    string tmp;
    is >> tmp; //class name

    is >> tmp;
    is >> learnBalanced;
    
    is >> tmp; //precomputedA:
    is >> tmp; //size:

    int preCompSize ( 0 );
    is >> preCompSize;
    precomputedA.clear();
    
    std::cerr << "precomputedA.size(): "<< preCompSize << std::endl;

    for ( int i = 0; i < preCompSize; i++ )
    {
      int nr;
      is >> nr;
      PrecomputedType pct;
      pct.setIoUntilEndOfFile ( false );
      pct.restore ( is, format );
      precomputedA.insert ( std::pair<int, PrecomputedType> ( nr, pct ) );
    }
    
    is >> tmp; //precomputedB:
    is >> tmp; //size:

    is >> preCompSize;
    precomputedB.clear();

    for ( int i = 0; i < preCompSize; i++ )
    {
      int nr;
      is >> nr;
      PrecomputedType pct;
      pct.setIoUntilEndOfFile ( false );
      pct.restore ( is, format );
      precomputedB.insert ( std::pair<int, PrecomputedType> ( nr, pct ) );
    }    
    
    is >> tmp;
    int precomputedTSize;
    is >> precomputedTSize;

    precomputedT.clear();

    if ( precomputedTSize > 0 )
    {
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

    //now restore the things we need for the variance computation
    is >> tmp;
    int sizeOfAForVarEst;
    is >> sizeOfAForVarEst;
    if ( sizeOfAForVarEst > 0 )
    
    if (precomputedAForVarEst.size() > 0)
    {
      precomputedAForVarEst.setIoUntilEndOfFile ( false );
      precomputedAForVarEst.restore ( is, format );
    }    

    is >> tmp; //precomputedTForVarEst
    is >> tmp; // NOTNULL or NULL
    if (tmp.compare("NOTNULL") == 0)
    {
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
      if (precomputedTForVarEst != NULL)
        delete precomputedTForVarEst;
    }
    
    //restore eigenvalues and eigenvectors
    is >> tmp; //eigenMax.size():
    int eigenMaxSize;
    is >> eigenMaxSize;
    
    for (int i = 0; i < eigenMaxSize; i++)
    {
      NICE::Vector eigenMaxEntry;
      is >> eigenMaxEntry;
      eigenMax.push_back( eigenMaxEntry );
    }
    
    is >> tmp; //eigenMaxVector.size():
    int eigenMaxVectorsSize;
    is >> eigenMaxVectorsSize;
    
    for (int i = 0; i < eigenMaxVectorsSize; i++)
    {
      NICE::Matrix eigenMaxVectorsEntry;
      is >> eigenMaxVectorsEntry;
      eigenMaxVectors.push_back( eigenMaxVectorsEntry );
    }       

    is >> tmp; //ikmsums:
    is >> tmp; //size:
    int ikmSumsSize ( 0 );
    is >> ikmSumsSize;
    ikmsums.clear();

    for ( int i = 0; i < ikmSumsSize; i++ )
    {
      int clNr ( 0 );
      is >> clNr;

      IKMLinearCombination *ikmsum = new IKMLinearCombination ();

      int nrOfModels ( 0 );
      is >> tmp;
      is >> nrOfModels;

      //the first one is always our noise-model
      IKMNoise * ikmnoise = new IKMNoise ();
      ikmnoise->restore ( is, format );

      ikmsum->addModel ( ikmnoise );

      //NOTE are there any more models you added? then add them here respectively in the correct order

      ikmsums.insert ( std::pair<int, IKMLinearCombination*> ( clNr, ikmsum ) );

      //the last one is the GHIK - which we do not have to restore, but simple reset it lateron
    }
    
    //restore the class numbers for binary settings (if mc-settings, these values will be negative by default)
    is >> tmp; // "binaryLabelPositive: " 
    is >> binaryLabelPositive;
    is >> tmp; // " binaryLabelNegative: "
    is >> binaryLabelNegative;          
  }
  else
  {
    std::cerr << "InStream not initialized - restoring not possible!" << std::endl;
  }
}

void FMKGPHyperparameterOptimization::store ( std::ostream & os, int format ) const
{
  if ( os.good() )
  {
    fmk->store ( os, format );

    os.precision ( numeric_limits<double>::digits10 + 1 );

    os << "FMKGPHyperparameterOptimization" << std::endl;

    os << "learnBalanced: " << learnBalanced << std::endl;

    //we only have to store the things we computed, since the remaining settings come with the config file afterwards
    
    os << "precomputedA: size: " << precomputedA.size() << std::endl;
    std::map< int, PrecomputedType >::const_iterator preCompIt = precomputedA.begin();
    for ( uint i = 0; i < precomputedA.size(); i++ )
    {
      os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( os, format );
      preCompIt++;
    }
    os << "precomputedB: size: " << precomputedB.size() << std::endl;
    preCompIt = precomputedB.begin();
    for ( uint i = 0; i < precomputedB.size(); i++ )
    {
      os << preCompIt->first << std::endl;
      ( preCompIt->second ).store ( os, format );
      preCompIt++;
    }    
    
    
    os << "precomputedT.size(): " << precomputedT.size() << std::endl;
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

    //now store the things needed for the variance estimation
    
    os << "precomputedAForVarEst.size(): "<< precomputedAForVarEst.size() << std::endl;
    
    if (precomputedAForVarEst.size() > 0)
    {
      precomputedAForVarEst.store ( os, format );
      os << std::endl; 
    }
    
    if ( precomputedTForVarEst != NULL )
    {
      os << "precomputedTForVarEst NOTNULL" << std::endl;
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
      os << "precomputedTForVarEst NULL" << std::endl;
    }
    
    //store the eigenvalues and eigenvectors
    os << "eigenMax.size(): " << std::endl;
    os << eigenMax.size() << std::endl;
    
    for (std::vector<NICE::Vector>::const_iterator it = this->eigenMax.begin(); it != this->eigenMax.end(); it++)
    {
      os << *it << std::endl;
    }
    
    os << "eigenMaxVectors.size(): " << std::endl;
    os << eigenMaxVectors.size() << std::endl;
    
    for (std::vector<NICE::Matrix>::const_iterator it = eigenMaxVectors.begin(); it != eigenMaxVectors.end(); it++)
    {
      os << *it << std::endl;
    }      

    os << "ikmsums: size: " << ikmsums.size() << std::endl;

    std::map<int, IKMLinearCombination * >::const_iterator ikmSumIt = ikmsums.begin();

    for ( uint i = 0; i < ikmsums.size(); i++ )
    {
      os << ikmSumIt->first << std::endl;
      os << "numberOfModels: " << ( ikmSumIt->second )->getNumberOfModels() << std::endl;
      //the last one os always the GHIK, which we do not have to restore
      for ( int j = 0; j < ( ikmSumIt->second )->getNumberOfModels() - 1; j++ )
      {
        ( ( ikmSumIt->second )->getModel ( j ) )->store ( os, format );
      }
      ikmSumIt++;
    }
    
    //store the class numbers for binary settings (if mc-settings, these values will be negative by default)
    os << "binaryLabelPositive: " << binaryLabelPositive << " binaryLabelNegative: " << binaryLabelNegative << std::endl;
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void FMKGPHyperparameterOptimization::clear ( ) {};

void FMKGPHyperparameterOptimization::addExample ( const NICE::SparseVector & x, const double & label, const bool & performOptimizationAfterIncrement )
{
  this->labels.append ( label );
  //have we seen this class already?
  if (knownClasses.find( label ) == knownClasses.end() )
  {
    knownClasses.insert( label );
    newClasses.insert( label );
  }    

  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  Timer t;
  t.start();
  fmk->addExample ( x, pf );
  t.stop();
  if (verboseTime)
    std::cerr << "Time used for adding the data to the fmk object: " << t.getLast() << std::endl;

  //TODO update the matrix for variance computations as well!!!
  
  // update the corresponding matrices A, B and lookup tables T  
  // optional: do the optimization again using the previously known solutions as initialization
  updateAfterSingleIncrement ( x, performOptimizationAfterIncrement );
  
  //clean up
  newClasses.clear();
}

void FMKGPHyperparameterOptimization::addMultipleExamples ( const std::vector<const NICE::SparseVector*> & newExamples, const NICE::Vector & _labels, const bool & performOptimizationAfterIncrement )
{
  if (this->knownClasses.size() == 1) //binary setting
  {
    int oldSize ( this->labels.size() );
    this->labels.resize ( this->labels.size() + _labels.size() );
    for ( uint i = 0; i < _labels.size(); i++ )
    {
      this->labels[i+oldSize] = _labels[i];
      //have we seen this class already?
      if ( (_labels[i]  != this->binaryLabelPositive) && (_labels[i]  != this->binaryLabelNegative) )
      {
        fthrow(Exception, "Binary setting does not allow adding new classes so far");
//         knownClasses.insert( _labels[i] );
//         newClasses.insert( _labels[i] );
      }
    }      
  }
  else //multi-class setting
  {
    int oldSize ( this->labels.size() );
    this->labels.resize ( this->labels.size() + _labels.size() );
    for ( uint i = 0; i < _labels.size(); i++ )
    {
      this->labels[i+oldSize] = _labels[i];
      //have we seen this class already?
      if (knownClasses.find( _labels[i] ) == knownClasses.end() )
      {
        knownClasses.insert( _labels[i] );
        newClasses.insert( _labels[i] );
      }
    }    
  }
  


  // add the new example to our data structure
  // It is necessary to do this already here and not lateron for internal reasons (see GMHIKernel for more details)
  Timer t;
  t.start();
  for ( std::vector<const NICE::SparseVector*>::const_iterator exampleIt = newExamples.begin(); exampleIt != newExamples.end(); exampleIt++ )
  {
    fmk->addExample ( **exampleIt , pf );
  }
  t.stop();
  if (verboseTime)
    std::cerr << "Time used for adding the data to the fmk object: " << t.getLast() << std::endl;
  
  Timer tVar;
  tVar.start();  
  //do we need  to update our matrices?
  if ( precomputedAForVarEst.size() != 0)
  {
    std::cerr << "update the variance matrices " << std::endl;
    //this computes everything from the scratch
    this->prepareVarianceApproximation();
    //this would perform a more sophisticated update
    //unfortunately, there is a bug somewhere
    //TODO fixme!
//     std::cerr << "update the LUTs needed for variance computation" << std::endl;
//     for ( std::vector<const NICE::SparseVector*>::const_iterator exampleIt = newExamples.begin(); exampleIt != newExamples.end(); exampleIt++ )
//     {  
//       std::cerr << "new example: " << std::endl;
//       (**exampleIt).store(std::cerr);
//       std::cerr << "now update the LUT for var est" << std::endl;
//       fmk->updatePreparationForKVNApproximation( **exampleIt, precomputedAForVarEst, pf );  
//       if ( q != NULL )
//       {
//         fmk->updateLookupTableForKVNApproximation( **exampleIt, precomputedTForVarEst, *q, pf );
//       }
//     }
//     std::cerr << "update of LUTs for variance compuation done" << std::endl;
  }
  tVar.stop();
  if (verboseTime)
    std::cerr << "Time used for computing the Variance Matrix and LUT: " << tVar.getLast() << std::endl;  
  


  // update the corresponding matrices A, B and lookup tables T
  // optional: do the optimization again using the previously known solutions as initialization
  updateAfterMultipleIncrements ( newExamples, performOptimizationAfterIncrement );
  
  //clean up
  newClasses.clear();
}
