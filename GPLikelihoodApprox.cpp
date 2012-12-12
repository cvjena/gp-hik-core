/** 
* @file GPLikelihoodApprox.cpp
* @brief GP likelihood approximation as a cost function (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/09/2012

*/
#include <iostream>

#include <core/algebra/CholeskyRobust.h>
#include <core/vector/Algorithms.h>
#include <core/vector/Eigen.h>

#include <core/basics/Timer.h>
#include <core/algebra/ILSConjugateGradients.h>
#include "kernels/GeneralizedIntersectionKernelFunction.h"
#include "kernels/IntersectionKernelFunction.h"


#include "GPLikelihoodApprox.h"
#include "IKMLinearCombination.h"
#include "GMHIKernel.h"
#include "algebra/LogDetApproxBaiAndGolub.h"


using namespace std;
using namespace NICE;
using namespace OPTIMIZATION;


GPLikelihoodApprox::GPLikelihoodApprox( const map<int, Vector> & binaryLabels,
                                        ImplicitKernelMatrix *ikm,
                                        IterativeLinearSolver *linsolver, 
                                        EigValues *eig,
                                        bool verifyApproximation,
                                        int _nrOfEigenvaluesToConsider
                                      ) 

      : CostFunction( ikm->getNumParameters() )
{
  this->binaryLabels = binaryLabels;
  this->ikm = ikm;
  this->linsolver = linsolver;
  this->eig = eig;

  if ( binaryLabels.size() == 1 )
    this->nrOfClasses = 2;
  else
    this->nrOfClasses = binaryLabels.size();

  this->min_nlikelihood = std::numeric_limits<double>::max();
  this->verifyApproximation = verifyApproximation;
  
  this->nrOfEigenvaluesToConsider = _nrOfEigenvaluesToConsider;
  
  lastAlphas = NULL;
  
  this->verbose = false;
  this->debug = false;
  
  this->usePreviousAlphas = true;

}

GPLikelihoodApprox::~GPLikelihoodApprox()
{
  //delete the pointer, but not the content (which is stored somewhere else)
  if (lastAlphas != NULL)
    lastAlphas = NULL;  
}

void GPLikelihoodApprox::calculateLikelihood ( double mypara, const FeatureMatrix & f, const std::map< int, NICE::Vector > & yset, double noise, double lambdaMax )
{
  // robust cholesky routine without noise !!
  CholeskyRobust cr ( true /*verbose*/, 0.0, false /*useCuda*/ );

  Timer t;
  t.start();
  cerr << "VERIFY: Calculating kernel matrix ..." << endl;
  Matrix K;
  IntersectionKernelFunction<double> hik;
  //old version, not needed anymore - we explore sparsity
//   K = hik.computeKernelMatrix(data_matrix, noise); // = K + sigma^2 I
  K = hik.computeKernelMatrix(f, noise);
  t.stop();
  cerr << "VERIFY: Time used for calculating kernel matrix is: " << t.getLast() << endl;

  cerr << "K is a " << K.rows() << " x " << K.cols() << " matrix" << endl;

  if ( K.containsNaN() ) 
    fthrow(Exception, "NaN values in the kernel matrix");

  cerr << "VERIFY: Computing likelihood ..." << endl;
  t.start();
  Matrix choleskyMatrix; 
  cr.robustChol ( K, choleskyMatrix ); // K = choleskyMatrix^T * choleskyMatrix
  double gt_logdet = (yset.size()) * cr.getLastLogDet();
  cerr << "chol * chol^T: " << ( choleskyMatrix * choleskyMatrix.transpose() )(0,0,4,4) << endl;

  double gt_dataterm = 0.0;
  for ( map< int, NICE::Vector >::const_iterator i = yset.begin(); i != yset.end(); i++ )
  {
    const Vector & y = i->second;
    Vector gt_alpha;
    choleskySolve ( choleskyMatrix, y, gt_alpha );
    cerr << "cholesky error: " << (K*gt_alpha - y).normL2() << endl;
    gt_dataterm += y.scalarProduct ( gt_alpha );
  }
  //cerr << "linsolve error: " << (tmp - y).normL2() << endl;
  t.stop();
  cerr << "VERIFY: Time used for calculating likelihood: " << t.getLast() << endl;
  
  cerr << "Something of K: " << K(0, 0, 4, 4) << endl;
  cerr << "frob norm: gt:" << K.frobeniusNorm() << endl;
  
  /*try {
    Vector *eigenv = eigenvalues ( K ); 
    cerr << "lambda_max: gt:" << eigenv->Max() << " est:" << lambdaMax << endl; 
    delete eigenv;
  } catch (...) {
    cerr << "NICE eigenvalues function failed!" << endl;
  }*/

  double gt_nlikelihood = gt_logdet + gt_dataterm;
  cerr << "OPTGT: " << mypara << " " << gt_nlikelihood << " " << gt_logdet << " " << gt_dataterm << endl;
}


double GPLikelihoodApprox::evaluate(const OPTIMIZATION::matrix_type & x)
{
  Vector xv;
 
  xv.resize ( x.rows() );
  for ( uint i = 0 ; i < x.rows(); i++ )
    xv[i] = x(i,0);

  // check whether we have been here before
  unsigned long hashValue = xv.getHashValue();
  if (verbose)  
    std::cerr << "Current parameter: " << xv << " (weird hash value is " << hashValue << ")" << std::endl;
  map<unsigned long, double>::const_iterator k = alreadyVisited.find(hashValue);
  
  if ( k != alreadyVisited.end() )
  {
    if (verbose)
      std::cerr << "Using cached value: " << k->second << std::endl;
    
    //already computed, simply return the cached value
    return k->second;
  }

  // set parameter value and check lower and upper bounds of pf
  if ( ikm->outOfBounds(xv) )
  {
    if (verbose)
      std::cerr << "Parameters are out of bounds" << std::endl;
    return numeric_limits<double>::max();
  }
  
  ikm->setParameters ( xv );
  if (verbose)  
    std::cerr << "setParameters xv: " << xv << std::endl;

  // --- calculate the likelihood
  // (a) logdet(K + sI)
  Timer t;
  if (verbose)  
    std::cerr << "Calculating eigendecomposition " << ikm->rows() << " x " << ikm->cols() << std::endl;
  t.start();
  Vector eigenmax;
  Matrix eigenmaxvectors;
 
  int rank = nrOfEigenvaluesToConsider;

  /** calculate the biggest eigenvalue */
  // We use a Arnoldi solver and not a specialized one.....
  // We might also think about a coordinate descent solver for Arnoldi iterations, however,
  // the current implementation converges very quickly
  //old version: just use the first eigenvalue
  
  //NOTE
  // in theory, we have these values already on hand since we've done it in FMKGPHypOpt.
  // Think about wether to give them as input to this function or not
  eig->getEigenvalues( *ikm, eigenmax, eigenmaxvectors, rank ); 
  if (verbose)
    std::cerr << "eigenmax: " << eigenmax << std::endl;
      
  t.stop();

  SparseVector binaryDataterms;
  Vector diagonalElements;
  
  ikm->getDiagonalElements ( diagonalElements );

  // set simple jacobi pre-conditioning
  ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( linsolver );

  //TODO why do we need this?  
  if ( linsolver_cg != NULL )
    linsolver_cg->setJacobiPreconditioner ( diagonalElements );
  

  // all alpha vectors will be stored!
  map<int, Vector> alphas;

  // This has to be done m times for the multi-class case
  if (verbose)
    std::cerr << "run ILS for every bin label. binaryLabels.size(): " << binaryLabels.size() << std::endl;
  for ( map<int, Vector>::const_iterator j = binaryLabels.begin(); j != binaryLabels.end() ; j++)
  {
    // (b) y^T (K+sI)^{-1} y
    int classCnt = j->first;
    if (verbose)
    {
      std::cerr << "Solving linear equation system for class " << classCnt << " ..." << std::endl;
      std::cerr << "Size of the kernel matrix " << ikm->rows() << std::endl;
    }

    /** About finding a good initial solution
     * K~ = K + sigma^2 I
     *
     * (0) we have already estimated alpha for a previous hyperparameter, then
     *     we should use this as an initial estimate. According to my quick
     *     tests this really helps!
     * (1) K~ \approx lambda_max v v^T
     * \lambda_max v v^T * alpha = y     | multiply with v^T from left
     * => \lambda_max v^T alpha = v^T y
     * => alpha = y / lambda_max could be a good initial start
     * If we put everything in the first equation this gives us
     * v = y ....which is somehow a weird assumption (cf Kernel PCA)
     *  This reduces the number of iterations by 5 or 8
     */
    Vector alpha;
    
    if ( (usePreviousAlphas) && (lastAlphas != NULL) )
    {
      std::map<int, NICE::Vector>::iterator alphaIt = lastAlphas->begin();
      alpha = (*lastAlphas)[classCnt];
    }
    else  
    {
      alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );
    }
    
    Vector initialAlpha;
    if ( verbose )
     initialAlpha = alpha;

    if ( verbose )
      cerr << "Using the standard solver ..." << endl;

    t.start();
    linsolver->solveLin ( *ikm, binaryLabels[classCnt], alpha );
    t.stop();
   
    //TODO This is only important for the incremental learning stuff.
//     if ( verbose )
//     {
//       double initialAlphaNorm ( initialAlpha.normL1() );
//       //compute the difference
//       initialAlpha -= alpha;
//       //take the abs of the differences
//       initialAlpha.absInplace();
//       //and compute a final score using a suitable norm
// //       double difference( initialAlpha.normInf() );
//       double difference( initialAlpha.normL1() );
//       std::cerr << "debug -- last entry of new alpha: " << abs(alpha[alpha.size() -1 ]) << std::endl;
//       std::cerr << "debug -- difference using inf norm: " << difference  << std::endl;
//       std::cerr << "debug -- relative difference using inf norm: " << difference / initialAlphaNorm  << std::endl;
//     }


    if ( verbose )
      std::cerr << "Time used for solving (K + sigma^2 I)^{-1} y: " << t.getLast() << std::endl;
    // this term is no approximation at all
    double dataterm = binaryLabels[classCnt].scalarProduct(alpha);
    binaryDataterms[classCnt] = (dataterm);

    alphas[classCnt] = alpha;
  }
  
  // approximation stuff
  if (verbose)  
    cerr << "Approximating logdet(K) ..." << endl;
  t.start();
  LogDetApproxBaiAndGolub la;
  la.setVerbose(this->verbose);

  //NOTE: this is already the squared frobenius norm, that we are looking for.
  double frobNormSquared(0.0);
  
  // ------------- LOWER BOUND, THAT IS USED --------------------
  // frobNormSquared ~ \sum \lambda_i^2 <-- LOWER BOUND
  for (int idx = 0; idx < rank; idx++)
  {
    frobNormSquared += (eigenmax[idx] * eigenmax[idx]);
  }

                
  if (verbose)
    cerr << " frob norm squared: est:" << frobNormSquared << endl;
  if (verbose)  
    std::cerr << "trace: " << diagonalElements.Sum() << std::endl;
  double logdet = la.getLogDetApproximationUpperBound( diagonalElements.Sum(), /* trace = n only for non-transformed features*/
                             frobNormSquared, /* use a rough approximation of the frobenius norm */
                             eigenmax[0], /* upper bound for eigen values */
                             ikm->rows() /* = n */ 
                          );
  
  t.stop();
  
  if (verbose)
    cerr << "Time used for approximating logdet(K): " << t.getLast() << endl;

  // (c) adding the two terms
  double nlikelihood = nrOfClasses*logdet;
  double dataterm = binaryDataterms.sum();
  nlikelihood += dataterm;

  if (verbose)
    cerr << "OPT: " << xv << " " << nlikelihood << " " << logdet << " " << dataterm << endl;

  if ( nlikelihood < min_nlikelihood )
  {
    min_nlikelihood = nlikelihood;
    ikm->getParameters ( min_parameter );
    min_alphas = alphas;
  }

  alreadyVisited.insert ( pair<int, double> ( hashValue, nlikelihood ) );
  return nlikelihood;
}

void GPLikelihoodApprox::setParameterLowerBound(const double & _parameterLowerBound)
{
  parameterLowerBound = _parameterLowerBound;
}
  
void GPLikelihoodApprox::setParameterUpperBound(const double & _parameterUpperBound)
{
  parameterUpperBound = _parameterUpperBound;
}

void GPLikelihoodApprox::setLastAlphas(std::map<int, NICE::Vector> * _lastAlphas)
{
  lastAlphas = _lastAlphas;
}

void GPLikelihoodApprox::setBinaryLabels(const std::map<int, Vector> & _binaryLabels)
{
  binaryLabels = _binaryLabels;
}

void GPLikelihoodApprox::setUsePreviousAlphas( const bool & _usePreviousAlphas )
{
  this->usePreviousAlphas = _usePreviousAlphas; 
}

void GPLikelihoodApprox::setVerbose( const bool & _verbose )
{
  this->verbose = _verbose;
}

void GPLikelihoodApprox::setDebug( const bool & _debug )
{
  this->debug = _debug;
}