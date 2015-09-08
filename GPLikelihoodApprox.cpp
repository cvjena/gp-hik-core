/** 
* @file GPLikelihoodApprox.cpp
* @brief GP likelihood approximation as a cost function (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/09/2012

*/

// STL includes
#include <iostream>

// NICE-core includes
#include <core/algebra/CholeskyRobust.h>
#include <core/algebra/ILSConjugateGradients.h>
// 
#include <core/basics/Timer.h>
// 
#include <core/vector/Algorithms.h>
#include <core/vector/Eigen.h>

//stuff used for verification only
#include "kernels/GeneralizedIntersectionKernelFunction.h"
#include "kernels/IntersectionKernelFunction.h"

// gp-hik-core includes
#include "gp-hik-core/GPLikelihoodApprox.h"
#include "gp-hik-core/IKMLinearCombination.h"
#include "gp-hik-core/GMHIKernel.h"
#include "gp-hik-core/algebra/LogDetApproxBaiAndGolub.h"


using namespace std;
using namespace NICE;
using namespace OPTIMIZATION;


GPLikelihoodApprox::GPLikelihoodApprox( const std::map<uint, NICE::Vector> & _binaryLabels,
                                        ImplicitKernelMatrix *_ikm,
                                        IterativeLinearSolver *_linsolver, 
                                        EigValues *_eig,
                                        bool _verifyApproximation,
                                        int _nrOfEigenvaluesToConsider
                                      ) 

      : CostFunction( _ikm->getNumParameters() )
{
  this->binaryLabels = _binaryLabels;
  this->ikm = _ikm;
  this->linsolver = _linsolver;
  this->eig = _eig;

  if ( _binaryLabels.size() == 1 )
    this->nrOfClasses = 2;
  else
    this->nrOfClasses = _binaryLabels.size();

  this->min_nlikelihood = std::numeric_limits<double>::max();
  this->verifyApproximation = _verifyApproximation;
  
  this->nrOfEigenvaluesToConsider = _nrOfEigenvaluesToConsider;
    
  this->verbose = false;
  this->debug = false;
  
  this->initialAlphaGuess = NULL;
}

GPLikelihoodApprox::~GPLikelihoodApprox()
{
  //we do not have to delete the memory here, since it will be handled externally...
  // TODO however, if we should copy the whole vector, than we also have to delete it here accordingly! Check this!
  if ( this->initialAlphaGuess != NULL )
    this->initialAlphaGuess = NULL;
}

const std::map<uint, Vector> & GPLikelihoodApprox::getBestAlphas () const
{
  if ( this->min_alphas.size() > 0 )
  {
  // did we already computed a local optimal solution?
    return this->min_alphas;
  }
  else if ( this->initialAlphaGuess != NULL)
  {
    std::cerr << "no known alpha vectors so far, take initial guess instaed" << std::endl;
    // computation not started, but initial guess was given, so use this one
    return *(this->initialAlphaGuess);
  }  
  
  // nothing known, min_alphas will be empty
  return this->min_alphas;
}

void GPLikelihoodApprox::calculateLikelihood ( double _mypara, 
                                               const FeatureMatrix & _f, 
                                               const std::map< uint, NICE::Vector > & _yset, 
                                               double _noise, 
                                               double lambdaMax 
                                             )
{
  // robust cholesky routine without noise !!
  CholeskyRobust cr ( true /*verbose*/, 0.0, false /*useCuda*/ );

  Timer t;
  t.start();
  cerr << "VERIFY: Calculating kernel matrix ..." << endl;
  Matrix K;
  IntersectionKernelFunction<double> hik;
  //old version, not needed anymore - we explore sparsity
//   K = hik.computeKernelMatrix(data_matrix, _noise); // = K + sigma^2 I
  K = hik.computeKernelMatrix(_f, _noise);
  t.stop();
  cerr << "VERIFY: Time used for calculating kernel matrix is: " << t.getLast() << endl;

  cerr << "K is a " << K.rows() << " x " << K.cols() << " matrix" << endl;

  if ( K.containsNaN() ) 
    fthrow(Exception, "NaN values in the kernel matrix");

  cerr << "VERIFY: Computing likelihood ..." << endl;
  t.start();
  Matrix choleskyMatrix; 
  cr.robustChol ( K, choleskyMatrix ); // K = choleskyMatrix^T * choleskyMatrix
  double gt_logdet = (_yset.size()) * cr.getLastLogDet();
  cerr << "chol * chol^T: " << ( choleskyMatrix * choleskyMatrix.transpose() )(0,0,4,4) << endl;

  double gt_dataterm = 0.0;
  for ( std::map< uint, NICE::Vector >::const_iterator i = _yset.begin(); i != _yset.end(); i++ )
  {
    const NICE::Vector & y = i->second;
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
  
  
  double gt_nlikelihood = gt_logdet + gt_dataterm;
  cerr << "OPTGT: " << _mypara << " " << gt_nlikelihood << " " << gt_logdet << " " << gt_dataterm << endl;
}

void GPLikelihoodApprox::computeAlphaDirect(const OPTIMIZATION::matrix_type & _x, 
                                            const NICE::Vector & _eigenValues 
                                           )
{
  Timer t;
  
  NICE::Vector diagonalElements; 
  ikm->getDiagonalElements ( diagonalElements );

  // set simple jacobi pre-conditioning
  ILSConjugateGradients *linsolver_cg = dynamic_cast<ILSConjugateGradients *> ( linsolver );

  if ( linsolver_cg != NULL )
    linsolver_cg->setJacobiPreconditioner ( diagonalElements );
  

  // all alpha vectors will be stored!
  std::map<uint, NICE::Vector> alphas;

  // This has to be done m times for the multi-class case
  if ( this->verbose )
    std::cerr << "run ILS for every bin label. binaryLabels.size(): " << binaryLabels.size() << std::endl;
  for ( std::map<uint, NICE::Vector>::const_iterator j = binaryLabels.begin(); j != binaryLabels.end() ; j++)
  {
    // (b) y^T (K+sI)^{-1} y
    uint classCnt = j->first;
    if ( this->verbose )
    {
      std::cerr << "Solving linear equation system for class " << classCnt << " ..." << std::endl;
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
    NICE::Vector alpha;
    
    alpha = (binaryLabels[classCnt] * (1.0 / _eigenValues[0]) );
    
    if ( verbose )
      std::cerr << "Using the standard solver ..." << std::endl;

    t.start();
    linsolver->solveLin ( *ikm, binaryLabels[classCnt], alpha );
    t.stop();
   
    alphas.insert( std::pair<uint, NICE::Vector> ( classCnt, alpha) );
  }  
  
  // save the parameter value and alpha vectors
  ikm->getParameters ( min_parameter );
  this->min_alphas = alphas;
}

double GPLikelihoodApprox::evaluate(const OPTIMIZATION::matrix_type & _x)
{
  NICE::Vector xv;
   
  xv.resize ( _x.rows() );
  for ( uint i = 0 ; i < _x.rows(); i++ )
    xv[i] = _x(i,0);

  // check whether we have been here before
  unsigned long hashValue = xv.getHashValue();
  if ( this->verbose )  
    std::cerr << "Current parameter: " << xv << " (weird hash value is " << hashValue << ")" << std::endl;
  std::map<unsigned long, double>::const_iterator k = alreadyVisited.find(hashValue);
  
  if ( k != alreadyVisited.end() )
  {
    if ( this->verbose )
      std::cerr << "Using cached value: " << k->second << std::endl;
    
    //already computed, simply return the cached value
    return k->second;
  }

  // set parameter value and check lower and upper bounds of pf
  if ( ikm->outOfBounds(xv) )
  {
    if ( this->verbose )
      std::cerr << "Parameters are out of bounds" << std::endl;
    return numeric_limits<double>::max();
  }
  
  ikm->setParameters ( xv );
  if ( this->verbose )  
    std::cerr << "setParameters xv: " << xv << std::endl;

  // --- calculate the likelihood
  // (a) logdet(K + sI)
  Timer t;
  if ( this->verbose )  
    std::cerr << "Calculating eigendecomposition " << ikm->rows() << " x " << ikm->cols() << std::endl;
  t.start();
  NICE::Vector eigenmax;
  NICE::Matrix eigenmaxvectors;
 
  int rank = nrOfEigenvaluesToConsider;

  /** calculate the biggest eigenvalue */
  // We use a Arnoldi solver and not a specialized one.....
  // We might also think about a coordinate descent solver for Arnoldi iterations, however,
  // the current implementation converges very quickly
  //old version: just use the first eigenvalue
  
  // we have to re-compute EV and EW in all cases, since we change the hyper parameter and thereby the kernel matrix 
  eig->getEigenvalues( *ikm, eigenmax, eigenmaxvectors, rank ); 
  if ( this->verbose )
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
  std::map<uint, NICE::Vector> alphas;

  // This has to be done m times for the multi-class case
  if ( this->verbose )
    std::cerr << "run ILS for every bin label. binaryLabels.size(): " << binaryLabels.size() << std::endl;
  
  for ( std::map<uint, NICE::Vector>::const_iterator j = binaryLabels.begin(); j != binaryLabels.end() ; j++)
  {
    // (b) y^T (K+sI)^{-1} y
    uint classCnt = j->first;
    if ( this->verbose )
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
    NICE::Vector alpha;
    if ( this->initialAlphaGuess != NULL )
    {
      std::map<uint, NICE::Vector>::iterator myIt = this->initialAlphaGuess->find(classCnt);
      if ( myIt != this->initialAlphaGuess->end() )
        alpha = myIt->second;
      else
      {
        //NOTE this should never happen in theory...
        alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );
      }
    }
    else
    {
      alpha = (binaryLabels[classCnt] * (1.0 / eigenmax[0]) );      
    }
    

    
    if ( verbose )
      cerr << "Using the standard solver ..." << endl;

    t.start();
    linsolver->solveLin ( *ikm, binaryLabels[classCnt], alpha );
    t.stop();
   

    if ( verbose )
      std::cerr << "Time used for solving (K + sigma^2 I)^{-1} y: " << t.getLast() << std::endl;
    // this term is no approximation at all
    double dataterm = binaryLabels[classCnt].scalarProduct(alpha);
    binaryDataterms[classCnt] = (dataterm);

    alphas[classCnt] = alpha;
  }
  
  // approximation stuff
  if ( this->verbose )  
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

                
  if ( this->verbose )
    cerr << " frob norm squared: est:" << frobNormSquared << endl;
  if ( this->verbose )  
    std::cerr << "trace: " << diagonalElements.Sum() << std::endl;
  double logdet = la.getLogDetApproximationUpperBound( diagonalElements.Sum(), /* trace = n only for non-transformed features*/
                             frobNormSquared, /* use a rough approximation of the frobenius norm */
                             eigenmax[0], /* upper bound for eigen values */
                             ikm->rows() /* = n */ 
                          );
  
  t.stop();
  
  if ( this->verbose )
    cerr << "Time used for approximating logdet(K): " << t.getLast() << endl;

  // (c) adding the two terms
  double nlikelihood = this->nrOfClasses*logdet;
  double dataterm    = binaryDataterms.sum();
  nlikelihood += dataterm;

  if ( this->verbose )
    cerr << "OPT: " << xv << " " << nlikelihood << " " << logdet << " " << dataterm << endl;

  if ( nlikelihood < min_nlikelihood )
  {
    min_nlikelihood = nlikelihood;
    ikm->getParameters ( min_parameter );
    this->min_alphas = alphas;
  }

  this->alreadyVisited.insert ( std::pair<unsigned long, double> ( hashValue, nlikelihood ) );
  return nlikelihood;
}

void GPLikelihoodApprox::setParameterLowerBound(const double & _parameterLowerBound)
{
  this->parameterLowerBound = _parameterLowerBound;
}
  
void GPLikelihoodApprox::setParameterUpperBound(const double & _parameterUpperBound)
{
  this->parameterUpperBound = _parameterUpperBound;
}

void GPLikelihoodApprox::setInitialAlphaGuess(std::map< uint, NICE::Vector >* _initialAlphaGuess)
{
  this->initialAlphaGuess = _initialAlphaGuess;
}


void GPLikelihoodApprox::setBinaryLabels(const std::map<uint, Vector> & _binaryLabels)
{
  this->binaryLabels = _binaryLabels;
}

void GPLikelihoodApprox::setVerbose( const bool & _verbose )
{
  this->verbose = _verbose;
}

void GPLikelihoodApprox::setDebug( const bool & _debug )
{
  this->debug = _debug;
}