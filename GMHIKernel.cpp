/** 
* @file GMHIKernel.cpp
* @brief Fast multiplication with histogram intersection kernel matrices (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/02/2012

*/
#include <iostream>

#include <core/vector/VVector.h>
#include <core/basics/Timer.h>

#include "GMHIKernel.h"

using namespace NICE;
using namespace std;


GMHIKernel::GMHIKernel( FastMinKernel *_fmk, ParameterizedFunction *_pf, const Quantization *_q )
{
  this->fmk = _fmk;
  this->q = _q;
  this->pf = _pf;
  verbose = false;
  useOldPreparation = false;

}

GMHIKernel::~GMHIKernel()
{
}

/** multiply with a vector: A*x = y */
void GMHIKernel::multiply (NICE::Vector & y, const NICE::Vector & x) const
{
  //do we want to use any quantization at all?
  if ( this->q != NULL )
  {
    double *T;
    if (useOldPreparation)
    {
      NICE::VVector A; 
      NICE::VVector B; 
      // prepare to calculate sum_i x_i K(x,x_i)
      fmk->hik_prepare_alpha_multiplications(x, A, B);
      T = fmk->hik_prepare_alpha_multiplications_fast(A, B, this->q, pf);
    }
    else
    {
      T = fmk->hikPrepareLookupTable(x, this->q, pf );
    }
    fmk->hik_kernel_multiply_fast ( T, this->q, x, y ); 
    delete [] T;
  }
  else //no quantization
  {
    NICE::VVector A; 
    NICE::VVector B; 
    // prepare to calculate sum_i x_i K(x,x_i)
    fmk->hik_prepare_alpha_multiplications(x, A, B);
    
    if (verbose)
    {
      int sizeOfDouble (sizeof(double));
      int sizeOfA(0);
      int sizeOfB(0);
      for (uint i = 0; i < A.size(); i++)
      {
        sizeOfA += A[i].size();
      }
      for (uint i = 0; i < B.size(); i++)
      {
        sizeOfB += B[i].size();
      }
      sizeOfA*=sizeOfDouble;
      sizeOfB*=sizeOfDouble;
      
      std::cerr << "multiplySparse: sizeof(A) + sizeof(B): " << sizeOfA + sizeOfB << std::endl;
    }
    // y = K * x
    //we only need x as input argument to add x*noise to beta
    //all necessary information for the "real" multiplication is already stored in y
    fmk->hik_kernel_multiply(A, B, x, y);
  }
}

/** get the number of rows in A */
uint GMHIKernel::rows () const
{
  // return the number of examples
  return fmk->get_n();
}

/** get the number of columns in A */
uint GMHIKernel::cols () const
{
  // return the number of examples
  return fmk->get_n();
}

/** set verbose-flag needed for output of size of A and B */
void GMHIKernel::setVerbose ( const bool& _verbose )
{
  verbose = _verbose;
}

void GMHIKernel::setUseOldPreparation( const bool & _useOldPreparation)
{
  useOldPreparation = _useOldPreparation;
}

uint GMHIKernel::getNumParameters() const 
{
  if ( this->pf == NULL )
    return 0;
  else
    return this->pf->parameters().size();
}

void GMHIKernel::getParameters( NICE::Vector & parameters ) const
{
  if ( this->pf == NULL )
    parameters.clear();
  else {
    parameters.resize( this->pf->parameters().size() );
    parameters = this->pf->parameters();
  }
}

void GMHIKernel::setParameters( const NICE::Vector & parameters )
{
  if ( pf == NULL && parameters.size() > 0 )
    fthrow(Exception, "Unable to set parameters of a non-parameterized GMHIKernel object");

  pf->parameters() = parameters;
  
  fmk->applyFunctionToFeatureMatrix( pf );
}

void GMHIKernel::getDiagonalElements ( Vector & diagonalElements ) const
{
  fmk->featureMatrix().hikDiagonalElements(diagonalElements);
  // add sigma^2 I
  diagonalElements += fmk->getNoise();
}

void GMHIKernel::getFirstDiagonalElement ( double & diagonalElement ) const
{
  Vector diagonalElements;
  fmk->featureMatrix().hikDiagonalElements(diagonalElements);
  diagonalElement = diagonalElements[0];
  // add sigma^2 I
  diagonalElement += fmk->getNoise();
}
    
bool GMHIKernel::outOfBounds(const Vector & parameters) const
{
  if ( pf == NULL && parameters.size() > 0 )
    fthrow(Exception, "Unable to check the bounds of a parameter without any parameterization");
  
  Vector uB = pf->getParameterUpperBounds();
  Vector lB = pf->getParameterLowerBounds();
  if ( uB.size() != parameters.size() || lB.size() != parameters.size() )
    fthrow(Exception, "Dimension of lower/upper bound vector " << lB.size() << " and " << uB.size() << " does not match the size of the parameter vector " << parameters.size() << ".");
  for ( uint i = 0 ; i < parameters.size() ; i++ )
    if ( (parameters[i] < lB[i]) || (parameters[i] > uB[i]) )
    {
      if (verbose)
        std::cerr << "Parameter " << i << " is out of bounds: " << lB[i] << " <= " << parameters[i] << " <= " << uB[i] << std::endl;
      return true;
    }

  return false;
}

Vector GMHIKernel::getParameterLowerBounds() const
{
  if ( pf == NULL )
    fthrow(Exception, "Unable to get the bounds without any parameterization");
  return pf->getParameterLowerBounds();
}

Vector GMHIKernel::getParameterUpperBounds() const
{
  if ( pf == NULL )
    fthrow(Exception, "Unable to get the bounds without any parameterization");
  return pf->getParameterUpperBounds();
}

double GMHIKernel::approxFrobNorm() const
{
  return this->fmk->getFrobNormApprox();
}

void GMHIKernel::setApproximationScheme(const int & _approxScheme)
{
  this->fmk->setApproximationScheme(_approxScheme);
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void GMHIKernel::addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement
			   )
{
  //nothing has to be done here, the fmk-object got new examples already in outer struct (FMKGPHyperparameterOptimization)
}

void GMHIKernel::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement
				    )
{
  //nothing has to be done here, the fmk-object got new examples already in outer struct (FMKGPHyperparameterOptimization)
}
