/** 
* @file IKMLinearCombination.cpp
* @brief Combination of several (implicit) kernel matrices, such as noise matrix and gp-hik kernel matrix (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/14/2012

*/
#include <iostream>

#include "IKMLinearCombination.h"

using namespace NICE;
using namespace std;


IKMLinearCombination::IKMLinearCombination()
{
  verbose = false;
}

IKMLinearCombination::~IKMLinearCombination()
{
  if (matrices.size() != 0)
  {
    for (int i = 0; i < matrices.size(); i++)
      delete matrices[i];
  }
}


void IKMLinearCombination::getDiagonalElements ( Vector & diagonalElements ) const
{
  diagonalElements.resize ( rows() );
  diagonalElements.set(0.0);
  
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    Vector diagonalElementsSingle;
    ikm->getDiagonalElements ( diagonalElementsSingle );
    diagonalElements += diagonalElementsSingle;
  }
}

void IKMLinearCombination::getFirstDiagonalElement ( double & diagonalElement ) const
{
  diagonalElement = 0.0;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    double firstElem;
    ikm->getFirstDiagonalElement(firstElem);
    diagonalElement += firstElem;
  }
}


uint IKMLinearCombination::getNumParameters() const
{
  return parameterRanges[ parameterRanges.size() - 1 ] + matrices[ parameterRanges.size() - 1 ]->getNumParameters();
}
    
void IKMLinearCombination::getParameters(Vector & parameters) const
{
  uint ind = 0;
  parameters.resize ( getNumParameters() );
  if (verbose)
    cerr << "Number of total parameters: " << parameters.size() << endl;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++, ind++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    if (verbose)
      cerr << "Model " << ind << " has " << ikm->getNumParameters() << " parameters" << endl;
    if ( ikm->getNumParameters() == 0 ) continue;
    Vector singleParameterRef = parameters.getRangeRef( parameterRanges[ ind ], parameterRanges[ ind ] + ikm->getNumParameters() - 1 );
    ikm->getParameters ( singleParameterRef );
  }
}

void IKMLinearCombination::setParameters(const Vector & parameters)
{
  uint ind = 0;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++, ind++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    if ( ikm->getNumParameters() == 0 ) continue;
    ikm->setParameters ( parameters.getRange( parameterRanges[ ind ], parameterRanges[ ind ] + ikm->getNumParameters() - 1) );
  }
}

void IKMLinearCombination::setVerbose(const bool& _verbose)
{
  verbose = _verbose;
}

bool IKMLinearCombination::outOfBounds(const Vector & parameters) const
{
  uint ind = 0;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++, ind++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    if ( ikm->getNumParameters() == 0 ) continue;
    if ( ikm->outOfBounds( parameters.getRange( parameterRanges[ ind ], parameterRanges[ ind ] + ikm->getNumParameters() - 1) ) )
      return true;
  }
  return false;
}

Vector IKMLinearCombination::getParameterLowerBounds() const
{
  Vector lB;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    if ( ikm->getNumParameters() == 0 ) continue;
    lB.append( ikm->getParameterLowerBounds() );
  }
  return lB;
}

Vector IKMLinearCombination::getParameterUpperBounds() const
{
  Vector uB;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    if ( ikm->getNumParameters() == 0 ) continue;
    uB.append( ikm->getParameterUpperBounds() );
  }
  return uB;
}

void IKMLinearCombination::updateParameterRanges()
{
  if ( matrices.size() == 0 ) {
    parameterRanges.clear();
  } else {
    parameterRanges.resize(matrices.size());
    parameterRanges[0] = 0;

    if ( matrices.size() == 1 ) return;

    uint ind = 1;
    vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin();
    for ( ; ind < parameterRanges.size(); i++, ind++ )
    {
      ImplicitKernelMatrix *ikm = *i;
      if (verbose)
        cerr << "Parameter range: size is " << parameterRanges.size() << ", index is " << ind << endl;
      parameterRanges[ind] = parameterRanges[ind-1] + ikm->getNumParameters();
      if (verbose)
        cerr << "Value is " << parameterRanges[ind] << endl;
    }
  }
}
    
void IKMLinearCombination::addModel ( ImplicitKernelMatrix *ikm )
{
  matrices.push_back ( ikm );
  updateParameterRanges();
}

void IKMLinearCombination::multiply (NICE::Vector & y, const NICE::Vector & x) const
{
  y.resize( rows() );
  y.set(0.0);
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    Vector ySingle;
    ikm->multiply ( ySingle, x );
    y += ySingle;
  }
}

uint IKMLinearCombination::rows () const
{
  return cols();
}

uint IKMLinearCombination::cols () const
{
  if ( matrices.empty() )
    fthrow(Exception, "No models stored, cols() and rows() are unavailable");
  return (* matrices.begin())->cols();
}

double IKMLinearCombination::approxFrobNorm() const
{
  double frobNormApprox(0.0);
  if (verbose)
    std::cerr << "IKMLinCom: single approx: " ;
  for ( vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    frobNormApprox += ikm->approxFrobNorm();
    if (verbose)
      std::cerr << ikm->approxFrobNorm() << " ";
  }
  if (verbose)
    std::cerr << std::endl;
  return frobNormApprox;
}

void IKMLinearCombination::setApproximationScheme(const int & _approxScheme)
{
  for ( std::vector<ImplicitKernelMatrix *>::const_iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    ikm->setApproximationScheme(_approxScheme);
  }
}

ImplicitKernelMatrix * IKMLinearCombination::getModel(const uint & idx) const
{
  if ( idx <= matrices.size() )
    return matrices[idx];
  else
    return NULL;
}

// ----------------- INCREMENTAL LEARNING METHODS -----------------------
void IKMLinearCombination::addExample(const NICE::SparseVector & x, const NICE::Vector & binLabels)
{
  for ( vector<ImplicitKernelMatrix *>::iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    ImplicitKernelMatrix *ikm = *i;
    ikm->addExample(x, binLabels);
  }
}