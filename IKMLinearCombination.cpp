/** 
* @file IKMLinearCombination.cpp
* @brief Combination of several (implicit) kernel matrices, such as noise matrix and gp-hik kernel matrix (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/14/2012
*/

// STL includes
#include <iostream>

// gp-hik-core includes
#include "gp-hik-core/IKMLinearCombination.h"

using namespace NICE;
using namespace std;


IKMLinearCombination::IKMLinearCombination()
{
  this->verbose = false;
}

IKMLinearCombination::~IKMLinearCombination()
{
  if ( this->matrices.size() != 0)
  {
    for (int i = 0; (uint)i < this->matrices.size(); i++)
      delete this->matrices[i];
  }
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// ///////////////////

void IKMLinearCombination::getDiagonalElements ( Vector & diagonalElements ) const
{
  diagonalElements.resize ( rows() );
  diagonalElements.set(0.0);
  
  for ( std::vector<NICE::ImplicitKernelMatrix *>::const_iterator i = this->matrices.begin(); i != this->matrices.end(); i++ )
  {
    NICE::ImplicitKernelMatrix *ikm = *i;
    NICE::Vector diagonalElementsSingle;
    ikm->getDiagonalElements ( diagonalElementsSingle );
    diagonalElements += diagonalElementsSingle;
  }
}

void IKMLinearCombination::getFirstDiagonalElement ( double & diagonalElement ) const
{
  diagonalElement = 0.0;
  for ( std::vector<NICE::ImplicitKernelMatrix *>::const_iterator i = this->matrices.begin(); i != this->matrices.end(); i++ )
  {
    NICE::ImplicitKernelMatrix *ikm = *i;
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

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void IKMLinearCombination::restore ( std::istream & is, int format )
{
  if (is.good())
  {
    is.precision (std::numeric_limits<double>::digits10 + 1); 
    
    std::string tmp;    

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "IKMLinearCombination" ) )
      {
        b_endOfBlock = true;
        continue;
      }                  
      
      tmp = this->removeStartTag ( tmp );
            
      is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );
    }
  }
}      

void IKMLinearCombination::store ( std::ostream & os, int format ) const
{
  if ( os.good() )
  {
    // show starting point
    os << this->createStartTag( "IKMLinearCombination" ) << std::endl;
      
    // done
    os << this->createEndTag( "IKMLinearCombination" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }  
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void IKMLinearCombination::addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement
			   )
{
  for ( std::vector<NICE::ImplicitKernelMatrix *>::iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    (*i)->addExample( example, label);
  }  
}

void IKMLinearCombination::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement
				    )
{
  for ( std::vector<NICE::ImplicitKernelMatrix *>::iterator i = matrices.begin(); i != matrices.end(); i++ )
  {
    (*i)->addMultipleExamples( newExamples, newLabels);
  }  
}