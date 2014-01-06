/** 
* @file IKMNoise.cpp
* @author Erik Rodner, Alexander Freytag
* @brief Noise matrix (for model regularization) as an implicit kernel matrix (Implementation)
* @date 02/14/2012
*/

// STL includes
#include <iostream>
#include <limits>

// NICE-core includes
#include "IKMNoise.h"

using namespace NICE;
using namespace std;

IKMNoise::IKMNoise()
{
  this->size = 0;
  this->noise = 0.1;
  this->optimizeNoise = false;
  this->verbose = false;
}

IKMNoise::IKMNoise( uint size, double noise, bool optimizeNoise )
{
  this->size = size;
  this->noise = noise;
  this->optimizeNoise = optimizeNoise;
  this->verbose = false;
}


IKMNoise::~IKMNoise()
{
}


void IKMNoise::getDiagonalElements ( Vector & diagonalElements ) const
{
  diagonalElements.resize( size );
  diagonalElements.set( noise );
}

void IKMNoise::getFirstDiagonalElement ( double & diagonalElement ) const
{
  if (verbose)
  {    
    std::cerr << "IKMNoise::getFirstDiagonalElement  and labels.size() is zero" << std::endl;
  }
  diagonalElement = noise ;
}


uint IKMNoise::getNumParameters() const
{
  return optimizeNoise ? 1 : 0;
}
    
void IKMNoise::getParameters(Vector & parameters) const
{
  if ( optimizeNoise )
  {
    parameters.resize(1);
    parameters[0] = log(noise);
  }
}

void IKMNoise::setParameters(const Vector & parameters)
{
  if ( optimizeNoise )
  {
    noise = exp(parameters[0]);
  }
}

bool IKMNoise::outOfBounds(const Vector & parameters) const
{
  // we do not have any restrictions
  return false;
}

Vector IKMNoise::getParameterLowerBounds() const
{
  Vector lB;
  if ( optimizeNoise ) {
    lB.resize(1);
    lB[0] = -std::numeric_limits<double>::max();
  }
  return lB;
}

Vector IKMNoise::getParameterUpperBounds() const
{
  Vector uB;
  if ( optimizeNoise ) {
    uB.resize(1);
    uB[0] = -std::numeric_limits<double>::max();
  }
  return uB;
}

void IKMNoise::multiply (NICE::Vector & y, const NICE::Vector & x) const
{
  y.resize( rows() );
  
  y = noise * x;
}

uint IKMNoise::rows () const
{
  return cols();
}

uint IKMNoise::cols () const
{
  return size;
}

double IKMNoise::approxFrobNorm() const
{
  NICE::Vector diagEl;
  this->getDiagonalElements ( diagEl);
  return diagEl.normL2();
}

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void IKMNoise::restore ( std::istream & is, int format )
{
  if (is.good())
  {
    is.precision (std::numeric_limits<double>::digits10 + 1); 
    
    std::string tmp;    

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "IKMNoise" ) )
      {
        b_endOfBlock = true;
        continue;
      }
                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("size") == 0 )
      {
          is >> size;
      }
      else if ( tmp.compare("noise") == 0 )
      {
          is >> noise;
      }
      else if ( tmp.compare("optimizeNoise") == 0 )
      {
          is >> optimizeNoise;
      }
      else
      {
	std::cerr << "WARNING -- unexpected IKMNoise object -- " << tmp << " -- for restoration... aborting" << std::endl;
	throw;	
      }
      
      is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );      
    }
   }
  else
  {
    std::cerr << "IKMNoise::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void IKMNoise::store ( std::ostream & os, int format ) const
{
  // show starting point
  os << this->createStartTag( "IKMNoise" ) << std::endl;
  
  
  
  os << this->createStartTag( "size" ) << std::endl;
  os << size << std::endl;
  os << this->createEndTag( "size" ) << std::endl;
  
  os << this->createStartTag( "noise" ) << std::endl;
  os << noise << std::endl;
  os << this->createEndTag( "noise" ) << std::endl;
  
  os << this->createStartTag( "optimizeNoise" ) << std::endl;
  os << optimizeNoise << std::endl;
  os << this->createEndTag( "optimizeNoise" ) << std::endl; 
  
  // done
  os << this->createEndTag( "IKMNoise" ) << std::endl;
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void IKMNoise::addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement
			   )
{
 this->size++;
}

void IKMNoise::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement
				    )
{
  this->size += newExamples.size();
}