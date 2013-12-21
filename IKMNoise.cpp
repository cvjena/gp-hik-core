/** 
* @file IKMNoise.cpp
* @author Erik Rodner, Alexander Freytag
* @brief Noise matrix (for model regularization) as an implicit kernel matrix (Implementation)
* @date 02/14/2012

*/
#include <iostream>
#include <limits>

#include "IKMNoise.h"

using namespace NICE;
using namespace std;

IKMNoise::IKMNoise()
{
  this->size = 0;
  this->noise = 0.1;
  this->optimizeNoise = false;
  this->np = 0;
  this->nn = 0;
  this->verbose = false;
}

IKMNoise::IKMNoise( uint size, double noise, bool optimizeNoise )
{
  this->size = size;
  this->noise = noise;
  this->optimizeNoise = optimizeNoise;
  this->np = 0;
  this->nn = 0;
  this->verbose = false;
}

IKMNoise::IKMNoise( const Vector & labels, double noise, bool optimizeNoise )
{
  this->size = labels.size();
  this->noise = noise;
  this->optimizeNoise = optimizeNoise;
  this->labels = labels;
  this->np = 0;
  this->nn = 0;
  this->verbose = false;
  for ( uint i = 0 ; i < labels.size(); i++ )
    if ( labels[i] == 1 ) 
      this->np++;
    else
      this->nn++;
    
  if (verbose)
  {
    std::cerr << "IKMNoise np : " << np << " nn: " << nn << std::endl;
  }
}


IKMNoise::~IKMNoise()
{
}


void IKMNoise::getDiagonalElements ( Vector & diagonalElements ) const
{
  diagonalElements.resize( size );
  if ( labels.size() == 0 ) {
    diagonalElements.set( noise );
  } else {
    for ( uint i = 0 ; i < labels.size(); i++ )
      if ( labels[i] == 1 ) {
        diagonalElements[i] = 2*np*noise/size;
      } else {
        diagonalElements[i] = 2*nn*noise/size;
      }
  }
}

void IKMNoise::getFirstDiagonalElement ( double & diagonalElement ) const
{
  if ( labels.size() == 0 )
  {
    if (verbose)
    {    
      std::cerr << "IKMNoise::getFirstDiagonalElement  and labels.size() is zero" << std::endl;
    }
    diagonalElement = noise ;
  }
  else
  {
    if ( labels[0] == 1 )
    {
      if (verbose)
      {          
        std::cerr << "IKMNoise::getFirstDiagonalElement -- and first entry is +1" << std::endl;
      }
      diagonalElement = 2*np*noise/size;
    } 
    else
    {
      if (verbose)
      {                
        std::cerr << "IKMNoise::getFirstDiagonalElement -- and first entry is -1" << std::endl;
      }
      diagonalElement = 2*nn*noise/size;
    }
  }
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
  
  if ( labels.size() == 0 )
  {
    y = noise * x;
  } else {
    for ( uint i = 0 ; i < labels.size(); i++ )
      if ( labels[i] == 1 ) {
        y[i] = 2*np*noise/size * x[i];
      } else {
        y[i] = 2*nn*noise/size * x[i];
      }
  }
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
      else if ( tmp.compare("np") == 0 )
      {
          is >> np;
      }
      else if ( tmp.compare("nn") == 0 )
      {
          is >> nn;
      }
      else if ( tmp.compare("labels") == 0 )
      {
          is >> labels;
      }
      else{ }
      
      is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );
    }
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
  
  os << this->createStartTag( "np" ) << std::endl;
  os << np << std::endl;
  os << this->createEndTag( "np" ) << std::endl;
  
  os << this->createStartTag( "nn" ) << std::endl;
  os << nn << std::endl;
  os << this->createEndTag( "nn" ) << std::endl;
  
  os << this->createStartTag( "labels" ) << std::endl;
  os << labels << std::endl;
  os << this->createEndTag( "labels" ) << std::endl;   
  
  // done
  os << this->createEndTag( "IKMNoise" ) << std::endl;
}
