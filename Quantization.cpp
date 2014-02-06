/** 
* @file Quantization.cpp
* @brief Quantization of one-dimensional signals with a standard range of [0,1] (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/09/2012

*/
#include <iostream>

#include "Quantization.h"

using namespace NICE;

Quantization::Quantization( )
{
  this->numBins = 1;
}

Quantization::Quantization( uint numBins )
{
  this->numBins = numBins;
}

Quantization::~Quantization()
{
}

uint Quantization::size() const
{
  return numBins;
}
  
double Quantization::getPrototype (uint bin) const
{
  return bin / (double)(numBins-1);
}
  
uint Quantization::quantize (double value) const
{
  if ( value <= 0.0 ) return 0;
  else if ( value >= 1.0 ) return numBins-1;
  else return (uint)( value * (numBins-1) + 0.5 );
}

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void Quantization::restore ( std::istream & is, int format )
{
  if (is.good())
  {    
    std::string tmp;    

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "Quantization" ) )
      {
        b_endOfBlock = true;
        continue;
      }                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("numBins") == 0 )
      {
          is >> numBins;
      }
      else
      {
        std::cerr << "WARNING -- unexpected Quantization object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;  
      }
      
      is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );      
    }
   }
  else
  {
    std::cerr << "Quantization::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void Quantization::store ( std::ostream & os, int format ) const
{
  // show starting point
  os << this->createStartTag( "Quantization" ) << std::endl;
  
  os << this->createStartTag( "numBins" ) << std::endl;
  os << numBins << std::endl;
  os << this->createEndTag( "numBins" ) << std::endl;
    
  // done
  os << this->createEndTag( "Quantization" ) << std::endl;
}