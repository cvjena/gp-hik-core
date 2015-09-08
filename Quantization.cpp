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
  this->ui_numBins = 1;
}

Quantization::Quantization( uint _numBins )
{
  this->ui_numBins = _numBins;
}

Quantization::~Quantization()
{
}

uint Quantization::size() const
{
  return this->ui_numBins;
}
  
double Quantization::getPrototype (uint _bin) const
{
  return _bin / (double)(this->ui_numBins-1);
}
  
uint Quantization::quantize (double _value) const
{
  if ( _value <= 0.0 ) 
    return 0;
  else if ( _value >= 1.0 ) 
    return this->ui_numBins-1;
  else 
    return (uint)( _value * (this->ui_numBins-1) + 0.5 );
}

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void Quantization::restore ( std::istream & _is, 
                             int _format 
                           )
{
  if ( _is.good() )
  {    
    std::string tmp;    

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      _is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "Quantization" ) )
      {
        b_endOfBlock = true;
        continue;
      }                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("ui_numBins") == 0 )
      {
          _is >> this->ui_numBins;
      }
      else
      {
        std::cerr << "WARNING -- unexpected Quantization object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;  
      }
      
      _is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );      
    }
   }
  else
  {
    std::cerr << "Quantization::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void Quantization::store ( std::ostream & _os, 
                           int _format 
                         ) const
{
  // show starting point
  _os << this->createStartTag( "Quantization" ) << std::endl;
  
  _os << this->createStartTag( "ui_numBins" ) << std::endl;
  _os << this->ui_numBins << std::endl;
  _os << this->createEndTag( "ui_numBins" ) << std::endl;
    
  // done
  _os << this->createEndTag( "Quantization" ) << std::endl;
}