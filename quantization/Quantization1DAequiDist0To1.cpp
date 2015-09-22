/** 
* @file Quantization1DAequiDist0To1.cpp
* @brief Quantization1DAequiDist0To1 of one-dimensional signals with a standard range of [0,1] (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/09/2012

*/
#include <iostream>

#include "Quantization1DAequiDist0To1.h"

using namespace NICE;

Quantization1DAequiDist0To1::Quantization1DAequiDist0To1( ) 
{
  this->ui_numBins = 1;
}

Quantization1DAequiDist0To1::Quantization1DAequiDist0To1( 
                               uint _numBins, 
                               NICE::Vector * v_upperBounds
                             )
{
  this->ui_numBins = _numBins;
  //
  // this class does not require any upper bounds...
}

Quantization1DAequiDist0To1::~Quantization1DAequiDist0To1()
{
}

double Quantization1DAequiDist0To1::getPrototype ( uint _bin, 
                                                   const uint & _dim       
                                                 ) const
{
  //  _dim will be ignored for this type of quantization. all dimensions are treated equally...  
  return _bin / (double)(this->ui_numBins-1);
}
  
uint Quantization1DAequiDist0To1::quantize ( double _value,
                                             const uint & _dim
                                           ) const
{
  //  _dim will be ignored for this type of quantization. all dimensions are treated equally...
  
  if ( _value <= 0.0 ) 
    return 0;
  else if ( _value >= 1.0 ) 
    return this->ui_numBins-1;
  else 
    return static_cast<uint> ( _value * (this->ui_numBins-1) + 0.5 );
}

void Quantization1DAequiDist0To1::computeParametersFromData ( const NICE::FeatureMatrix *  _fm )
{
  // nothing to do here...
}
// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void Quantization1DAequiDist0To1::restore ( std::istream & _is, 
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
      
      if ( this->isEndTag( tmp, "Quantization1DAequiDist0To1" ) )
      {
        b_endOfBlock = true;
        continue;
      }                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("Quantization") == 0 )
      {
        // restore parent object
        Quantization::restore( _is );
      }       
      else
      {
        std::cerr << "WARNING -- unexpected Quantization1DAequiDist0To1 object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;  
      } 
    }
   }
  else
  {
    std::cerr << "Quantization1DAequiDist0To1::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void Quantization1DAequiDist0To1::store ( std::ostream & _os, 
                                          int _format 
                                        ) const
{
  // show starting point
  _os << this->createStartTag( "Quantization1DAequiDist0To1" ) << std::endl;
  
  // store parent object
  Quantization::store( _os ); 
    
  // done
  _os << this->createEndTag( "Quantization1DAequiDist0To1" ) << std::endl;
}
