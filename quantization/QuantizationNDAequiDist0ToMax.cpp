/** 
* @file QuantizationNDAequiDist0ToMax.cpp
* @brief QuantizationNDAequiDist0ToMax of one-dimensional signals with selectable interval [0, vMax] (Implementation)
* @author Alexander Freytag
* @date 13-10-2015 ( dd-mm-yyyy )

*/
#include <iostream>

#include "QuantizationNDAequiDist0ToMax.h"

using namespace NICE;

QuantizationNDAequiDist0ToMax::QuantizationNDAequiDist0ToMax( ) 
{
  this->ui_numBins = 1;
}

QuantizationNDAequiDist0ToMax::QuantizationNDAequiDist0ToMax( 
                               uint _numBins, 
                               NICE::Vector * _upperBounds
                             )
{
  this->ui_numBins    = _numBins;
  if ( (_upperBounds != NULL)  && (_upperBounds->size() > 0) )
    this->v_upperBounds = (*_upperBounds);
  else
    this->v_upperBounds = NICE::Vector( 1 );  
}

QuantizationNDAequiDist0ToMax::~QuantizationNDAequiDist0ToMax()
{
}

  
double QuantizationNDAequiDist0ToMax::getPrototype ( uint _bin, 
                                                     const uint & _dim       
                                                   ) const
{
  return (this->v_upperBounds[_dim]*_bin) / (double)(this->ui_numBins-1);
}
  
uint QuantizationNDAequiDist0ToMax::quantize ( double _value,
                                               const uint & _dim
                                             ) const
{
  if ( _value <= 0.0 ) 
    return 0;
  else if ( _value >= this->v_upperBounds[_dim] ) 
    return this->ui_numBins-1;
  else 
      return static_cast<uint> ( floor( _value/this->v_upperBounds[_dim]  * (this->ui_numBins-1) + 0.5 ) );
}


void QuantizationNDAequiDist0ToMax::computeParametersFromData ( const NICE::Vector & _maxValuesPerDimension )
{
  this->v_upperBounds = _maxValuesPerDimension;
}


// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void QuantizationNDAequiDist0ToMax::restore ( std::istream & _is, 
                                              int _format 
                                            )
{
  if ( _is.good() )
  {    
    std::string tmp;
    _is >> tmp; //class name 
    
    if ( ! this->isStartTag( tmp, "Quantization1DAequiDist0ToMax" ) )
    {
        std::cerr << " WARNING - attempt to restore Quantization1DAequiDist0ToMax, but start flag " << tmp << " does not match! Aborting... " << std::endl;
        throw;
    } 

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      _is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "QuantizationNDAequiDist0ToMax" ) )
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
        std::cerr << "WARNING -- unexpected QuantizationNDAequiDist0ToMax object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;  
      }
      //FIXME also store and restore the upper bounds      
//       _is >> tmp; // end of block 
//       tmp = this->removeEndTag ( tmp );     
    }
   }
  else
  {
    std::cerr << "QuantizationNDAequiDist0ToMax::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void QuantizationNDAequiDist0ToMax::store ( std::ostream & _os, 
                                            int _format 
                                          ) const
{
  // show starting point
  _os << this->createStartTag( "QuantizationNDAequiDist0ToMax" ) << std::endl;
  
  // store parent object
  Quantization::store( _os ); 
    
  // done
  _os << this->createEndTag( "QuantizationNDAequiDist0ToMax" ) << std::endl;
}
