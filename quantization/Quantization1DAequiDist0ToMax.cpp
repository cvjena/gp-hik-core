/** 
* @file Quantization1DAequiDist0ToMax.cpp
* @brief Quantization1DAequiDist0ToMax of one-dimensional signals with selectable interval [0, vMax] (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 13-10-2015 ( dd-mm-yyyy )

*/
#include <iostream>

#include "Quantization1DAequiDist0ToMax.h"

using namespace NICE;

Quantization1DAequiDist0ToMax::Quantization1DAequiDist0ToMax( ) 
{
  this->ui_numBins = 1;
}

Quantization1DAequiDist0ToMax::Quantization1DAequiDist0ToMax( 
                               uint _numBins, 
                               NICE::Vector * _upperBounds
                             )
{
  this->ui_numBins = _numBins;
  this->v_upperBounds.resize ( 1 );
  if ( (_upperBounds != NULL)  && (_upperBounds->size() > 0) )
    this->v_upperBounds[0] = (*_upperBounds)[0];
  else
    this->v_upperBounds[0] = 1.0;
}

Quantization1DAequiDist0ToMax::~Quantization1DAequiDist0ToMax()
{
}

  
double Quantization1DAequiDist0ToMax::getPrototype ( uint _bin, 
                                                     const uint & _dim       
                                                   ) const
{
  //  _dim will be ignored for this type of quantization. all dimensions are treated equally...  
  return (this->v_upperBounds[0]*_bin) / (double)(this->ui_numBins-1);
}
  
uint Quantization1DAequiDist0ToMax::quantize ( double _value,
                                               const uint & _dim
                                             ) const
{
  //  _dim will be ignored for this type of quantization. all dimensions are treated equally...
  
  if ( _value <= 0.0 ) 
    return 0;
  else if ( _value >= this->v_upperBounds[0] ) 
    return this->ui_numBins-1;
  else 
    return (uint)( _value/this->v_upperBounds[0]  * (this->ui_numBins-1) + 0.5 );
}



void Quantization1DAequiDist0ToMax::computeParametersFromData ( const NICE::FeatureMatrix *  _fm )
{
      double vmax = ( _fm->getLargestValue( ) );       
      this->v_upperBounds.resize ( 1 );
      this->v_upperBounds ( 0 ) = vmax;
}
// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void Quantization1DAequiDist0ToMax::restore ( std::istream & _is, 
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
      
      if ( this->isEndTag( tmp, "Quantization1DAequiDist0ToMax" ) )
      {
        b_endOfBlock = true;
        continue;
      }                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("Quantization") == 0 )
      {
        // restore parent object
        Quantization::restore(  _is );
      }       
      else
      {
        std::cerr << "WARNING -- unexpected Quantization1DAequiDist0ToMax object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;  
      }
    }
   }
  else
  {
    std::cerr << "Quantization1DAequiDist0ToMax::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void Quantization1DAequiDist0ToMax::store ( std::ostream & _os, 
                                            int _format 
                                          ) const
{
  // show starting point
  _os << this->createStartTag( "Quantization1DAequiDist0ToMax" ) << std::endl;
  
  // store parent object
  Quantization::store( _os ); 
    
  // done
  _os << this->createEndTag( "Quantization1DAequiDist0ToMax" ) << std::endl;
}