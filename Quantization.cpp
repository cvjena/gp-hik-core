/** 
* @file Quantization.cpp
* @brief Quantization of one-dimensional signals with a standard range of [0,1] (Implementation)
* @author Erik Rodner
* @date 01/09/2012

*/
#include <iostream>

#include "Quantization.h"

using namespace NICE;


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
