/** 
* @file Quantization1DAequiDist0ToMax.h
* @brief Quantization of one-dimensional signals with selectable interval [0, vMax] (Interface)
* @author Alexander Freytag
* @date 13-10-2015 ( dd-mm-yyyy )
*/
#ifndef _NICE_QUANTIZATION1DAEQUIDIST0TOMAXINCLUDE
#define _NICE_QUANTIZATION1DAEQUIDIST0TOMAXINCLUDE

// NICE-core includes
#include <core/basics/types.h>
#include <core/basics/Persistent.h>

#include "gp-hik-core/quantization/Quantization.h"

namespace NICE {
  
 /** Quantization1DAequiDist0ToMax
 * @class Quantization
 * @brief Quantization of one-dimensional signals with selectable interval [0, vMax]
 * @author Alexander Freytag
 */
 
class Quantization1DAequiDist0ToMax  : public NICE::Quantization
{

  /** TODO
   * The current implementation only provides uniform quantization. We could extend this
   * by giving a ParameterizedFunction object to the constructor, which would allow us to inverse transform function values
   * before performing the binning.
   */

  protected:

  public:

  /** 
   * @brief default constructor
   * @author Alexander Freytag
   * @date 06-02-2014
   */
  
  Quantization1DAequiDist0ToMax( );
  
  /**
   * @brief simple constructor
   * @author Erik Rodner
   * @date 
   */
  Quantization1DAequiDist0ToMax( uint _numBins, 
                               NICE::Vector * v_upperBounds = NULL
                              );
    
  /** simple destructor */
  virtual ~Quantization1DAequiDist0ToMax();
  
  /**
  * @brief get specific word or prototype element of the quantization
  *
  * @param bin the index of the bin
  *
  * @return value of the prototype
  */
  virtual double getPrototype ( uint _bin, 
                                const uint & _dim = 0    
                              ) const;    
  
  /**
  * @brief Determine for a given signal value the bin in the vocabulary. This is not the corresponding prototype, which 
  * has to be requested with getPrototype afterwards
  *
  * @param value signal function value
  *
  * @return index of the bin entry corresponding to the given signal value
  */
  virtual uint quantize ( double _value, 
                          const uint & _dim = 0
                        ) const;
                        
                        
  virtual void computeParametersFromData ( const NICE::Vector & _maxValuesPerDimension );
  
  ///////////////////// INTERFACE PERSISTENT /////////////////////
  // interface specific methods for store and restore
  ///////////////////// INTERFACE PERSISTENT /////////////////////
  virtual void restore ( std::istream & _is, 
                         int _format = 0 
                       );
  virtual void store ( std::ostream & _os, 
                       int _format = 0 
                     ) const; 
  virtual void clear () {};    

 
};

}

#endif
