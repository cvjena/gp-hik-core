/** 
* @file Quantization.h
* @brief Quantization of signals (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 01/09/2012
*/
#ifndef _NICE_QUANTIZATIONINCLUDE
#define _NICE_QUANTIZATIONINCLUDE

// NICE-core includes
#include <core/basics/types.h>
#include <core/basics/Persistent.h>
// 
#include <core/vector/VectorT.h>

// gp-hik-core includes
#include "gp-hik-core/FeatureMatrixT.h"

namespace NICE {
  
 /** 
 * @class Quantization
 * @brief Quantization of signals
 * @author Erik Rodner, Alexander Freytag
 */
 
class Quantization  : public NICE::Persistent
{

  /** TODO
   * The current implementation only provides uniform quantization. We could extend this
   * by giving a ParameterizedFunction object to the constructor, which would allow us to inverse transform function values
   * before performing the binning.
   */

  protected:

    uint ui_numBins;
    
    // NOTE: we do not need lower bounds, 
    // since our features are required to be non-negative 
    // (for applying the HIK as kernel function)
    NICE::Vector v_upperBounds;    
    

  public:

  /** 
   * @brief default constructor
   * @author Alexander Freytag
   * @date 06-02-2014
   */
  
  Quantization( );
  
  /**
   * @brief simple constructor
   * @author Erik Rodner
   * @date 
   */
  Quantization( uint _numBins, 
                NICE::Vector * v_upperBounds = NULL
              );

  /** simple destructor */
  virtual ~Quantization();

  /**
  * @brief get the size of the vocabulary, i.e. the number of bins
  */
  virtual uint getNumberOfBins() const;  

  /**
  * @brief get specific word or prototype element of the quantization
  *
  * @param bin the index of the bin
  *
  * @return value of the prototype
  */
  virtual double getPrototype ( uint _bin, 
                                const uint & _dim = 0    
                              ) const = 0;

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
                        ) const = 0;
                        
                        
                        
                        
  //FIXME should the argument _fm be templated?
  virtual void computeParametersFromData ( const NICE::FeatureMatrix *  _fm ) = 0;                        
  
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
