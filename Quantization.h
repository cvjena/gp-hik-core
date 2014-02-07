/** 
* @file Quantization.h
* @brief Quantization of one-dimensional signals with a standard range of [0,1] (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 01/09/2012
*/
#ifndef _NICE_QUANTIZATIONINCLUDE
#define _NICE_QUANTIZATIONINCLUDE

// NICE-core includes
#include <core/basics/types.h>
#include <core/basics/Persistent.h>

namespace NICE {
  
 /** 
 * @class Quantization
 * @brief Quantization of one-dimensional signals with a standard range of [0,1]
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

    uint numBins;

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
  Quantization( uint numBins );
    
  /** simple destructor */
  virtual ~Quantization();

  /**
  * @brief get the size of the vocabulary, i.e. the number of bins
  */
  virtual uint size() const;

  /**
  * @brief get specific word or prototype element of the quantization
  *
  * @param bin the index of the bin
  *
  * @return value of the prototype
  */
  virtual double getPrototype (uint bin) const;

  /**
  * @brief Determine for a given signal value the bin in the vocabulary. This is not the corresponding prototype, which 
  * has to be requested with getPrototype afterwards
  *
  * @param value signal function value
  *
  * @return index of the bin entry corresponding to the given signal value
  */
  virtual uint quantize (double value) const;
  
  ///////////////////// INTERFACE PERSISTENT /////////////////////
  // interface specific methods for store and restore
  ///////////////////// INTERFACE PERSISTENT /////////////////////
  virtual void restore ( std::istream & is, int format = 0 );
  virtual void store ( std::ostream & os, int format = 0 ) const; 
  virtual void clear () {};  
     
};

}

#endif
