/** 
* @file Quantization.h
* @brief Quantization of one-dimensional signals with a standard range of [0,1] (Interface)
* @author Erik Rodner
* @date 01/09/2012
*/
#ifndef _NICE_QUANTIZATIONINCLUDE
#define _NICE_QUANTIZATIONINCLUDE

#include <core/basics/types.h>

namespace NICE {
  
 /** 
 * @class Quantization
 * @brief Quantization of one-dimensional signals with a standard range of [0,1]
 * @author Erik Rodner
 */
 
class Quantization
{

  /** TODO
   * The current implementation only provides uniform quantization. We could extend this
   * by giving a ParameterizedFunction object to the constructor, which would allow us to inverse transform function values
   * before performing the binning.
   */

  protected:

  uint numBins;

  public:

  /** simple constructor */
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
     
};

}

#endif
