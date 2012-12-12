/** 
* @file PFWeightedDim.h
* @brief Parameterized Function: weights for each dimension (Interface + Implementation)
* @author Erik Rodner

*/
#ifndef _NICE_PFWEIGHTEDDIMINCLUDE
#define _NICE_PFWEIGHTEDDIMINCLUDE

#include <math.h>
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFWeightedDim
 * @brief Parameterized Function: weights for each dimension
 * @author Erik Rodner
 */
 
class PFWeightedDim : public ParameterizedFunction
{
  protected:

    double upperBound;
    double lowerBound;
    uint dimension;

  public:

  PFWeightedDim( uint dimension, 
            double lB = -std::numeric_limits<double>::max(), 
            double uB = std::numeric_limits<double>::max() ) : 
            ParameterizedFunction(dimension) 
  { 
    this->dimension = dimension;
    upperBound = uB;
    lowerBound = lB;
    if ( uB < 1.0 )
      m_parameters.set(uB);
    else
      m_parameters.set(1.0);
  };
  
  ~PFWeightedDim(){};
    
  double f ( uint index, double x ) const { return m_parameters[index] * m_parameters[index] * x; }

  bool isOrderPreserving() const { return true; };

  Vector getParameterUpperBounds() const { return Vector(m_parameters.size(), upperBound); };
  Vector getParameterLowerBounds() const { return Vector(m_parameters.size(), lowerBound); };
  
  void setParameterLowerBounds(const NICE::Vector & _newLowerBounds) { if (_newLowerBounds.size() > 0) lowerBound = _newLowerBounds(0);};
  void setParameterUpperBounds(const NICE::Vector & _newUpperBounds) { if (_newUpperBounds.size() > 0) upperBound = _newUpperBounds(0);};
  
  /** Persistent interface */
  virtual void restore ( std::istream & is, int format = 0 )
  {
    if (is.good())
    {
      is.precision (std::numeric_limits<double>::digits10 + 1);
      
      std::string tmp;
      is >> tmp;
      is >> upperBound;

      is >> tmp;
      is >> lowerBound;   
      
      is >> tmp;
      is >> dimension;
    }
    ParameterizedFunction::restore(is);
  };  
  virtual void store ( std::ostream & os, int format = 0 ) const
  {
    if (os.good())
    {
      os.precision (std::numeric_limits<double>::digits10 + 1); 
      os << "upperBound: " << std::endl <<  upperBound << std::endl;
      os << "lowerBound: " << std::endl <<  lowerBound << std::endl;
      os << "dimension: " << std::endl << dimension << std::endl;
    }
    ParameterizedFunction::store(os);
  };  
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "weightedDim";};
  
};

}

#endif
