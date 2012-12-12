/** 
* @file PFExp.h
* @author Erik Rodner
* @brief Parameterized Function: exponential operation -- exp(fabs(x), exponent) (Interface + Implementation)
*/
#ifndef _NICE_PFEXPINCLUDE
#define _NICE_PFEXPINCLUDE

#include <math.h>
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFExp
 * @brief Parameterized Function: Parameterized Function: exponential operation -- exp(fabs(x), exponent)
 * @author Erik Rodner
 */
 
class PFExp : public ParameterizedFunction
{
  protected:
    double upperBound;
    double lowerBound;

  public:

  /** simple constructor, we only have one parameter */
  PFExp( double exponent = 1.0, 
            double lB = -std::numeric_limits<double>::max(), 
            double uB = std::numeric_limits<double>::max() ) : 
            ParameterizedFunction(1) 
  { 
    m_parameters[0] = exponent; 
    upperBound = uB;
    lowerBound = lB;
  };
  
  ~PFExp(){};

  double f ( uint index, double x ) const { return (exp(fabs(x) * m_parameters[0]) - 1.0) / (exp(m_parameters[0]) - 1.0); }

  bool isOrderPreserving() const { return true; };

  Vector getParameterUpperBounds() const { return Vector(1, upperBound); };
  Vector getParameterLowerBounds() const { return Vector(1, lowerBound); };
  
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
    }
    ParameterizedFunction::store(os);
  };
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "exp";};
  
};

}

#endif
