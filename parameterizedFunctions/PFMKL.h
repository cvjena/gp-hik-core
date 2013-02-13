/** 
* @file PFMKL.h
* @brief Parameterized Function: weights for Multiple Kernel Learning approach (Interface + Implementation)
* @author Alexander Freytag

*/
#ifndef _NICE_PFMULTIPLEKERNELLEARNINGINCLUDE
#define _NICE_PFMULTIPLEKERNELLEARNINGINCLUDE

#include <math.h>
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFMKL
 * @brief Parameterized Function: weights for Multiple Kernel Learning approach
 * @author Alexander Freytag
 */
 
class PFMKL : public ParameterizedFunction
{
  protected:

    double upperBound;
    double lowerBound;
    std::set<int> steps;

  public:

  PFMKL(    const std::set<int> & _steps,
            double lB = -std::numeric_limits<double>::max(), 
            double uB = std::numeric_limits<double>::max() ) : 
            ParameterizedFunction(_steps.size()+1) 
  { 
    upperBound = uB;
    lowerBound = std::max( lB, 0.0 );
    if ( uB < 1.0 )
      m_parameters.set(uB);
    else
      m_parameters.set(1.0);
    steps = _steps;
  };
  
  ~PFMKL(){};
    
  double f ( uint index, double x ) const
  { 
    int dummyCnt ( 0 );
    for (std::set<int>::const_iterator it = steps.begin(); it != steps.end(); it++, dummyCnt++)
    {
      if ( index < *it)
        return x * m_parameters[dummyCnt];
    }
    //default value, should never be reached
    return 0.0;
  }

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
  
  virtual std::string sayYourName() const {return "MKL weighting";};
  
};

}

#endif
