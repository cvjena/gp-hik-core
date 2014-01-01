/** 
* @file PFAbsExp.h
* @author Erik Rodner, Alexander Freytag
* @brief Parameterized Function: absolute value and exponential operation -- pow(fabs(x), exponent) (Interface + Implementation)
* @date 01/04/2012
*/
#ifndef _NICE_PFABSEXPINCLUDE
#define _NICE_PFABSEXPINCLUDE

// STL includes
#include <math.h>

// NICE-core includes
#include <core/vector/VectorT.h>

// NICE-core includes
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFAbsExp
 * @brief Parameterized Function: absolute value and exponential operation -- pow(fabs(x), exponent)
 * @author Erik Rodner, Alexander Freytag
 */
 
class PFAbsExp : public ParameterizedFunction
{
  protected:
    double upperBound;
    double lowerBound;

  public:

  /** simple constructor, we only have one parameter */
  PFAbsExp( double exponent = 1.0, 
            double lB = -std::numeric_limits<double>::max(), 
            double uB = std::numeric_limits<double>::max() ) : 
            ParameterizedFunction(1) 
  { 
    m_parameters[0] = exponent; 
    upperBound = uB;
    lowerBound = lB;
  };
  
  ~PFAbsExp(){};
    
  double f ( uint index, double x ) const { 
    return pow(fabs(x),m_parameters[0]); 
  }

  bool isOrderPreserving() const { return true; };

  Vector getParameterUpperBounds() const { return NICE::Vector(1, upperBound); };
  Vector getParameterLowerBounds() const { return NICE::Vector(1, lowerBound); };
  
  void setParameterLowerBounds(const NICE::Vector & _newLowerBounds) { if (_newLowerBounds.size() > 0) lowerBound = _newLowerBounds(0);};
  void setParameterUpperBounds(const NICE::Vector & _newUpperBounds) { if (_newUpperBounds.size() > 0) upperBound = _newUpperBounds(0);};
  
  /** Persistent interface */
  virtual void restore ( std::istream & is, int format = 0 )
  {
    if (is.good())
    {
      is.precision (std::numeric_limits<double>::digits10 + 1); 
      
      std::string tmp;    

      bool b_endOfBlock ( false ) ;
      
      while ( !b_endOfBlock )
      {
	is >> tmp; // start of block 
	
	if ( this->isEndTag( tmp, "PFAbsExp" ) )
	{
	  b_endOfBlock = true;
	  continue;
	}
		    
	
	tmp = this->removeStartTag ( tmp );
	
	if ( tmp.compare("upperBound") == 0 )
	{
	  is >> upperBound;
	  is >> tmp; // end of block 
	  tmp = this->removeEndTag ( tmp );
	}
	else if ( tmp.compare("lowerBound") == 0 )
	{
	  is >> lowerBound;
	  is >> tmp; // end of block 
	  tmp = this->removeEndTag ( tmp );	    
	}
	else if ( tmp.compare("ParameterizedFunction") == 0 )
	{
	  // restore parent object
	  ParameterizedFunction::restore(is);
	}	
	else
	{
	  std::cerr << "WARNING -- unexpected PFAbsExp object -- " << tmp << " -- for restoration... aborting" << std::endl;
	  throw;	
	}      
      }
      

    }
    else
    {
      std::cerr << "PFAbsExp::restore -- InStream not initialized - restoring not possible!" << std::endl;
    }   
  };
  
  virtual void store ( std::ostream & os, int format = 0 ) const
  {
    if (os.good())
    {
      // show starting point
      os << this->createStartTag( "PFAbsExp" ) << std::endl;      
      
      os.precision (std::numeric_limits<double>::digits10 + 1); 

      os << this->createStartTag( "upperBound" ) << std::endl;
      os << upperBound << std::endl;
      os << this->createEndTag( "upperBound" ) << std::endl; 
      
      os << this->createStartTag( "lowerBound" ) << std::endl;
      os << lowerBound << std::endl;
      os << this->createEndTag( "lowerBound" ) << std::endl;
      
      // store parent object
      ParameterizedFunction::store(os);      
      
      // done
      os << this->createEndTag( "PFAbsExp" ) << std::endl; 
    }

  };
  
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "absexp";};
     
};

}

#endif
