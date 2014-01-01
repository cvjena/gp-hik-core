/** 
* @file PFWeightedDim.h
* @brief Parameterized Function: weights for each dimension (Interface + Implementation)
* @author Erik Rodner, Alexander Freytag

*/
#ifndef _NICE_PFWEIGHTEDDIMINCLUDE
#define _NICE_PFWEIGHTEDDIMINCLUDE

// STL includes
#include <math.h>

// NICE-core includes
#include <core/vector/VectorT.h>

// NICE-core includes
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFWeightedDim
 * @brief Parameterized Function: weights for each dimension
 * @author Erik Rodner, Alexander Freytag
 */
 
class PFWeightedDim : public ParameterizedFunction
{
  protected:

    double upperBound;
    double lowerBound;

  public:

  PFWeightedDim( uint dimension, 
            double lB = -std::numeric_limits<double>::max(), 
            double uB = std::numeric_limits<double>::max() ) : 
            ParameterizedFunction(dimension) 
  { 
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

  Vector getParameterUpperBounds() const { return NICE::Vector(m_parameters.size(), upperBound); };
  Vector getParameterLowerBounds() const { return NICE::Vector(m_parameters.size(), lowerBound); };
  
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
	
	if ( this->isEndTag( tmp, "PFWeightedDim" ) )
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
	  std::cerr << "WARNING -- unexpected PFWeightedDim object -- " << tmp << " -- for restoration... aborting" << std::endl;
	  throw;	
	}
      }
    }
    else
    {
      std::cerr << "PFWeightedDim::restore -- InStream not initialized - restoring not possible!" << std::endl;
    }
  };  
  virtual void store ( std::ostream & os, int format = 0 ) const
  {
    if (os.good())
    {
      // show starting point
      os << this->createStartTag( "PFWeightedDim" ) << std::endl;      
      
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
      os << this->createEndTag( "PFWeightedDim" ) << std::endl;
    }

  };  
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "weightedDim";};
  
};

}

#endif
