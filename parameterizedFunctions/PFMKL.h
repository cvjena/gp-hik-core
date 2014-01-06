/** 
* @file PFMKL.h
* @brief Parameterized Function: weights for Multiple Kernel Learning approach (Interface + Implementation)
* @author Alexander Freytag

*/
#ifndef _NICE_PFMULTIPLEKERNELLEARNINGINCLUDE
#define _NICE_PFMULTIPLEKERNELLEARNINGINCLUDE

// STL includes
#include <math.h>

// NICE-core includes
#include <core/vector/VectorT.h>

// NICE-core includes
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
	
	if ( this->isEndTag( tmp, "PFMKL" ) )
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
	else if ( tmp.compare("steps") == 0 )
	{
	    is >> tmp; // start of block 
	    
	    int numberOfSteps;
	    if ( ! this->isStartTag( tmp, "numberOfSteps" ) )
	    {
	      std::cerr << "Attempt to restore PFMKL, but found no information about numberOfSteps elements. Aborting..." << std::endl;
	      throw;
	    }
	    else
	    {
	      is >> numberOfSteps;
	      is >> tmp; // end of block 
	      tmp = this->removeEndTag ( tmp );     
	    }
	    
	    is >> tmp; // start of block 
	    
	    if ( ! this->isStartTag( tmp, "stepInfo" ) )
	    {
	      std::cerr << "Attempt to restore PFMKL, but found no stepInfo. Aborting..." << std::endl;
	      throw;
	    }
	    else
	    {
	      steps.clear();
	      
	      for ( int tmpCnt = 0; tmpCnt < numberOfSteps; tmpCnt++)
	      {
		int tmpStep;
		is >> tmpStep;
		steps.insert ( tmpStep ); 
	      }
	      
	      is >> tmp; // end of block 
	      tmp = this->removeEndTag ( tmp ); 	      
	    }
	    
	    is >> tmp; // end of block 
	    tmp = this->removeEndTag ( tmp );
	    
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
	  std::cerr << "WARNING -- unexpected PFMKL object -- " << tmp << " -- for restoration... aborting" << std::endl;
	  throw;	
	}
	
      
      }
      
      // restore parent object
      ParameterizedFunction::restore(is);
    }
    else
    {
      std::cerr << "PFMKL::restore -- InStream not initialized - restoring not possible!" << std::endl;
    }
  };  
  virtual void store ( std::ostream & os, int format = 0 ) const
  {
    if (os.good())
    {
      // show starting point
      os << this->createStartTag( "PFMKL" ) << std::endl;      
      
      os.precision (std::numeric_limits<double>::digits10 + 1); 

      os << this->createStartTag( "upperBound" ) << std::endl;
      os << upperBound << std::endl;
      os << this->createEndTag( "upperBound" ) << std::endl; 
      
      os << this->createStartTag( "lowerBound" ) << std::endl;
      os << lowerBound << std::endl;
      os << this->createEndTag( "lowerBound" ) << std::endl;
      
      os << this->createStartTag( "steps" ) << std::endl;
	os << this->createStartTag( "numberOfSteps" ) << std::endl;
	os << steps.size() << std::endl;
	os << this->createEndTag( "numberOfSteps" ) << std::endl;    
	
        os << this->createStartTag( "stepInfo" ) << std::endl;;
        for ( std::set<int>::const_iterator mySetIt = steps.begin(); mySetIt != steps.end(); mySetIt++)
	  os << *mySetIt << " ";
        os << std::endl;
	os << this->createEndTag( "stepInfo" ) << std::endl;
      os << this->createEndTag( "steps" ) << std::endl;      

      // store parent object
      ParameterizedFunction::store(os);       
      
      // done
      os << this->createEndTag( "PFMKL" ) << std::endl; 
    }
  };  
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "MKL weighting";};
  
};

}

#endif
