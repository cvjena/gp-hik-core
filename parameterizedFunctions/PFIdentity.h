/** 
* @file PFIdentity.h
* @author Alexander Freytag
* @brief Parameterized Function: simple identity (Interface + Implementation)
* @date 01/09/2015
*/
#ifndef _NICE_PFIDENTITYINCLUDE
#define _NICE_PFIDENTITYINCLUDE

// STL includes
#include <math.h>

// NICE-core includes
#include <core/vector/VectorT.h>

// NICE-core includes
#include "ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class PFIdentity
 * @brief Parameterized Function: simple identity 
 * @author Alexander Freytag
 */
 
class PFIdentity : public ParameterizedFunction
{
  protected:

  public:

  /** simple constructor, we only have one parameter */
  PFIdentity( ) : 
            ParameterizedFunction(0) 
  { 
  };
  
  ~PFIdentity(){};
    
  double f ( uint _index, double _x ) const { 
    return _x; 
  }

  bool isOrderPreserving() const { return true; };

  Vector getParameterUpperBounds() const { return NICE::Vector(0); };
  Vector getParameterLowerBounds() const { return NICE::Vector(0); };
  
  void setParameterLowerBounds(const NICE::Vector & _newLowerBounds) { };
  void setParameterUpperBounds(const NICE::Vector & _newUpperBounds) { };
  
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
        
        if ( this->isEndTag( tmp, "PFIdentity" ) )
        {
          b_endOfBlock = true;
          continue;
        }
              
        
        tmp = this->removeStartTag ( tmp );
        
        if ( tmp.compare("ParameterizedFunction") == 0 )
        {
          // restore parent object
          ParameterizedFunction::restore(is);
        } 
        else
        {
          std::cerr << "WARNING -- unexpected PFIdentity object -- " << tmp << " -- for restoration... aborting" << std::endl;
          throw;  
        }      
      }
      

    }
    else
    {
      std::cerr << "PFIdentity::restore -- InStream not initialized - restoring not possible!" << std::endl;
    }   
  };
  
  virtual void store ( std::ostream & os, int format = 0 ) const
  {
    if (os.good())
    {
      // show starting point
      os << this->createStartTag( "PFIdentity" ) << std::endl;      
      
      os.precision (std::numeric_limits<double>::digits10 + 1); 
      
      // store parent object
      ParameterizedFunction::store(os);      
      
      // done
      os << this->createEndTag( "PFIdentity" ) << std::endl; 
    }

  };
  
  virtual void clear () {};
  
  virtual std::string sayYourName() const {return "identity";};
     
};

}

#endif
