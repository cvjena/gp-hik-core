#ifndef _NICE_ONLINELEARNABLEINCLUDE
#define _NICE_ONLINELEARNABLEINCLUDE


// NICE-core includes
#include <core/vector/SparseVectorT.h>
#include <core/vector/VectorT.h>

namespace NICE {


 /** 
 * @class OnlineLearnable
 * @brief Interface specifying learning algorithms implementing methods for online updates
 * @author Alexander Freytag
 * @date 01-01-2014 (dd-mm-yyyy)
 */ 
 
class OnlineLearnable {

 
  public:
    // Interface specifications
    virtual void addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement = true
			   ) = 0;
			   
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement = true
				    ) = 0;    


    // Provided functions and overloaded stream operators
    virtual ~OnlineLearnable () {};
    
    // just to prevent senseless compiler warnings
    OnlineLearnable() {};   

};


} // namespace

#endif
