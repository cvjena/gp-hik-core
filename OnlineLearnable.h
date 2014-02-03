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
    
    /** 
     * @brief Interface method to add a single example to the current object
     * @author Alexander Freytag
     * @param newExample example to be added
     * @param newLabel corresponding class labels
     * @param performOptimizationAfterIncrement (optional) whether or not to run a hyper parameter optimization after adding new examples
     */    
    virtual void addExample( const NICE::SparseVector * newExample, 
                              const double & newLabel, 
                              const bool & performOptimizationAfterIncrement = true
                            ) = 0;

    /** 
     * @brief Interface method to add multiple example to the current object
     * @author Alexander Freytag
     * @param newExamples vector of example to be added
     * @param newLabels vector of corresponding class labels
     * @param performOptimizationAfterIncrement (optional) whether or not to run a hyper parameter optimization after adding new examples
     */                                       
                            
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
                                      const NICE::Vector & newLabels,
                                      const bool & performOptimizationAfterIncrement = true
                                    ) = 0;    


    /** 
     * @brief simple destructor
     * @author Alexander Freytag
     */                                       
    virtual ~OnlineLearnable () {};
    
    
    /** 
     * @brief simple destructor
     * @author Alexander Freytag
     */    
    // just to prevent senseless compiler warnings
    OnlineLearnable() {};   

};


} // namespace

#endif
