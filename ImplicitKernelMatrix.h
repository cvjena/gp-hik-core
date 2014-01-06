/** 
* @file ImplicitKernelMatrix.h
* @author Erik Rodner, Alexander Freytag
* @brief An implicit kernel matrix, allowing for fast multiplication with arbitrary vectors (Interface)
* @date 02/14/2012

*/
#ifndef _NICE_IMPLICITKERNELMATRIXINCLUDE
#define _NICE_IMPLICITKERNELMATRIXINCLUDE

// STL includes
#include <iostream>

// NICE-core includes
#include <core/algebra/GenericMatrix.h>
// 
#include <core/basics/Persistent.h>

// gp-hik-core includes
#include "gp-hik-core/OnlineLearnable.h"

namespace NICE {
  
/** @class ImplicitKernelMatrix
 * @brief An implicit kernel matrix, allowing for fast multiplication with arbitrary vectors
 * @author Erik Rodner, Alexander Freytag
 * @date 02/14/2012
 */

class ImplicitKernelMatrix : public GenericMatrix, public NICE::Persistent, public NICE::OnlineLearnable
{

  protected:

  public:

    /** simple constructor */
    ImplicitKernelMatrix();
      
    /** simple destructor */
    virtual ~ImplicitKernelMatrix();

    //get set methods
    virtual void getDiagonalElements ( Vector & diagonalElements ) const = 0;
    virtual void getFirstDiagonalElement ( double & diagonalElement ) const = 0;

    virtual uint getNumParameters() const = 0;

    virtual void getParameters(Vector & parameters) const = 0;
    virtual void setParameters(const Vector & parameters) = 0;
    virtual bool outOfBounds(const Vector & parameters) const = 0;

    virtual Vector getParameterLowerBounds() const = 0;
    virtual Vector getParameterUpperBounds() const = 0;
    
    virtual double approxFrobNorm() const = 0;
    virtual void setApproximationScheme(const int & _approxScheme) = 0;
    
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    virtual void restore ( std::istream & is, int format = 0 ) = 0;
    virtual void store ( std::ostream & os, int format = 0 )  const = 0;
    virtual void clear () = 0;
    
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////    
    
    virtual void addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement = true
			   ) = 0;
			   
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement = true
				    ) = 0;      
    
    //high order methods
    virtual void  multiply (NICE::Vector &y, const NICE::Vector &x) const = 0;
};

}

#endif