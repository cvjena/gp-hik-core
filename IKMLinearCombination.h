/** 
* @file IKMLinearCombination.h
* @brief Combination of several (implicit) kernel matrices, such as noise matrix and gp-hik kernel matrix (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 02/14/2012

*/
#ifndef _NICE_IKMLINEARCOMBINATIONINCLUDE
#define _NICE_IKMLINEARCOMBINATIONINCLUDE

// STL includes
#include <vector>

// gp-hik-core includes
#include "gp-hik-core/ImplicitKernelMatrix.h"
#include "gp-hik-core/OnlineLearnable.h"

namespace NICE {

 /** 
 * @class IKMLinearCombination
 * @brief Combination of several (implicit) kernel matrices, such as noise matrix and gp-hik kernel matrix
 * @author Erik Rodner, Alexander Freytag
 */

class IKMLinearCombination : public ImplicitKernelMatrix
{

  protected:
    std::vector< ImplicitKernelMatrix * > matrices;
    std::vector<int> parameterRanges;
    bool verbose;

    void updateParameterRanges();
  public:

    /** simple constructor */
    IKMLinearCombination();
      
    /** simple destructor */
    virtual ~IKMLinearCombination();

    virtual void getDiagonalElements ( Vector & diagonalElements ) const;
    virtual void getFirstDiagonalElement ( double & diagonalElement ) const;
    virtual uint getNumParameters() const;
    
    virtual void getParameters(Vector & parameters) const;
    virtual void setParameters(const Vector & parameters);
    virtual bool outOfBounds(const Vector & parameters) const;
    
    void setVerbose(const bool & _verbose);

    virtual Vector getParameterLowerBounds() const;
    virtual Vector getParameterUpperBounds() const;

    void addModel ( ImplicitKernelMatrix *ikm );
    
    /** multiply with a vector: A*x = y */
    virtual void multiply (NICE::Vector & y, const NICE::Vector & x) const;

    /** get the number of rows in A */
    virtual uint rows () const;

    /** get the number of columns in A */
    virtual uint cols () const;
    
    virtual double approxFrobNorm() const;
    
    virtual void setApproximationScheme(const int & _approxScheme);
    
    ImplicitKernelMatrix * getModel(const uint & idx) const;
    inline int getNumberOfModels(){return matrices.size();};
    
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    virtual void restore ( std::istream & is, int format = 0 ) ;
    virtual void store ( std::ostream & os, int format = 0 ) const;  
    virtual void clear () {};
    
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////    
    
    virtual void addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement = true
			   );
			   
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement = true
				    );      

};

}

#endif
