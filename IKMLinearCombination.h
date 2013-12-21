/** 
* @file IKMLinearCombination.h
* @brief Combination of several (implicit) kernel matrices, such as noise matrix and gp-hik kernel matrix (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 02/14/2012

*/
#ifndef _NICE_IKMLINEARCOMBINATIONINCLUDE
#define _NICE_IKMLINEARCOMBINATIONINCLUDE

#include <vector>
#include "ImplicitKernelMatrix.h"

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
    
    /** Persistent interface */
    virtual void restore ( std::istream & is, int format = 0 ) ;
    virtual void store ( std::ostream & os, int format = 0 ) const;  
    virtual void clear () {};

};

}

#endif
