/** 
* @file GMHIKernel.h
* @author Erik Rodner, Alexander Freytag
* @brief Fast multiplication with histogram intersection kernel matrices (Interface)
* @date 01/02/2012

*/
#ifndef _NICE_GMHIKERNELINCLUDE
#define _NICE_GMHIKERNELINCLUDE

#include <vector>

#include <core/algebra/GenericMatrix.h>

#include "ImplicitKernelMatrix.h"
#include "FeatureMatrixT.h"
#include "FastMinKernel.h"

namespace NICE {

 /** 
 * @class GMHIKernel
 * @brief Fast multiplication with histogram intersection kernel matrices
 * @author Erik Rodner, Alexander Freytag
 */

class GMHIKernel : public ImplicitKernelMatrix
{

  protected:

    FastMinKernel *fmk;
    const Quantization *q;
    ParameterizedFunction *pf;

    bool verbose;

    bool use_sparse_implementation;
    bool useOldPreparation;

  public:

    /** simple constructor */
    GMHIKernel( FastMinKernel *_fmk, ParameterizedFunction *_pf = NULL, const Quantization *_q = NULL );
      
    /** multiply with a vector: A*x = y */
    virtual void multiply (NICE::Vector & y, const NICE::Vector & x) const;

    /** get the number of rows in A */
    virtual uint rows () const;

    /** get the number of columns in A */
    virtual uint cols () const;

    /** simple destructor */
    virtual ~GMHIKernel();
   
    /** get the diagonal elements of the current matrix */
    virtual void getDiagonalElements ( Vector & diagonalElements ) const;
    virtual void getFirstDiagonalElement ( double & diagonalElement ) const;

    uint getNumParameters() const;
    void getParameters(Vector & parameters) const;
    void setParameters(const Vector & parameters);
    bool outOfBounds(const Vector & parameters) const;

    Vector getParameterLowerBounds() const;
    Vector getParameterUpperBounds() const;

    void setVerbose( const bool & _verbose);
    void setUseOldPreparation( const bool & _useOldPreparation);
    
    virtual double approxFrobNorm() const;
    virtual void setApproximationScheme(const int & _approxScheme);
    
    void setFastMinKernel(NICE::FastMinKernel * _fmk){fmk = _fmk;};
    
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    virtual void restore ( std::istream & is, int format = 0 ) {};//fmk->restore( is, format );};
    virtual void store ( std::ostream & os, int format = 0 ) const {};//fmk->store( os, format );};
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
