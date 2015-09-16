/**
* @file GMHIKernelRaw.h
* @author Erik Rodner, Alexander Freytag
* @brief Fast multiplication with histogram intersection kernel matrices (Interface)

*/
#ifndef _NICE_GMHIKERNELRAWINCLUDE
#define _NICE_GMHIKERNELRAWINCLUDE

#include <vector>

#include <core/algebra/GenericMatrix.h>

namespace NICE {

 /**
 * @class GMHIKernel
 * @brief Fast multiplication with histogram intersection kernel matrices
 * @author Erik Rodner, Alexander Freytag
 */

class GMHIKernelRaw : public GenericMatrix
{

  protected:
    typedef struct sparseVectorElement {
        uint example_index;
        double value;

        bool operator< (const sparseVectorElement & a) const
        {
            return value < a.value;
        }

    } sparseVectorElement;

    sparseVectorElement **examples_raw;

    uint *nnz_per_dimension;
    uint num_dimension;
    uint num_examples;

    void initData ( const std::vector< const NICE::SparseVector *> &_examples );

  public:

    /** simple constructor */
    GMHIKernelRaw( const std::vector< const NICE::SparseVector *> &_examples );

    /** multiply with a vector: A*x = y */
    virtual void multiply (NICE::Vector & y, const NICE::Vector & x) const;

    /** get the number of rows in A */
    virtual uint rows () const;

    /** get the number of columns in A */
    virtual uint cols () const;

    /** simple destructor */
    virtual ~GMHIKernelRaw();
};

}
#endif
