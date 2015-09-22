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
  public:
    typedef struct sparseVectorElement {
        uint example_index;
        double value;

        bool operator< (const sparseVectorElement & a) const
        {
            return value < a.value;
        }

    } sparseVectorElement;

  protected:

    sparseVectorElement **examples_raw;
    double **table_A;
    double **table_B;

    NICE::Vector diagonalElements;

    uint *nnz_per_dimension;
    uint num_dimension;
    uint num_examples;
    double d_noise;

    void initData ( const std::vector< const NICE::SparseVector *> & examples );
    void cleanupData ();
    double **allocateTable() const;
    void copyTable(double **src, double **dst) const;

  public:

    /** simple constructor */
    GMHIKernelRaw( const std::vector< const NICE::SparseVector *> & examples, const double d_noise = 0.1 );

    /** multiply with a vector: A*x = y; this is not really const anymore!! */
    virtual void multiply (NICE::Vector & y, const NICE::Vector & x) const;

    /** get the number of rows in A */
    virtual uint rows () const;

    /** get the number of columns in A */
    virtual uint cols () const;

    double **getTableA() const;
    double **getTableB() const;
    uint *getNNZPerDimension() const;

    /** simple destructor */
    virtual ~GMHIKernelRaw();

    sparseVectorElement **getDataMatrix() const { return examples_raw; };
    void updateTables ( const NICE::Vector _x ) const;

    /** get the diagonal elements of the current matrix */
    void getDiagonalElements ( NICE::Vector & _diagonalElements ) const;

};

}
#endif
