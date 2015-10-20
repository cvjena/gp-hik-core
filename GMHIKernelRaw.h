/**
* @file GMHIKernelRaw.h
* @author Erik Rodner, Alexander Freytag
* @brief Fast multiplication with histogram intersection kernel matrices (Interface)

*/
#ifndef _NICE_GMHIKERNELRAWINCLUDE
#define _NICE_GMHIKERNELRAWINCLUDE

#include <vector>

#include <core/algebra/GenericMatrix.h>

#include "quantization/Quantization.h"

namespace NICE {

 /**
 * @class GMHIKernelRaw
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
    double *table_T;

    NICE::Vector diagonalElements;

    uint *nnz_per_dimension;
    uint num_dimension;
    uint num_examples;
    double d_noise;

    /** object performing feature quantization */
    NICE::Quantization *q;



    /////////////////////////
    /////////////////////////
    //  PROTECTED METHODS  //
    /////////////////////////
    /////////////////////////

    void initData ( const std::vector< const NICE::SparseVector *> & examples );
    void cleanupData ();

    double** allocateTableAorB() const;
    double* allocateTableT() const;

    void copyTableAorB(double **src, double **dst) const;
    void copyTableT(double *src, double *dst) const;


    double * computeTableT ( const NICE::Vector & _alpha
                           );

    /////////////////////////
    /////////////////////////
    //    PUBLIC METHODS   //
    /////////////////////////
    /////////////////////////

  public:

    /** simple constructor */
    GMHIKernelRaw( const std::vector< const NICE::SparseVector *> & examples,
                   const double d_noise = 0.1,
                   const NICE::Quantization * _q = NULL
                 );

    /** multiply with a vector: A*x = y; this is not really const anymore!! */
    virtual void multiply ( NICE::Vector & y,
                            const NICE::Vector & x
                          ) const;

    /** get the number of rows in A */
    virtual uint rows () const;

    /** get the number of columns in A */
    virtual uint cols () const;

    double **getTableA() const;
    double **getTableB() const;
    double *getTableT() const;

    uint *getNNZPerDimension() const;
    uint getNumberOfDimensions() const;

    /** simple destructor */
    virtual ~GMHIKernelRaw();

    sparseVectorElement **getDataMatrix() const { return examples_raw; };
    void updateTables ( const NICE::Vector _x ) const;

    /** get the diagonal elements of the current matrix */
    void getDiagonalElements ( NICE::Vector & _diagonalElements ) const;


    double getLargestValue ( ) const;

    NICE::Vector getLargestValuePerDimension ( ) const;

};

}
#endif
