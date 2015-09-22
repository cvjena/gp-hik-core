/**
* @file GMHIKernelRaw.cpp
* @brief Fast multiplication with histogram intersection kernel matrices (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/02/2012

*/
#include <iostream>

#include <core/vector/VVector.h>
#include <core/basics/Timer.h>

#include "GMHIKernelRaw.h"

using namespace NICE;
using namespace std;


GMHIKernelRaw::GMHIKernelRaw( const std::vector< const NICE::SparseVector *> &_examples, const double _d_noise )
{
    this->examples_raw = NULL;
    this->nnz_per_dimension = NULL;
    this->table_A = NULL;
    this->table_B = NULL;

    initData(_examples);
    this->d_noise = _d_noise;
}

GMHIKernelRaw::~GMHIKernelRaw()
{
    cleanupData();
}

void GMHIKernelRaw::cleanupData()
{
    if ( this->examples_raw != NULL ) {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (examples_raw[d] != NULL)
                delete [] examples_raw[d];
        delete [] this->examples_raw;
        this->examples_raw = NULL;
    }
    if ( this->nnz_per_dimension != NULL ) {
        delete [] this->nnz_per_dimension;
        this->nnz_per_dimension = NULL;
    }
    if ( this->table_A != NULL ) {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (table_A[d] != NULL)
                delete [] table_A[d];
        delete [] this->table_A;
        this->table_A = NULL;
    }
    if ( this->table_B != NULL ) {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (table_B[d] != NULL)
                delete [] table_B[d];
        delete [] this->table_B;
        this->table_B = NULL;
    }

}

void GMHIKernelRaw::initData ( const std::vector< const NICE::SparseVector *> &_examples )
{
    if (_examples.size() == 0 )
        fthrow(Exception, "No examples given for learning");

    cleanupData();

    this->num_dimension = _examples[0]->getDim();
    this->examples_raw = new sparseVectorElement *[num_dimension];
    this->nnz_per_dimension = new uint [num_dimension];
    this->num_examples = _examples.size();

    // waste memory and allocate a non-sparse data block
    sparseVectorElement **examples_raw_increment = new sparseVectorElement *[num_dimension];
    for (uint d = 0; d < this->num_dimension; d++)
    {
        this->examples_raw[d] = new sparseVectorElement [ this->num_examples ];
        examples_raw_increment[d] = this->examples_raw[d];
        this->nnz_per_dimension[d] = 0;
    }

    // additionally allocate a Vector with as many entries as examples
    // this vector will contain the L1 norm values of all examples + noise
    // thereby, it represents the diagonal entries of our kernel matrix for
    // the special case of minimum kernel
    this->diagonalElements.resize ( this->num_examples );
    this->diagonalElements.set ( this->d_noise );


    uint example_index = 0;
    NICE::Vector::iterator itDiagEl = this->diagonalElements.begin();

    // minor pre-allocation
    uint index;
    double value;
    double l1norm;

    for ( std::vector< const NICE::SparseVector * >::const_iterator i = _examples.begin();
          i != _examples.end();
          i++, example_index++, itDiagEl++
        )
    {
        l1norm = 0.0;
        const NICE::SparseVector *x = *i;
        for ( NICE::SparseVector::const_iterator j = x->begin(); j != x->end(); j++ )
        {
            index = j->first;
            value = j->second;
            examples_raw_increment[index]->value = value;
            examples_raw_increment[index]->example_index = example_index;
            // move to the next element
            examples_raw_increment[index]++;
            this->nnz_per_dimension[index]++;

            l1norm = l1norm + value;
        }
        *itDiagEl = *itDiagEl + l1norm;
    }

    delete [] examples_raw_increment;

    // sort along each dimension
    for (uint d = 0; d < this->num_dimension; d++)
    {
        uint nnz = this->nnz_per_dimension[d];
        if ( nnz > 1 )
            std::sort( this->examples_raw[d], this->examples_raw[d] + nnz );
    }

    // pre-allocate the A and B matrices
    this->table_A = allocateTable();
    this->table_A = new double *[this->num_dimension];
    this->table_B = new double *[this->num_dimension];
    for (uint i = 0; i < this->num_dimension; i++)
    {
        uint nnz = this->nnz_per_dimension[i];
        if (nnz>0) {
            this->table_A[i] = new double [ nnz ];
            this->table_B[i] = new double [ nnz ];
        } else {
            this->table_A[i] = NULL;
            this->table_B[i] = NULL;
        }
    }
}

double **GMHIKernelRaw::allocateTable() const
{
    double **table;
    table = new double *[this->num_dimension];
    for (uint i = 0; i < this->num_dimension; i++)
    {
        uint nnz = this->nnz_per_dimension[i];
        if (nnz>0) {
            table[i] = new double [ nnz ];
        } else {
            table[i] = NULL;
        }
    }
    return table;
}

void GMHIKernelRaw::copyTable(double **src, double **dst) const
{
    for (uint i = 0; i < this->num_dimension; i++)
    {
        uint nnz = this->nnz_per_dimension[i];
        if (nnz>0) {
            for (uint j = 0; j < nnz; j++)
                dst[i][j] = src[i][j];
        } else {
            dst[i] = NULL;
        }
    }
}

void GMHIKernelRaw::updateTables ( const NICE::Vector _x ) const
{
    for (uint dim = 0; dim < this->num_dimension; dim++)
    {
      double alpha_sum = 0.0;
      double alpha_times_x_sum = 0.0;
      uint nnz = nnz_per_dimension[dim];

      // loop through all elements in sorted order
      sparseVectorElement *training_values_in_dim = examples_raw[dim];
      for ( uint cntNonzeroFeat = 0; cntNonzeroFeat < nnz; cntNonzeroFeat++, training_values_in_dim++ )
      {
        // index of the feature
        int index = training_values_in_dim->example_index;
        // element of the feature
        double elem = training_values_in_dim->value;

        alpha_times_x_sum += _x[index] * elem;
        this->table_A[dim][cntNonzeroFeat] = alpha_times_x_sum;

        alpha_sum += _x[index];
        this->table_B[dim][cntNonzeroFeat] = alpha_sum;
      }
    }

}

/** multiply with a vector: A*x = y */
void GMHIKernelRaw::multiply (NICE::Vector & _y, const NICE::Vector & _x) const
{
  // STEP 1: initialize tables A and B
  updateTables(_x);

  _y.resize( this->num_examples );
  _y.set(0.0);

  for (uint dim = 0; dim < this->num_dimension; dim++)
  {
    uint nnz = this->nnz_per_dimension[dim];
    uint nz  = this->num_examples - nnz;

    if ( nnz == 0 ) {
      // all values are zero in this dimension :) and we can simply ignore the feature
      continue;
    }

    sparseVectorElement *training_values_in_dim = examples_raw[dim];
    for ( uint cntNonzeroFeat = 0; cntNonzeroFeat < nnz; cntNonzeroFeat++, training_values_in_dim++ )
    {
      uint feat = training_values_in_dim->example_index;
      uint inversePosition = cntNonzeroFeat;
      double fval = training_values_in_dim->value;

      double firstPart = this->table_A[dim][inversePosition];
      double secondPart = this->table_B[dim][nnz-1] - this->table_B[dim][inversePosition];

      _y[feat] += firstPart + fval * secondPart;
    }
  }

  for (uint feat = 0; feat < this->num_examples; feat++)
    _y[feat] += this->d_noise * _x[feat];


}

/** get the number of rows in A */
uint GMHIKernelRaw::rows () const
{
  // return the number of examples
  return num_examples;
}

/** get the number of columns in A */
uint GMHIKernelRaw::cols () const
{
  // return the number of examples
  return num_examples;
}

double **GMHIKernelRaw::getTableA() const
{
    double **t = allocateTable();
    copyTable(this->table_A, t);
    return t;
}

double **GMHIKernelRaw::getTableB() const
{
    double **t = allocateTable();
    copyTable(this->table_B, t);
    return t;
}

uint *GMHIKernelRaw::getNNZPerDimension() const
{
    uint *v = new uint[this->num_dimension];
    for (uint i = 0; i < this->num_dimension; i++)
        v[i] = this->nnz_per_dimension[i];
    return v;
}


void NICE::GMHIKernelRaw::getDiagonalElements( NICE::Vector & _diagonalElements) const
{
    _diagonalElements = this->diagonalElements;
}
