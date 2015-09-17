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
    initData(_examples);
    this->d_noise = _d_noise;
}

GMHIKernelRaw::~GMHIKernelRaw()
{
}

void GMHIKernelRaw::initData ( const std::vector< const NICE::SparseVector *> &_examples )
{
    if (_examples.size() == 0 )
        fthrow(Exception, "No examples given for learning");

    // TODO: clean up data if it exists

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

    uint example_index = 0;
    for (std::vector< const NICE::SparseVector * >::const_iterator i = _examples.begin();
            i != _examples.end(); i++, example_index++)
    {
        const NICE::SparseVector *x = *i;
        for ( NICE::SparseVector::const_iterator j = x->begin(); j != x->end(); j++ )
        {
            uint index = j->first;
            double value = j->second;
            examples_raw_increment[index]->value = value;
            examples_raw_increment[index]->example_index = example_index;
            // move to the next element
            examples_raw_increment[index]++;
            this->nnz_per_dimension[index]++;
        }
    }

    // sort along each dimension
    for (uint d = 0; d < this->num_dimension; d++)
    {
        uint nnz = this->nnz_per_dimension[d];
        if ( nnz > 1 )
            std::sort( this->examples_raw[d], this->examples_raw[d] + nnz );
    }

    // pre-allocate the A and B matrices
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

/** multiply with a vector: A*x = y */
void GMHIKernelRaw::multiply (NICE::Vector & _y, const NICE::Vector & _x) const
{
  // STEP 1: initialize tables A and B
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
      cntNonzeroFeat++;
    }
  }

  _y.resize( this->num_examples );
  _y.set(0.0);

  for (uint dim = 0; dim < this->num_dimension; dim++)
  {
    uint nnz = this->nnz_per_dimension[dim];

    if ( nnz == this->num_examples ) {
      // all values are zero in this dimension :) and we can simply ignore the feature
      continue;
    }

    sparseVectorElement *training_values_in_dim = examples_raw[dim];
    for ( uint cntNonzeroFeat = 0; cntNonzeroFeat < nnz; cntNonzeroFeat++, training_values_in_dim++ )
    {
      uint feat = training_values_in_dim->example_index;
      uint inversePosition = cntNonzeroFeat;
      double fval = training_values_in_dim->value;

      double firstPart( this->table_A[dim][inversePosition] );
      double secondPart( this->table_B[dim][this->num_examples-1-nnz] - this->table_B[dim][inversePosition]);

      _y[cntNonzeroFeat] += firstPart + fval * secondPart;
    }
  }

  for (uint feat = 0; feat < this->num_examples; feat++)
  {
    _y[feat] += this->d_noise * _x[feat];
  }


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


