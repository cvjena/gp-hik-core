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


GMHIKernelRaw::GMHIKernelRaw( const std::vector< const NICE::SparseVector *> &_examples )
{
    initData(_examples);

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
    for (uint d = 0; d < num_dimension; d++)
    {
        this->examples_raw[d] = new sparseVectorElement [ this->num_dimension ];
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
        std::sort( this->examples_raw[d], this->examples_raw[d] + this->nnz_per_dimension[d] );
    }
}

/** multiply with a vector: A*x = y */
void GMHIKernelRaw::multiply (NICE::Vector & y, const NICE::Vector & x) const
{
    /*
    NICE::VVector A;
    NICE::VVector B;
    // prepare to calculate sum_i x_i K(x,x_i)
    fmk->hik_prepare_alpha_multiplications(x, A, B);

    fmk->hik_kernel_multiply(A, B, x, y);
    */
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


