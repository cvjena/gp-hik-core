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


GMHIKernelRaw::GMHIKernelRaw( const std::vector< const NICE::SparseVector *> &_examples,
                              const double _d_noise
                              const NICE::Quantization * _q
                            )
{
    this->examples_raw = NULL;
    this->nnz_per_dimension = NULL;
    this->table_A = NULL;
    this->table_B = NULL;
    this->d_noise = _d_noise;
    this->q       = _q;

    initData(_examples);
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
    this->table_A = allocateTableAorB();
    this->table_B = allocateTableAorB();

    // Quantization for classification?
    if ( this->q != NULL )
    {
      // (1) if yes, setup the parameters of the quantization object
      this->q->computeParametersFromData ( this );
      this->table_T = allocateTableT();
    }
}

double **GMHIKernelRaw::allocateTableAorB() const
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

double **GMHIKernelRaw::allocateTableT() const
{
    double **table;
    table = new double *[this->num_dimension * this->q->getNumberOfBins()];
    return table;
}

void GMHIKernelRaw::copyTableAorB(double **src, double **dst) const
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

void GMHIKernelRaw::copyTableC(double **src, double **dst) const
{
    for (uint i = 0; i < this->num_dimension; i++)
    {
        for (uint j = 0; j < this->q->getNumberOfBins(); j++)
        {
            //FIXME can we speed this up using pointer increments?
            dst[i][j] = src[i][j];
        }
    }
}

void GMHIKernelRaw::updateTables ( const NICE::Vector _x ) const
{
    // pre-computions if quantization is activated
    double * prototypes;
    double * p_prototypes;

    // store prototypes
    if ( this->q != NULL)
    {
        // number of quantization bins
        uint hmax = _q->getNumberOfBins();


        double * prototypes   = new double [ hmax * this->ui_d ];
        double * p_prototypes = prototypes;

        for (uint dim = 0; dim < this->ui_d; dim++)
        {
          for ( uint i = 0 ; i < hmax ; i++ )
          {
            if ( _pf != NULL )
            {
              *p_prototypes = _pf->f ( dim, _q->getPrototype( i, dim ) );
            } else
            {
              *p_prototypes = _q->getPrototype( i, dim );
            }

            p_prototypes++;
          }
        }
    }

    // start the actual computations of A, B, and optionally T
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

        if ( this->q != NULL)
        {
//            // index of the element, which is always bigger than the current value fval
//            uint index = 0;
//            // we use the quantization of the original features! the transformed feature were
//            // already used to calculate A and B, this of course assumes monotonic functions!!!
//            uint qBin = _q->quantize ( i->first, dim );

//            // the next loop is linear in max(hmax, n)
//            // REMARK: this could be changed to hmax*log(n), when
//            // we use binary search

//            for (uint j = 0; j < hmax; j++)
//            {
//              double fval = prototypes[ dim*hmax + j ];
//              double t;

//              if (  (index == 0) && (j < qBin) ) {
//                // current element is smaller than everything else
//                // resulting value = fval * sum_l=1^n alpha_l
//                t = fval*( _B[dim][this->ui_n-1 - nrZeroIndices] );
//              } else {

//                 // move to next example, if necessary
//                while ( (j >= qBin) && ( index < (this->ui_n-1-nrZeroIndices)) )
//                {
//                  index++;
//                  iPredecessor = i;
//                  i++;

//                  if ( i->first !=  iPredecessor->first )
//                    qBin = _q->quantize ( i->first, dim );
//                }
//                // compute current element in the lookup table and keep in mind that
//                // index is the next element and not the previous one
//                //NOTE pay attention: this is only valid if all entries are positive! -
//                // If not, ask whether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
//                if ( (j >= qBin) && ( index==(this->ui_n-1-nrZeroIndices) ) ) {
//                  // the current element (fval) is equal or bigger to the element indexed by index
//                  // in fact, the term B[dim][this->n-1-nrZeroIndices] - B[dim][index] is equal to zero and vanishes, which is logical, since all elements are smaller than j!
//                  t = _A[dim][index];// + fval*( _B[dim][this->ui_n-1-nrZeroIndices] - _B[dim][index] );
//                } else {
//                  // standard case
//                  t = _A[dim][index-1] + fval*( _B[dim][this->ui_n-1-nrZeroIndices] - _B[dim][index-1] );
//                }
//              }

//              Tlookup[ dim*hmax + j ] = t;
//            }
        }
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
    double **t = allocateTableAorB();
    copyTableAorB(this->table_A, t);
    return t;
}

double **GMHIKernelRaw::getTableB() const
{
    double **t = allocateTableAorB();
    copyTableAorB(this->table_B, t);
    return t;
}

double **GMHIKernelRaw::getTableT() const
{
    double **t = allocateTableT();
    copyTableT(this->table_T, t);
    return t;
}

uint *GMHIKernelRaw::getNNZPerDimension() const
{
    uint *v = new uint[this->num_dimension];
    for (uint i = 0; i < this->num_dimension; i++)
        v[i] = this->nnz_per_dimension[i];
    return v;
}


uint NICE::GMHIKernelRaw::getNumberOfDimensions() const
{
    return this->num_dimension;
}

void NICE::GMHIKernelRaw::getDiagonalElements( NICE::Vector & _diagonalElements) const
{
    _diagonalElements = this->diagonalElements;
}

double NICE::GMHIKernelRaw::getLargestValue ( ) const
{
  double vmax (0.0);
  double vtmp (0.0);

  uint tmpIdx ( 0 );
  // compare largest elements of all dimensions
  for (uint d = 0; d < this->num_dimension; d++)
  {
      uint nnz = this->nnz_per_dimension[d];

      if ( nnz > 1 )
      {
          tmpIdx = tmpIdx + nnz;
          vtmp   = this->examples_raw[tmpIdx];
          if ( vtmp > vmax )
          {
            vmax = vtmp;
          }
      }
  }

  return vmax;
}


NICE::Vector NICE::GMHIKernelRaw::getLargestValuePerDimension ( ) const
{
  NICE::Vector vmax ( this->get_d() );

  NICE::Vector::iterator vmaxIt = vmax.begin();

  uint tmpIdx ( 0 );
  for (uint d = 0; d < this->num_dimension; d++, vmaxIt++)
  {
      uint nnz = this->nnz_per_dimension[d];

      if ( nnz > 1 )
      {
          tmpIdx  = tmpIdx + nnz;
          *vmaxIt = this->examples_raw[tmpIdx];
      }
      else
      {
          *vmaxIt = 0.0;
      }
  }

  return vmax;
}
