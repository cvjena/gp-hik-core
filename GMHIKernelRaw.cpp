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
                              const double _d_noise,
                              NICE::Quantization * _q
                            )
{
    this->examples_raw = NULL;
    this->nnz_per_dimension = NULL;
    this->table_A = NULL;
    this->table_B = NULL;
    this->table_T = NULL;
    this->d_noise = _d_noise;
    this->q       = _q;

    this->initData(_examples);
}

GMHIKernelRaw::~GMHIKernelRaw()
{
    this->cleanupData();
}

void GMHIKernelRaw::cleanupData()
{
    // data structure of examples
    if ( this->examples_raw != NULL )
    {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (examples_raw[d] != NULL)
                delete [] examples_raw[d];
        delete [] this->examples_raw;
        this->examples_raw = NULL;
    }

    // counter of non-zero examples in each dimension
    if ( this->nnz_per_dimension != NULL )
    {
        delete [] this->nnz_per_dimension;
        this->nnz_per_dimension = NULL;
    }

    // LUT A for classification without quantization
    if ( this->table_A != NULL )
    {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (table_A[d] != NULL)
                delete [] table_A[d];
        delete [] this->table_A;
        this->table_A = NULL;
    }

    // LUT B for classification without quantization
    if ( this->table_B != NULL )
    {
        for ( uint d = 0; d < this->num_dimension; d++ )
            if (table_B[d] != NULL)
                delete [] table_B[d];
        delete [] this->table_B;
        this->table_B = NULL;
    }

    // LUT T for classification with quantization
    if ( this->table_T != NULL )
    {
        delete [] this->table_T;
        this->table_T = NULL;
    }
}

void GMHIKernelRaw::initData ( const std::vector< const NICE::SparseVector *> &_examples )
{
    if (_examples.size() == 0 )
        fthrow(Exception, "No examples given for learning");

    cleanupData();

    this->num_dimension     = _examples[0]->getDim();
    this->examples_raw      = new sparseVectorElement *[num_dimension];
    this->nnz_per_dimension = new uint [num_dimension];
    this->num_examples      = _examples.size();

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
    uint i_dimNonZero;
    double value;
    double l1norm;

    // iterate over all provided training examples to process their data
    for ( std::vector< const NICE::SparseVector * >::const_iterator i = _examples.begin();
          i != _examples.end();
          i++, example_index++, itDiagEl++
        )
    {
        l1norm = 0.0;
        const NICE::SparseVector *x = *i;
        // loop over all non-zero dimensions, copy dimension and value into our data structure, and compute the L1 norm
        for ( NICE::SparseVector::const_iterator j = x->begin(); j != x->end(); j++ )
        {
            i_dimNonZero = j->first;
            value        = j->second;

            examples_raw_increment[i_dimNonZero]->value = value;
            examples_raw_increment[i_dimNonZero]->example_index = example_index;

            // move data pointer to the next element in the current dimension
            examples_raw_increment[i_dimNonZero]++;
            this->nnz_per_dimension[i_dimNonZero]++;

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
      NICE::Vector _maxValuesPerDimension = this->getLargestValuePerDimension();
      this->q->computeParametersFromData ( _maxValuesPerDimension );
      this->table_T = this->allocateTableT();
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

double *GMHIKernelRaw::allocateTableT() const
{
    double *table;
    table = new double [this->num_dimension * this->q->getNumberOfBins()];
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

void GMHIKernelRaw::copyTableT(double *_src, double *_dst) const
{
  double * p_src = _src;
  double * p_dst = _dst;
  for ( int i = 0; i < this->num_dimension * this->q->getNumberOfBins(); i++, p_src++, p_dst++ )
  {
    *p_dst = *p_src;
  }
}

void GMHIKernelRaw::updateTablesAandB ( const NICE::Vector _x ) const
{
    // start the actual computations of A, B, and optionally T
    for (uint dim = 0; dim < this->num_dimension; dim++)
    {
      double alpha_sum = 0.0;
      double alpha_times_x_sum = 0.0;
      uint nnz = nnz_per_dimension[dim];
      

      //////////
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

void GMHIKernelRaw::updateTableT ( const NICE::Vector _x ) const
{
    // sanity check
    if ( this->q == NULL)
    {
        return;
    }



    // number of quantization bins
    uint hmax = this->q->getNumberOfBins();


    double * prototypes;
    prototypes   = new double [ hmax * this->num_dimension ];

    double * p_prototypes;
    p_prototypes = prototypes;

    // compute all prototypes to compare against lateron
    for (uint dim = 0; dim < this->num_dimension; dim++)
    {
      for ( uint i = 0 ; i < hmax ; i++ )
      {
        *p_prototypes = this->q->getPrototype( i, dim );
         p_prototypes++;
      }
    }

    // start the actual computation of  T
    for (uint dim = 0; dim < this->num_dimension; dim++)
    {
      uint nnz = nnz_per_dimension[dim];

        uint idxProtoElem; // denotes the bin number in dim i of a quantized example, previously termed qBin

        sparseVectorElement * i            = examples_raw[dim];
        sparseVectorElement * iPredecessor = examples_raw[dim];

        // index of the element, which is always bigger than the current value fval
        int indexElem = 0;
        // element of the feature
        double elem = i->value;

        for (uint idxProto = 0; idxProto < hmax; idxProto++) // previously j
        {
          double fvalProto = prototypes[ dim*hmax + idxProto ];
          double t;


          idxProtoElem = this->q->quantize ( elem, dim );


          if (  (indexElem == 0) && (idxProto < idxProtoElem) )
          {
            // current prototype is smaller than everything else
            // resulting value = fval * sum_l=1^n alpha_l
            t = fvalProto*( this->table_B[ dim ][ nnz-1 ] );
          }
          else
          {
            //move to next example, which is smaller then the current prototype (if necessary)
            // pay attentation to not loop over the number of non-zero elements
               while ( (idxProto >= idxProtoElem) && ( indexElem < ( nnz - 1 ) ) ) //(this->ui_n-1-nrZeroIndices)) )
               {
                 indexElem++;
                 iPredecessor = i;
                 i++;

                 if ( i->value !=  iPredecessor->value )
                 {
                   idxProtoElem = this->q->quantize ( i->value, dim );
                 }
               }
               // compute current element in the lookup table and keep in mind that
               // indexElem is the next element and not the previous one


               if ( (idxProto >= idxProtoElem) && ( indexElem==( nnz-1 ) ) )
               {
                 // the current prototype is equal to or larger than the largest training example in this dimension
                 // -> the term B[ dim ][ nnz-1 ] - B[ dim ][ indexElem ] is equal to zero and vanishes, which is logical, since all elements are smaller than j!
                 t = table_A[ dim ][ indexElem ];
               }
               else
               {
                 // standard case
                 t = table_A[ dim ][ indexElem-1 ] + fvalProto*( table_B[ dim ][ nnz-1 ] - table_B[ dim ][ indexElem-1 ] );
               }

           }

           this->table_T[ dim*hmax + idxProto ] = t;
        }//for-loop over prototypes
    }//for-loop over dimensions


    // clean-up prototypes
    if ( this->q != NULL)
    {
      delete [] prototypes;
    }
}

/** multiply with a vector: A*x = y */
void GMHIKernelRaw::multiply (NICE::Vector & _y, const NICE::Vector & _x) const
{
  // STEP 1: initialize tables A and B
  this->updateTablesAandB(_x);

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

double * GMHIKernelRaw::getTableT() const
{
    double * T = this->allocateTableT();
    copyTableT(this->table_T, T);
    return T;
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


NICE::Vector NICE::GMHIKernelRaw::getLargestValuePerDimension ( ) const
{
  NICE::Vector vmax ( this->num_dimension );

  NICE::Vector::iterator vmaxIt = vmax.begin();

  for (uint d = 0; d < this->num_dimension; d++, vmaxIt++)
  {
      uint nnz = this->nnz_per_dimension[d];

      if ( nnz > 0 )
      {
          *vmaxIt = this->examples_raw[ d ][ nnz-1 ].value;
      }
      else
      {
          *vmaxIt = 0.0;
      }
  }

  return vmax;
}
