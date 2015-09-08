/** 
 * @file FastMinKernel.cpp
 * @brief Efficient GPs with HIK for classification by regression (Implementation)
 * @author Alexander Freytag
 * @date 06-12-2011 (dd-mm-yyyy)
*/

// STL includes
#include <iostream>

// NICE-core includes
#include <core/basics/vectorio.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "FastMinKernel.h"

using namespace std;
using namespace NICE;

/* protected methods*/


/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////


FastMinKernel::FastMinKernel()
{
  this->ui_d         = 0;
  this->ui_n         = 0;
  this->d_noise      = 1.0;
  this->approxScheme = MEDIAN;
  this->b_verbose    = false;
  this->setDebug(false);
}

FastMinKernel::FastMinKernel( const std::vector<std::vector<double> > & _X, 
                              const double _noise,
                              const bool _debug, 
                              const uint & _dim
                            )
{
  this->setDebug(_debug);
//   this->hik_prepare_kernel_multiplications ( _X, this->X_sorted, _dim);
  this->X_sorted.set_features( _X, _dim);
  this->ui_d         = this->X_sorted.get_d();
  this->ui_n         = this->X_sorted.get_n();
  this->d_noise      = _noise;
  this->approxScheme = MEDIAN;
  this->b_verbose    = false;
}
      
#ifdef NICE_USELIB_MATIO
FastMinKernel::FastMinKernel ( const sparse_t & _X, 
                               const double _noise, 
                               const std::map<uint, uint> & _examples,
                               const bool _debug, 
                               const uint & _dim
                             ) : this->X_sorted( _X, _examples, _dim )
{
  this->ui_d         = this->X_sorted.get_d();
  this->ui_n         = this->X_sorted.get_n();
  this->d_noise      = _noise;
  this->approxScheme = MEDIAN;
  this->b_verbose    = false;
  this->setDebug(_debug);
}
#endif

FastMinKernel::FastMinKernel ( const std::vector< const NICE::SparseVector * > & _X, 
                               const double _noise, 
                               const bool _debug, 
                               const bool & _dimensionsOverExamples, 
                               const uint & _dim)
{
  this->setDebug(_debug);
  this->X_sorted.set_features( _X, _dimensionsOverExamples, _dim);
//   this->hik_prepare_kernel_multiplications ( _X, this->X_sorted, _dimensionsOverExamples, _dim);
  this->ui_d         = this->X_sorted.get_d();
  this->ui_n         = this->X_sorted.get_n();
  this->d_noise      = _noise;
  this->approxScheme = MEDIAN;
  this->b_verbose    = false;
}

FastMinKernel::~FastMinKernel()
{
}


///////////////////// ///////////////////// /////////////////////
//                         GET / SET
//                   INCLUDING ACCESS OPERATORS
///////////////////// ///////////////////// //////////////////// 

uint FastMinKernel::get_n() const
{
  return this->ui_n;
}


uint FastMinKernel::get_d() const 
{
  return this->ui_d;
}

double FastMinKernel::getSparsityRatio()  const
{
  return this->X_sorted.computeSparsityRatio();
}

void FastMinKernel::setVerbose( const bool & _verbose)
{
  this->b_verbose = _verbose;
}

bool FastMinKernel::getVerbose( )   const
{
  return this->b_verbose;
}

void FastMinKernel::setDebug( const bool & _debug)
{
  this->b_debug = _debug;
  this->X_sorted.setDebug( _debug );
}

bool FastMinKernel::getDebug( )   const
{
  return this->b_debug;
}


      
///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

void FastMinKernel::applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *_pf)
{
  this->X_sorted.applyFunctionToFeatureMatrix( _pf );
}

void FastMinKernel::hik_prepare_kernel_multiplications(const std::vector<std::vector<double> > & _X, 
                                                       NICE::FeatureMatrixT<double> & _X_sorted, 
                                                       const uint & _dim
                                                      )
{
  //FIXME why do we hand over the feature matrix here?
  _X_sorted.set_features( _X, _dim);
}

void FastMinKernel::hik_prepare_kernel_multiplications(const std::vector< const NICE::SparseVector * > & _X, 
                                                       NICE::FeatureMatrixT<double> & _X_sorted, 
                                                       const bool & _dimensionsOverExamples, 
                                                       const uint & _dim
                                                      )
{
  //FIXME why do we hand over the feature matrix here?  
  _X_sorted.set_features( _X, _dimensionsOverExamples, _dim );
}

void FastMinKernel::hik_prepare_alpha_multiplications(const NICE::Vector & _alpha, 
                                                      NICE::VVector & _A, 
                                                      NICE::VVector & _B) const
{
//   std::cerr << "FastMinKernel::hik_prepare_alpha_multiplications" << std::endl;
//   std::cerr << "alpha: " << alpha << std::endl;
  _A.resize( this->ui_d );
  _B.resize( this->ui_d );

  //  efficient calculation of k*alpha
  //  ---------------------------------
  //  
  //  sum_i alpha_i k(x^i,x) = sum_i alpha_i sum_k min(x^i_k,x_k)
  //  = sum_k sum_i alpha_i min(x^i_k, x_k)
  //  
  //  now let us define l_k = { i | x^i_k <= x_k }
  //  and u_k = { i | x^i_k > x_k }, this leads to
  //  
  //  = sum_k ( sum_{l \in l_k} alpha_l x^i_k + sum_{u \in u_k} alpha_u x_k
  //  = sum_k ( sum_{l \in l_k} \alpha_l x^l_k + x_k * sum_{u \in u_k}
  //  alpha_u
  // 
  //  We also define 
  //  l^j_k = { i | x^i_j <= x^j_k } and
  //  u^j_k = { i | x^i_k > x^j_k }
  //
  //  We now need the partial sums 
  //
  //  (Definition 1)
  //  a_{k,j} = \sum_{l \in l^j_k} \alpha_l x^l_k 
  //
  //  and \sum_{u \in u^j_k} \alpha_u 
  //  according to increasing values of x^l_k
  //
  //  With 
  //  (Definition 2)
  //  b_{k,j} =  \sum_{l \in l^j_k} \alpha_l, 
  //
  //  we get
  //  \sum_{u \in u^j_k} \alpha_u  = \sum_{u=1}^n alpha_u - \sum_{l \in l^j_k} \alpha_l
  //  = b_{k,n} - b_{k,j}

  //  we only need as many entries as we have nonZero entries in our features for the corresponding dimensions
  for (uint i = 0; i < this->ui_d; i++)
  {
    uint numNonZero = this->X_sorted.getNumberOfNonZeroElementsPerDimension(i);
    //DEBUG
    //std::cerr << "number of non-zero elements in dimension " << i << " / " << d << ": " << numNonZero << std::endl;
    _A[i].resize( numNonZero );
    _B[i].resize( numNonZero  );
  }
  
  //  for more information see hik_prepare_alpha_multiplications
  
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    double alpha_sum(0.0);
    double alpha_times_x_sum(0.0);

    uint cntNonzeroFeat(0);
    
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    // loop through all elements in sorted order
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      
      // index of the feature
      int index = de.first;
      // transformed element of the feature
      //
      double elem( de.second );
                
      alpha_times_x_sum += _alpha[index] * elem;
      _A[dim][cntNonzeroFeat] = alpha_times_x_sum;
      
      alpha_sum += _alpha[index];
      _B[dim][cntNonzeroFeat] = alpha_sum;
      cntNonzeroFeat++;
    }
  }

}

double *FastMinKernel::hik_prepare_alpha_multiplications_fast(const NICE::VVector & _A, 
                                                              const NICE::VVector & _B,
                                                              const Quantization & _q,
                                                              const ParameterizedFunction *_pf 
                                                             ) const
{
  //NOTE keep in mind: for doing this, we already have precomputed A and B using hik_prepare_alpha_multiplications!
  
  // number of quantization bins
  uint hmax = _q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( _pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = _pf->f ( 1, _q.getPrototype(i) );
    } else {
      prototypes[i] = _q.getPrototype(i);
    }


  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->ui_d ];
//     std::cerr << "size of LUT: " << hmax * this->ui_d << std::endl;
//   sizeOfLUT = hmax * this->d;


  // loop through all dimensions
  for ( uint dim = 0; dim < this->ui_d; dim++ )
  {
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    uint index = 0;
    // we use the quantization of the original features! the transformed feature were
    // already used to calculate A and B, this of course assumes monotonic functions!!!
    uint qBin = _q.quantize ( i->first ); 

    // the next loop is linear in max(hmax, n)
    // REMARK: this could be changed to hmax*log(n), when
    // we use binary search
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n alpha_l
        t = fval*( _B[dim][this->ui_n-1 - nrZeroIndices] );
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->ui_n-1-nrZeroIndices)) )
        {
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = _q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if we all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= qBin) && ( index==(this->ui_n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // in fact, the term B[dim][this->n-1-nrZeroIndices] - B[dim][index] is equal to zero and vanishes, which is logical, since all elements are smaller than j!
          t = _A[dim][index];// + fval*( _B[dim][this->ui_n-1-nrZeroIndices] - _B[dim][index] );
        } else {
          // standard case
          t = _A[dim][index-1] + fval*( _B[dim][this->ui_n-1-nrZeroIndices] - _B[dim][index-1] );
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;
}

double *FastMinKernel::hikPrepareLookupTable(const NICE::Vector & _alpha, 
                                             const Quantization & _q, 
                                             const ParameterizedFunction *_pf 
                                            ) const
{
  // number of quantization bins
  uint hmax = _q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( _pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = _pf->f ( 1, _q.getPrototype(i) );
    } else {
      prototypes[i] = _q.getPrototype(i);
    }

  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->ui_d ];
//   sizeOfLUT = hmax * this->d;
  
  // loop through all dimensions
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    
    double alphaSumTotalInDim(0.0);
    double alphaTimesXSumTotalInDim(0.0);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      alphaSumTotalInDim += _alpha[i->second.first];
      alphaTimesXSumTotalInDim += _alpha[i->second.first] * i->second.second;
    }    
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    uint index = 0;
    
    // we use the quantization of the original features! Nevetheless, the resulting lookupTable is computed using the transformed ones
    uint qBin = _q.quantize ( i->first ); 
    
    double alpha_sum(0.0);
    double alpha_times_x_sum(0.0);
    double alpha_sum_prev(0.0);
    double alpha_times_x_sum_prev(0.0);
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n alpha_l
        //t = fval*( B[dim][this->n-1 - nrZeroIndices] );
        t = fval*alphaSumTotalInDim;
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->ui_n-1-nrZeroIndices)) )
        {
          alpha_times_x_sum_prev = alpha_times_x_sum;
          alpha_sum_prev = alpha_sum;
          alpha_times_x_sum += _alpha[i->second.first] * i->second.second; //i->dataElement.transformedFeatureValue
          alpha_sum += _alpha[i->second.first]; //i->dataElement.OrigIndex
          
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = _q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= qBin) && ( index==(this->ui_n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // in fact, the term B[dim][this->n-1-nrZeroIndices] - B[dim][index] is equal to zero and vanishes, which is logical, since all elements are smaller than j!
//           double lastTermAlphaTimesXSum;
//           double lastTermAlphaSum;
          t = alphaTimesXSumTotalInDim;
        } else {
          // standard case
          t = alpha_times_x_sum + fval*( alphaSumTotalInDim - alpha_sum );
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;
}


void FastMinKernel::hikUpdateLookupTable(double * _T, 
                                         const double & _alphaNew, 
                                         const double & _alphaOld, 
                                         const uint & _idx, 
                                         const Quantization & _q, 
                                         const ParameterizedFunction *_pf 
                                        ) const
{
  
  if (_T == NULL)
  {
    fthrow(Exception, "FastMinKernel::hikUpdateLookupTable LUT not initialized, run FastMinKernel::hikPrepareLookupTable first!");
    return;
  }
  
  // number of quantization bins
  uint hmax = _q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( _pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = _pf->f ( 1, _q.getPrototype(i) );
    } else {
      prototypes[i] = _q.getPrototype(i);
    }
  
  double diffOfAlpha(_alphaNew - _alphaOld);
  
  // loop through all dimensions
  for ( uint dim = 0; dim < this->ui_d; dim++ )
  {  
    double x_i ( (this->X_sorted( dim, _idx)) );
    
    //TODO we could also check wether x_i < tol, if we would store the tol explicitely
    if ( x_i == 0.0 ) //nothing to do in this dimension
      continue;

    //TODO we could speed up this by first doing a binary search for the position where the min changes, and then do two separate for-loops
    for (uint j = 0; j < hmax; j++)
    {
        double fval;
        uint q_bin = _q.quantize(x_i);
        
        if ( q_bin > j )
          fval = prototypes[j];
        else
          fval = x_i;      
        
      _T[ dim*hmax + j ] += diffOfAlpha*fval;
    }
  }

  delete [] prototypes;
}


void FastMinKernel::hik_kernel_multiply(const NICE::VVector & _A, 
                                        const NICE::VVector & _B, 
                                        const NICE::Vector & _alpha, 
                                        NICE::Vector & _beta
                                       ) const
{
  _beta.resize( this->ui_n );
  _beta.set(0.0);

  // runtime is O(n*d), we do no benefit from an additional lookup table here
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    // -- efficient sparse solution
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);

    if ( nrZeroIndices == this->ui_n ) {
      // all values are zero in this dimension :) and we can simply ignore the feature
      continue;
    }

    uint cnt(0);
    for ( multimap< double, SortedVectorSparse<double>::dataelement>::const_iterator i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, cnt++)
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      uint feat = de.first;
      uint inversePosition = cnt; 
      double fval = de.second;

      // in which position was the element sorted in? actually we only care about the nonzero elements, so we have to subtract the number of zero elements. 
      //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not

      //we definitly know that this element exists in inversePermutation, so we have not to check wether find returns .end() or not
      //int inversePosition(inversePermutation.find(feat)->second - nrZeroIndices);
      // sum_{l \in L_k} \alpha_l x^l_k
      //
      // A is zero for zero feature values (x^l_k is zero for all l \in L_k)
      double firstPart( _A[dim][inversePosition] );
      // sum_{u \in U_k} alpha_u
      // B is not zero for zero feature values, but we do not
      // have to care about them, because it is multiplied with
      // the feature value
      // DEBUG for Björns code
      if ( dim >= _B.size() )
        fthrow(Exception, "dim exceeds B.size: " << dim << " " << _B.size() );
      if ( _B[dim].size() == 0 )
        fthrow(Exception, "B[dim] is empty");
      if ( (this->ui_n-1-nrZeroIndices < 0)  || ((uint)(this->ui_n-1-nrZeroIndices) >= _B[dim].size() ) )
        fthrow(Exception, "n-1-nrZeroIndices is invalid: " << this->ui_n << " " << nrZeroIndices << " " << _B[dim].size() << " d: " << this->ui_d);
      if ( inversePosition < 0 || (uint)inversePosition >= _B[dim].size() )
        fthrow(Exception, "inverse position is invalid: " << inversePosition << " " << _B[dim].size() );
      double secondPart( _B[dim][this->ui_n-1-nrZeroIndices] - _B[dim][inversePosition]);

      _beta[feat] += firstPart + fval * secondPart; // i->elementpointer->dataElement->Value
    }
  }
  
  //FIXME
  //do we really want to considere noisy labels?
  for (uint feat = 0; feat < this->ui_n; feat++)
  {
    _beta[feat] += this->d_noise*_alpha[feat];
  }
}

void FastMinKernel::hik_kernel_multiply_fast(const double *_Tlookup, 
                                             const Quantization & _q, 
                                             const NICE::Vector & _alpha, 
                                             NICE::Vector & _beta) const
{
  _beta.resize( this->ui_n );
  _beta.set(0.0);

  // runtime is O(n*d), we do no benefit from an additional lookup table here
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    // -- efficient sparse solution
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();

    uint cnt(0);
    for ( multimap< double, SortedVectorSparse<double>::dataelement>::const_iterator i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, cnt++)
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      uint feat = de.first;
      uint qBin = _q.quantize(i->first);
      _beta[feat] += _Tlookup[dim*_q.size() + qBin];
    }
  }
  
  //do we really want to considere noisy labels?
  for (uint feat = 0; feat < this->ui_n; feat++)
  {
    _beta[feat] += this->d_noise*_alpha[feat];
  }
}

void FastMinKernel::hik_kernel_sum(const NICE::VVector & _A, 
                                   const NICE::VVector & _B, 
                                   const NICE::SparseVector & _xstar, 
                                   double & _beta, 
                                   const ParameterizedFunction *_pf) const
{
  // sparse version of hik_kernel_sum, no really significant changes,
  // we are just skipping zero elements
  _beta = 0.0;
  for (SparseVector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++)
  {
  
    uint dim = i->first;
    double fval = i->second;
        
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);    
    
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero and let us ignore it completely
      continue;
    }

    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    
    bool posIsZero ( position == 0 );
    
    if ( !posIsZero )
    {
      position--;
    }
    
    
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    //sum_{l \in L_k} \alpha_l x^l_k
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8
    if ( !posIsZero && ((position-nrZeroIndices) < this->ui_n) ) 
    {
      firstPart = (_A[dim][position-nrZeroIndices]);
    }      
    
    // sum_{u \in U_k} alpha_u
    
    // sum_{u \in U_k} alpha_u
    // => double secondPart( B(dim, n-1) - B(dim, position));
    //TODO in the next line there occurs the following error
    // Invalid read of size 8        
    double secondPart( _B[dim][this->ui_n-1-nrZeroIndices]);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if ( !posIsZero && (position >= nrZeroIndices) )
    {
      secondPart-= _B[dim][position-nrZeroIndices];
    }
    
    if ( _pf != NULL )
    {
      fval = _pf->f ( dim, fval );
    }   
      
    // but apply using the transformed one
    _beta += firstPart + secondPart* fval;
  }
}

void FastMinKernel::hik_kernel_sum(const NICE::VVector & _A, 
                                   const NICE::VVector & _B, 
                                   const NICE::Vector & _xstar, 
                                   double & _beta, 
                                   const ParameterizedFunction *_pf
                                  ) const
{
  _beta = 0.0;
  uint dim ( 0 );
  for (NICE::Vector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++, dim++)
  {
 
    double fval = *i;
    
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero and let us ignore it completely
      continue;
    }

    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    
    bool posIsZero ( position == 0 );
    
    if ( !posIsZero )
    {
      position--;
    }    
    
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    //sum_{l \in L_k} \alpha_l x^l_k
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8
    
    
    if ( !posIsZero && ((position-nrZeroIndices) < this->ui_n)  ) 
    {
      firstPart = (_A[dim][position-nrZeroIndices]);
    }
    
    // sum_{u \in U_k} alpha_u
    
    // sum_{u \in U_k} alpha_u
    // => double secondPart( B(dim, n-1) - B(dim, position));
    //TODO in the next line there occurs the following error
    // Invalid read of size 8      
    double secondPart( _B[dim][this->ui_n-1-nrZeroIndices] );
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    
    if ( !posIsZero && (position >= nrZeroIndices) )
    {
      secondPart-= _B[dim][position-nrZeroIndices];
    }
    
      
    
    if ( _pf != NULL )
    {
      fval = _pf->f ( dim, fval );
    }   
    
    // but apply using the transformed one
    _beta += firstPart + secondPart* fval;
  }
}

void FastMinKernel::hik_kernel_sum_fast(const double *_Tlookup, 
                                        const Quantization & _q, 
                                        const NICE::Vector & _xstar, 
                                        double & _beta
                                       ) const
{
  _beta = 0.0;
  if ( _xstar.size() != this->ui_d)
  {
    fthrow(Exception, "FastMinKernel::hik_kernel_sum_fast sizes of xstar and training data does not match!");
    return;
  }

  // runtime is O(d) if the quantizer is O(1)
  for ( uint dim = 0; dim < this->ui_d; dim++)
  {
    double v = _xstar[dim];
    uint qBin = _q.quantize(v);
    
    _beta += _Tlookup[dim*_q.size() + qBin];
  }
}

void FastMinKernel::hik_kernel_sum_fast(const double *_Tlookup, 
                                        const Quantization & _q, 
                                        const NICE::SparseVector & _xstar, 
                                        double & _beta
                                       ) const
{
  _beta = 0.0;
  // sparse version of hik_kernel_sum_fast, no really significant changes,
  // we are just skipping zero elements
  // for additional comments see the non-sparse version of hik_kernel_sum_fast
  // runtime is O(d) if the quantizer is O(1)
  for (SparseVector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++ )
  {
    uint dim = i->first;
    double v = i->second;
    uint qBin = _q.quantize(v);
    
    _beta += _Tlookup[dim*_q.size() + qBin];
  }
}

double *FastMinKernel::solveLin(const NICE::Vector & _y, 
                                NICE::Vector & _alpha,
                                const Quantization & _q, 
                                const ParameterizedFunction *_pf, 
                                const bool & _useRandomSubsets, 
                                uint _maxIterations, 
                                const uint & _sizeOfRandomSubset, 
                                double _minDelta, 
                                bool _timeAnalysis
                               ) const
{ 
  uint sizeOfRandomSubset(_sizeOfRandomSubset);

  bool verboseMinimal ( false );
  
  // number of quantization bins
  uint hmax = _q.size();
  
  NICE::Vector diagonalElements(_y.size(),0.0);
  this->X_sorted.hikDiagonalElements(diagonalElements);
  diagonalElements += this->d_noise;
  
  NICE::Vector pseudoResidual (_y.size(),0.0);
  NICE::Vector delta_alpha (_y.size(),0.0);
  double alpha_old;
  double alpha_new;
  double x_i;
  
  // initialization
  if (_alpha.size() != _y.size())
  {
    _alpha.resize( _y.size() );
  }
  _alpha.set(0.0);
  
  double *Tlookup = new double [ hmax * this->ui_d ];
  if ( (hmax*this->ui_d) <= 0 ) 
    return Tlookup;
  
  memset(Tlookup, 0, sizeof(Tlookup[0])*hmax*this->ui_d);
  
  uint iter;
  Timer t;
  if ( _timeAnalysis )
    t.start();
  
  if (_useRandomSubsets)
  {
    std::vector<uint> indices( _y.size() );
    for (uint i = 0; i < _y.size(); i++)
      indices[i] = i;
    
    if (sizeOfRandomSubset <= 0) 
      sizeOfRandomSubset = _y.size();
    
    for ( iter = 1; iter <= _maxIterations; iter++ ) 
    {
      NICE::Vector perm;
      this->randomPermutation( perm, indices, _sizeOfRandomSubset );
 
      if ( _timeAnalysis )
      {
        t.stop();
        Vector r;
        this->hik_kernel_multiply_fast(Tlookup, _q, _alpha, r);
        r = r - _y;
        
        double res = r.normL2();
        double resMax = r.normInf();

        std::cerr << "SimpleGradientDescent: TIME " << t.getSum() << " " << res << " " << resMax << std::endl;

        t.start();
      }
     
      for ( uint i = 0; i < sizeOfRandomSubset; i++)  
      {

        pseudoResidual(perm[i]) = -_y(perm[i]) + (this->d_noise * _alpha(perm[i]));
        for (uint j = 0; j < this->ui_d; j++)
        {
          x_i = this->X_sorted(j,perm[i]);
          pseudoResidual(perm[i]) += Tlookup[j*hmax + _q.quantize(x_i)];
        }
      
        //NOTE: this threshhold could also be a parameter of the function call
        if ( fabs(pseudoResidual(perm[i])) > 1e-7 )
        {
          alpha_old = _alpha(perm[i]);
          alpha_new = alpha_old - (pseudoResidual(perm[i])/diagonalElements(perm[i]));
          _alpha(perm[i]) = alpha_new;


          delta_alpha(perm[i]) = alpha_old-alpha_new;
         
          this->hikUpdateLookupTable(Tlookup, alpha_new, alpha_old, perm[i], _q, _pf ); // works correctly
          
        } else
        {
          delta_alpha(perm[i]) = 0.0;
        }
        
      }
      // after this only residual(i) is the valid residual... we should
      // really update the whole vector somehow
      
      double delta = delta_alpha.normL2();
      if ( this->b_verbose ) {
        cerr << "FastMinKernel::solveLin: iteration " << iter << " / " << _maxIterations << endl;     
        cerr << "FastMinKernel::solveLin: delta = " << delta << endl;
        cerr << "FastMinKernel::solveLin: pseudo residual = " << pseudoResidual.scalarProduct(pseudoResidual) << endl;
      }
      
      if ( delta < _minDelta ) 
      {
        if ( this->b_verbose )
          cerr << "FastMinKernel::solveLin: small delta" << endl;
        break;
      }    
    }
  }
  else //don't use random subsets
  {   
    for ( iter = 1; iter <= _maxIterations; iter++ ) 
    {
      
      for ( uint i = 0; i < _y.size(); i++ )
      {
          
        pseudoResidual(i) = -_y(i) + (this->d_noise* _alpha(i));
        for (uint j = 0; j < this->ui_d; j++)
        {
          x_i = this->X_sorted(j,i);
          pseudoResidual(i) += Tlookup[j*hmax + _q.quantize(x_i)];
        }
      
        //NOTE: this threshhold could also be a parameter of the function call
        if ( fabs(pseudoResidual(i)) > 1e-7 )
        {
          alpha_old = _alpha(i);
          alpha_new = alpha_old - (pseudoResidual(i)/diagonalElements(i));
          _alpha(i) = alpha_new;
          delta_alpha(i) = alpha_old-alpha_new;
          
          this->hikUpdateLookupTable(Tlookup, alpha_new, alpha_old, i, _q, _pf ); // works correctly
          
        } else
        {
          delta_alpha(i) = 0.0;
        }
        
      }
      
      double delta = delta_alpha.normL2();
      if ( this->b_verbose ) {
        std::cerr << "FastMinKernel::solveLin: iteration " << iter << " / " << _maxIterations << std::endl;     
        std::cerr << "FastMinKernel::solveLin: delta = " << delta << std::endl;
        std::cerr << "FastMinKernel::solveLin: pseudo residual = " << pseudoResidual.scalarProduct(pseudoResidual) << std::endl;
      }
      
      if ( delta < _minDelta ) 
      {
        if ( this->b_verbose )
          std::cerr << "FastMinKernel::solveLin: small delta" << std::endl;
        break;
      }    
    }
  }
  
  if (verboseMinimal)
    std::cerr << "FastMinKernel::solveLin -- needed " << iter << " iterations" << std::endl;
  return Tlookup;
}

void FastMinKernel::randomPermutation(NICE::Vector & _permutation, 
                                      const std::vector<uint> & _oldIndices, 
                                      const uint & _newSize
                                     ) const
{
  std::vector<uint> indices(_oldIndices);
  const uint oldSize = _oldIndices.size();
  uint resultingSize (std::min( oldSize, _newSize) );
  _permutation.resize(resultingSize);
  
  for ( uint i = 0; i < resultingSize; i++)
  {
    uint newIndex(rand() % indices.size());
    _permutation[i] = indices[newIndex ];
    indices.erase(indices.begin() + newIndex);
  }
}

double FastMinKernel::getFrobNormApprox()
{
  double frobNormApprox(0.0);
  
  switch (this->approxScheme)
  {
    case MEDIAN:
    {
      //\| K \|_F^1 ~ (n/2)^2 \left( \sum_k \median_k \right)^2
      //motivation: estimate half of the values in dim k to zero and half of them to the median (-> lower bound expectation)
      for ( uint i = 0; i < this->ui_d; i++ )
      {
        double median = this->X_sorted.getFeatureValues(i).getMedian();
        frobNormApprox += median;
      }
      
      frobNormApprox = fabs(frobNormApprox) * this->ui_n/2.0;
      break;
    }
    case EXPECTATION:
    {
      std::cerr << "EXPECTATION" << std::endl;
      //\| K \|_F^1^2 ~ \sum K_{ii}^2     +    (n^2 - n) \left( \frac{1}{3} \sum_k \left( 2 a_k + b_k \right) \right)
      // with a_k = minimal value in dim k and b_k maximal value
      
      //first term
      NICE::Vector diagEl;
      X_sorted.hikDiagonalElements(diagEl);
      frobNormApprox += diagEl.normL2();
      
      //second term
      double secondTerm(0.0);
      for ( uint i = 0; i < this->ui_d; i++ )
      {
        double minInDim;
        minInDim = this->X_sorted.getFeatureValues(i).getMin();
        double maxInDim;
        maxInDim = this->X_sorted.getFeatureValues(i).getMax();
        std::cerr << "min: " << minInDim << " max: " << maxInDim << std::endl;
        secondTerm += 2.0*minInDim + maxInDim;
      }
      secondTerm /= 3.0;
      secondTerm = pow(secondTerm, 2);
      secondTerm *= (this->ui_n * ( this->ui_n - 1 ));
      frobNormApprox += secondTerm;
      
      
      frobNormApprox = sqrt(frobNormApprox);
      
      break;
    }
    default:
    { //do nothing, approximate with zero :)
      break;
    }
  }
  return frobNormApprox;
}

void FastMinKernel::setApproximationScheme(const int & _approxScheme)
{
  switch(_approxScheme)
  {
    case 0:
    {
      this->approxScheme = MEDIAN;
      break;
    }
    case 1:
    {
      this->approxScheme = EXPECTATION;
      break;
    }
    default:
    {
      this->approxScheme = MEDIAN;
      break;
    }
  }
}

void FastMinKernel::hikPrepareKVNApproximation(NICE::VVector & _A) const
{
  _A.resize( this->ui_d );

  //  efficient calculation of |k_*|^2 = k_*^T * k_*
  //  ---------------------------------
  //  
  //    \sum_{i=1}^{n} \left( \sum_{d=1}^{D} \min (x_d^*, x_d^i) \right)^2
  //  <=\sum_{i=1}^{n} \sum_{d=1}^{D} \left( \min (x_d^*, x_d^i) \right)^2  
  //  = \sum_{d=1}^{D} \sum_{i=1}^{n} \left( \min (x_d^*, x_d^i) \right)^2
  //  = \sum_{d=1}^{D} \left( \sum_{i:x_d^i < x_*^d} (x_d^i)^2 + \sum_{j: x_d^* \leq x_d^j} (x_d^*)^2 \right)
  //
  //  again let us define l_d = { i | x_d^i <= x_d^* }
  //  and u_d = { i | x_d^i > x_d^* }, this leads to
  //  
  //  = \sum_{d=1}^{D} ( \sum_{l \in l_d} (x_d^l)^2 + \sum_{u \in u_d} (x_d^*)^2
  //  = \sum_{d=1}^{D} ( \sum_{l \in l_d} (x_d^l)^2 + (x_d^*)^2 \sum_{u \in u_d} 1
  // 
  //  We also define 
  //  l_d^j = { i | x_d^i <= x_d^j } and
  //  u_d^j = { i | x_d^i > x_d^j }
  //
  //  We now need the partial sums 
  //
  //  (Definition 1)
  //  a_{d,j} = \sum_{l \in l_d^j} (x_d^l)^2
  //  according to increasing values of x_d^l
  //
  //  We end at
  //  |k_*|^2 <= \sum_{d=1}^{D} \left( a_{d,r_d} + (x_d^*)^2 * |u_d^{r_d}| \right)
  //  with r_d being the index of the last example in the ordered sequence for dimension d, that is not larger than x_d^*

  //  we only need as many entries as we have nonZero entries in our features for the corresponding dimensions
  for ( uint i = 0; i < this->ui_d; i++ )
  {
    uint numNonZero = this->X_sorted.getNumberOfNonZeroElementsPerDimension(i);
    _A[i].resize( numNonZero );
  }
  //  for more information see hik_prepare_alpha_multiplications
  
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    double squared_sum(0.0);

    uint cntNonzeroFeat(0);
    
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    // loop through all elements in sorted order
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      
      // de: first - index, second - transformed feature
      double elem( de.second );
                
      squared_sum += pow( elem, 2 );
      _A[dim][cntNonzeroFeat] = squared_sum;

      cntNonzeroFeat++;
    }
  }
}

double * FastMinKernel::hikPrepareKVNApproximationFast(NICE::VVector & _A, 
                                                       const Quantization & _q, 
                                                       const ParameterizedFunction *_pf ) const
{
  //NOTE keep in mind: for doing this, we already have precomputed A using hikPrepareSquaredKernelVector!
  
  // number of quantization bins
  uint hmax = _q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( _pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = _pf->f ( 1, _q.getPrototype(i) );
    } else {
      prototypes[i] = _q.getPrototype(i);
    }


  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->ui_d ];

  // loop through all dimensions
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    uint index = 0;
    // we use the quantization of the original features! the transformed feature were
    // already used to calculate A and B, this of course assumes monotonic functions!!!
    uint qBin = _q.quantize ( i->first ); 

    // the next loop is linear in max(hmax, n)
    // REMARK: this could be changed to hmax*log(n), when
    // we use binary search
    //FIXME we should do this!
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n 1
        t = pow( fval, 2 ) * (this->ui_n-nrZeroIndices-index);
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->ui_n-nrZeroIndices)) )
        {
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = _q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= qBin) && ( index==(this->ui_n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // the second term vanishes, which is logical, since all elements are smaller than j!
          t = _A[dim][index];
        } else {
          // standard case
          t =  _A[dim][index-1] + pow( fval, 2 ) * (this->ui_n-nrZeroIndices-(index) );
//           A[dim][index-1] + fval * (n-nrZeroIndices-(index) );//fval*fval * (n-nrZeroIndices-(index-1) );
          
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;  
}

double* FastMinKernel::hikPrepareLookupTableForKVNApproximation(const Quantization & _q,
                                                                const ParameterizedFunction *_pf 
                                                               ) const
{
  // number of quantization bins
  uint hmax = _q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( _pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = _pf->f ( 1, _q.getPrototype(i) );
    } else {
      prototypes[i] = _q.getPrototype(i);
    }

  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->ui_d ];
  
  // loop through all dimensions
  for (uint dim = 0; dim < this->ui_d; dim++)
  {
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
         
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    uint index = 0;
    
    // we use the quantization of the original features! Nevetheless, the resulting lookupTable is computed using the transformed ones
    uint qBin = _q.quantize ( i->first ); 
    
    double sum(0.0);
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n 1
        t = pow( fval, 2 ) * (this->ui_n-nrZeroIndices-index);
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->ui_n-nrZeroIndices)) )
        {
          sum += pow( i->second.second, 2 ); //i->dataElement.transformedFeatureValue
          
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = _q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if we all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= qBin) && ( index==(this->ui_n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // the second term vanishes, which is logical, since all elements are smaller than j!
          t = sum;
        } else {
          // standard case
          t = sum + pow( fval, 2 ) * (this->ui_n-nrZeroIndices-(index) );
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;  
}

    //////////////////////////////////////////
    // variance computation: sparse inputs
    //////////////////////////////////////////    

void FastMinKernel::hikComputeKVNApproximation(const NICE::VVector & _A, 
                                               const NICE::SparseVector & _xstar, 
                                               double & _norm, 
                                               const ParameterizedFunction *_pf ) 
{
  _norm = 0.0;
  for (SparseVector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++)
  {
  
    uint dim = i->first;
    double fval = i->second;
    
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero so let us ignore them completely
      continue;
    }

    
    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    
    bool posIsZero ( position == 0 );
    
    if ( !posIsZero )
    {
      position--;
    }  
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if ( !posIsZero && ((position-nrZeroIndices) < this->ui_n) ) 
      firstPart = (_A[dim][position-nrZeroIndices]);
    
     
    if ( _pf != NULL )
      fval = _pf->f ( dim, fval );
    
    fval = fval * fval;
    
    double secondPart( 0.0);    
    
    if ( !posIsZero )
      secondPart = fval * (this->ui_n-nrZeroIndices-(position+1));
    else //if x_d^* is smaller than every non-zero training example
      secondPart = fval * (this->ui_n-nrZeroIndices);
    
    // but apply using the transformed one
    _norm += firstPart + secondPart;
  }  
}

void FastMinKernel::hikComputeKVNApproximationFast(const double *_Tlookup, 
                                                   const Quantization & _q, 
                                                   const NICE::SparseVector & _xstar, 
                                                   double & _norm
                                                  ) const
{
  _norm = 0.0;
  // runtime is O(d) if the quantizer is O(1)
  for (SparseVector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++ )
  {
    uint dim = i->first;
    double v = i->second;
    // we do not need a parameterized function here, since the quantizer works on the original feature values. 
    // nonetheless, the lookup table was created using the parameterized function    
    uint qBin = _q.quantize(v);
    
    _norm += _Tlookup[dim*_q.size() + qBin];
  }  
}

void FastMinKernel::hikComputeKernelVector ( const NICE::SparseVector& _xstar, 
                                             NICE::Vector & _kstar 
                                           ) const
{
  //init
  _kstar.resize( this->ui_n );
  _kstar.set(0.0);
  
  if ( this->b_debug )
  {
    std::cerr << " FastMinKernel::hikComputeKernelVector -- input: " << std::endl;
    _xstar.store( std::cerr);
  }
  
  //let's start :)
  for (SparseVector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++)
  {
  
    uint dim = i->first;
    double fval = i->second;
    
    if ( this->b_debug )
      std::cerr << "dim: " << dim  << " fval: " << fval << std::endl;
    
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero so let us ignore them completely
      continue;
    }
    

    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    //position--;
    
    if ( this->b_debug )
      std::cerr << " position: " << position << std::endl;
    
    //get the non-zero elements for this dimension  
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    
    //run over the non-zero elements and add the corresponding entries to our kernel vector

    uint count(nrZeroIndices);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, count++ )
    {
      uint origIndex(i->second.first); //orig index (i->second.second would be the transformed feature value)
      
      if ( this->b_debug )
        std::cerr << "i->1.2: " << i->second.first <<  " origIndex: " << origIndex << " count: " << count << " position: " << position << std::endl;
      if (count < position)
        _kstar[origIndex] += i->first; //orig feature value
      else
        _kstar[origIndex] += fval;
    }
    
  }  
}

    //////////////////////////////////////////
    // variance computation: non-sparse inputs
    //////////////////////////////////////////  

void FastMinKernel::hikComputeKVNApproximation(const NICE::VVector & _A, 
                                               const NICE::Vector & _xstar, 
                                               double & _norm,
                                               const ParameterizedFunction *_pf ) 
{
  _norm = 0.0;
  uint dim ( 0 );
  for (Vector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++, dim++)
  {
  
    double fval = *i;
    
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero so let us ignore them completely
      continue;
    }

    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    
    bool posIsZero ( position == 0 );
    
    if ( !posIsZero )
    {
      position--;
    } 
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if ( !posIsZero && ((position-nrZeroIndices) < this->ui_n) ) 
      firstPart = (_A[dim][position-nrZeroIndices]);
    
    double secondPart( 0.0);
      
    if ( _pf != NULL )
      fval = _pf->f ( dim, fval );
    
    fval = fval * fval;
    
    if ( !posIsZero ) 
      secondPart = fval * (this->ui_n-nrZeroIndices-(position+1));
    else //if x_d^* is smaller than every non-zero training example
      secondPart = fval * (this->ui_n-nrZeroIndices);
    
    // but apply using the transformed one
    _norm += firstPart + secondPart;
  }  
}

void FastMinKernel::hikComputeKVNApproximationFast(const double *_Tlookup, 
                                                   const Quantization & _q, 
                                                   const NICE::Vector & _xstar, 
                                                   double & _norm
                                                  ) const
{
  _norm = 0.0;
  // runtime is O(d) if the quantizer is O(1)
  uint dim ( 0 );
  for (Vector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++, dim++ )
  {
    double v = *i;
    // we do not need a parameterized function here, since the quantizer works on the original feature values. 
    // nonetheless, the lookup table was created using the parameterized function    
    uint qBin = _q.quantize(v);
    
    _norm += _Tlookup[dim*_q.size() + qBin];
  }  
}


void FastMinKernel::hikComputeKernelVector( const NICE::Vector & _xstar, 
                                            NICE::Vector & _kstar) const
{
  //init
  _kstar.resize(this->ui_n);
  _kstar.set(0.0);

 
  //let's start :)
  uint dim ( 0 );
  for (NICE::Vector::const_iterator i = _xstar.begin(); i != _xstar.end(); i++, dim++)
  {
  
    double fval = *i;
    
    uint nrZeroIndices = this->X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == this->ui_n ) {
      // all features are zero so let us ignore them completely
      continue;
    }
    

    uint position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    this->X_sorted.findFirstLargerInDimension(dim, fval, position);
    //position--;    
    
    
    //get the non-zero elements for this dimension  
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = this->X_sorted.getFeatureValues(dim).nonzeroElements();
    
    //run over the non-zero elements and add the corresponding entries to our kernel vector

    uint count(nrZeroIndices);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, count++ )
    {
      uint origIndex(i->second.first); //orig index (i->second.second would be the transformed feature value)
      if (count < position)
        _kstar[origIndex] += i->first; //orig feature value
      else
        _kstar[origIndex] += fval;
    }
  }  
}

///////////////////// INTERFACE PERSISTENT /////////////////////
// interface specific methods for store and restore
///////////////////// INTERFACE PERSISTENT ///////////////////// 

void FastMinKernel::restore ( std::istream & _is, 
                              int _format )
{
  bool b_restoreVerbose ( false );
  if ( _is.good() )
  {
    if ( b_restoreVerbose ) 
      std::cerr << " restore FastMinKernel" << std::endl;
    
    std::string tmp;
    _is >> tmp; //class name 
    
    if ( ! this->isStartTag( tmp, "FastMinKernel" ) )
    {
        std::cerr << " WARNING - attempt to restore FastMinKernel, but start flag " << tmp << " does not match! Aborting... " << std::endl;
        throw;
    }   
        
    _is.precision (numeric_limits<double>::digits10 + 1);
    
    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      _is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "FastMinKernel" ) )
      {
        b_endOfBlock = true;
        continue;
      }      
      
      tmp = this->removeStartTag ( tmp );
      
      if ( b_restoreVerbose )
        std::cerr << " currently restore section " << tmp << " in FastMinKernel" << std::endl;
      
      if ( tmp.compare("ui_n") == 0 )
      {
        _is >> this->ui_n;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("ui_d") == 0 )
      {
        _is >> this->ui_d;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      } 
      else if ( tmp.compare("d_noise") == 0 )
      {
        _is >> this->d_noise;
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("approxScheme") == 0 )
      {
        int approxSchemeInt;
        _is >> approxSchemeInt;
        setApproximationScheme(approxSchemeInt);
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );	
      }
      else if ( tmp.compare("X_sorted") == 0 )
      {
        this->X_sorted.restore(_is,_format);
        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }       
      else
      {
        std::cerr << "WARNING -- unexpected FastMinKernel object -- " << tmp << " -- for restoration... aborting" << std::endl;
        throw;
      }
    }
   }
  else
  {
    std::cerr << "FastMinKernel::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void FastMinKernel::store ( std::ostream & _os, 
                            int _format 
                          ) const
{
  if (_os.good())
  {    
    // show starting point
    _os << this->createStartTag( "FastMinKernel" ) << std::endl;    
    
    _os.precision (numeric_limits<double>::digits10 + 1);

    _os << this->createStartTag( "ui_n" ) << std::endl;
    _os << this->ui_n << std::endl;
    _os << this->createEndTag( "ui_n" ) << std::endl;
    
    
    _os << this->createStartTag( "ui_d" ) << std::endl;
    _os << this->ui_d << std::endl;
    _os << this->createEndTag( "ui_d" ) << std::endl;

    
    _os << this->createStartTag( "d_noise" ) << std::endl;
    _os << this->d_noise << std::endl;
    _os << this->createEndTag( "d_noise" ) << std::endl;

    
    _os << this->createStartTag( "approxScheme" ) << std::endl;
    _os << this->approxScheme << std::endl;
    _os << this->createEndTag( "approxScheme" ) << std::endl;
    
    _os << this->createStartTag( "X_sorted" ) << std::endl;
    //store the underlying data
    this->X_sorted.store(_os, _format);
    _os << this->createEndTag( "X_sorted" ) << std::endl;   
    
    
    // done
    _os << this->createEndTag( "FastMinKernel" ) << std::endl;        
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }    
}

void FastMinKernel::clear ()
{
  std::cerr << "FastMinKernel clear-function called" << std::endl;
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void FastMinKernel::addExample( const NICE::SparseVector * _example, 
                                const double & _label, 
                                const bool & _performOptimizationAfterIncrement
                                )
{
  // no parameterized function was given - use default 
  this->addExample ( _example );
}


void FastMinKernel::addMultipleExamples( const std::vector< const NICE::SparseVector * > & _newExamples,
                                         const NICE::Vector & _newLabels,
                                         const bool & _performOptimizationAfterIncrement
                                       )
{
  // no parameterized function was given - use default   
  this->addMultipleExamples ( _newExamples );
}

void FastMinKernel::addExample( const NICE::SparseVector * _example, 
                                const NICE::ParameterizedFunction *_pf
                              )
{ 
  this->X_sorted.add_feature( *_example, _pf );
  this->ui_n++;
}

void FastMinKernel::addMultipleExamples( const std::vector< const NICE::SparseVector * > & _newExamples,
                                         const NICE::ParameterizedFunction *_pf
                                         )
{
  for ( std::vector< const NICE::SparseVector * >::const_iterator exIt = _newExamples.begin();
        exIt != _newExamples.end();
        exIt++ )
  {
    this->X_sorted.add_feature( **exIt, _pf );
    this->ui_n++;     
  } 
}

