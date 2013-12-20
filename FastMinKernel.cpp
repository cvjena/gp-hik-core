/** 
 * @file FastMinKernel.cpp
 * @brief Efficient GPs with HIK for classification by regression (Implementation)
 * @author Alexander Freytag
 * @date 06-12-2011 (dd-mm-yyyy)
*/
#include <iostream>
//#include "tools.h"

#include "core/basics/vectorio.h"
#include "core/basics/Timer.h"
#include "FastMinKernel.h"

using namespace std;
using namespace NICE;

/* protected methods*/


/* public methods*/


FastMinKernel::FastMinKernel()
{
  this->d = -1;
  this->n = -1;
  this->noise = 1.0;
  approxScheme = MEDIAN;
  verbose = false;
  this->setDebug(false);
}

FastMinKernel::FastMinKernel( const std::vector<std::vector<double> > & X, const double noise, const bool _debug, const int & _dim)
{
  this->setDebug(_debug);
  this->hik_prepare_kernel_multiplications ( X, this->X_sorted, _dim);
  this->d = X_sorted.get_d();
  this->n = X_sorted.get_n();
  this->noise = noise;
  approxScheme = MEDIAN;
  verbose = false;
}
      
#ifdef NICE_USELIB_MATIO
FastMinKernel::FastMinKernel ( const sparse_t & X, const double noise, const std::map<int, int> & examples, const bool _debug, const int & _dim) : X_sorted( X, examples, _dim )
{
  this->d = X_sorted.get_d();
  this->n = X_sorted.get_n();
  this->noise = noise;
  approxScheme = MEDIAN;
  verbose = false;
  this->setDebug(_debug);
}
#endif

FastMinKernel::FastMinKernel ( const vector< SparseVector * > & X, const double noise, const bool _debug, const bool & dimensionsOverExamples, const int & _dim)
{
  this->setDebug(_debug);
  this->hik_prepare_kernel_multiplications ( X, this->X_sorted, dimensionsOverExamples, _dim);
  this->d = X_sorted.get_d();
  this->n = X_sorted.get_n();
  this->noise = noise;
  approxScheme = MEDIAN;
  verbose = false;
}

FastMinKernel::~FastMinKernel()
{
}

void FastMinKernel::applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *pf)
{
  this->X_sorted.applyFunctionToFeatureMatrix(pf);
}

void FastMinKernel::hik_prepare_kernel_multiplications(const std::vector<std::vector<double> > & X, NICE::FeatureMatrixT<double> & X_sorted, const int & _dim)
{
  X_sorted.set_features(X, _dim);
}

void FastMinKernel::hik_prepare_kernel_multiplications(const std::vector< NICE::SparseVector * > & X, NICE::FeatureMatrixT<double> & X_sorted, const bool & dimensionsOverExamples, const int & _dim)
{
  X_sorted.set_features(X, dimensionsOverExamples, _dim);
}

void FastMinKernel::hik_prepare_alpha_multiplications(const NICE::Vector & alpha, NICE::VVector & A, NICE::VVector & B) const
{
//   std::cerr << "FastMinKernel::hik_prepare_alpha_multiplications" << std::endl;
//   std::cerr << "alpha: " << alpha << std::endl;
  A.resize(d);
  B.resize(d);

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
  for (int i = 0; i < d; i++)
  {
    uint numNonZero = X_sorted.getNumberOfNonZeroElementsPerDimension(i);
    //DEBUG
    //std::cerr << "number of non-zero elements in dimension " << i << " / " << d << ": " << numNonZero << std::endl;
    A[i].resize( numNonZero );
    B[i].resize( numNonZero  );
  }
  
  //  for more information see hik_prepare_alpha_multiplications
  
  for (int dim = 0; dim < d; dim++)
  {
    double alpha_sum(0.0);
    double alpha_times_x_sum(0.0);

    int cntNonzeroFeat(0);
    
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    // loop through all elements in sorted order
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      
      // index of the feature
      int index = de.first;
      // transformed element of the feature
      //
      double elem( de.second );
                
      alpha_times_x_sum += alpha[index] * elem;
      A[dim][cntNonzeroFeat] = alpha_times_x_sum;
      
      alpha_sum += alpha[index];
      B[dim][cntNonzeroFeat] = alpha_sum;
      cntNonzeroFeat++;
    }
  }

//   A.store(std::cerr);
//   B.store(std::cerr);
}

double *FastMinKernel::hik_prepare_alpha_multiplications_fast(const NICE::VVector & A, const NICE::VVector & B, const Quantization & q, const ParameterizedFunction *pf ) const
{
  //NOTE keep in mind: for doing this, we already have precomputed A and B using hik_prepare_alpha_multiplications!
  
  // number of quantization bins
  uint hmax = q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = pf->f ( 1, q.getPrototype(i) );
    } else {
      prototypes[i] = q.getPrototype(i);
    }


  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->d ];
//     std::cerr << "size of LUT: " << hmax * this->d << std::endl;
//   sizeOfLUT = hmax * this->d;


  // loop through all dimensions
  for (int dim = 0; dim < this->d; dim++)
  {
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    int index = 0;
    // we use the quantization of the original features! the transformed feature were
    // already used to calculate A and B, this of course assumes monotonic functions!!!
    int qBin = q.quantize ( i->first ); 

    // the next loop is linear in max(hmax, n)
    // REMARK: this could be changed to hmax*log(n), when
    // we use binary search
    
    for (int j = 0; j < (int)hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n alpha_l
        t = fval*( B[dim][this->n-1 - nrZeroIndices] );
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->n-1-nrZeroIndices)) )
        {
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if we all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= qBin) && ( index==(this->n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // in fact, the term B[dim][this->n-1-nrZeroIndices] - B[dim][index] is equal to zero and vanishes, which is logical, since all elements are smaller than j!
          t = A[dim][index];// + fval*( B[dim][this->n-1-nrZeroIndices] - B[dim][index] );
        } else {
          // standard case
          t = A[dim][index-1] + fval*( B[dim][this->n-1-nrZeroIndices] - B[dim][index-1] );
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;
}

double *FastMinKernel::hikPrepareLookupTable(const NICE::Vector & alpha, const Quantization & q, const ParameterizedFunction *pf ) const
{
  // number of quantization bins
  uint hmax = q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = pf->f ( 1, q.getPrototype(i) );
    } else {
      prototypes[i] = q.getPrototype(i);
    }

  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->d ];
//   sizeOfLUT = hmax * this->d;
  
  // loop through all dimensions
  for (int dim = 0; dim < this->d; dim++)
  {
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    
    double alphaSumTotalInDim(0.0);
    double alphaTimesXSumTotalInDim(0.0);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      alphaSumTotalInDim += alpha[i->second.first];
      alphaTimesXSumTotalInDim += alpha[i->second.first] * i->second.second;
    }    
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    int index = 0;
    
    // we use the quantization of the original features! Nevetheless, the resulting lookupTable is computed using the transformed ones
    int qBin = q.quantize ( i->first ); 
    
    double alpha_sum(0.0);
    double alpha_times_x_sum(0.0);
    double alpha_sum_prev(0.0);
    double alpha_times_x_sum_prev(0.0);
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < (uint)qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n alpha_l
        //t = fval*( B[dim][this->n-1 - nrZeroIndices] );
        t = fval*alphaSumTotalInDim;
      } else {

         // move to next example, if necessary   
        while ( (j >= (uint)qBin) && ( index < (this->n-1-nrZeroIndices)) )
        {
          alpha_times_x_sum_prev = alpha_times_x_sum;
          alpha_sum_prev = alpha_sum;
          alpha_times_x_sum += alpha[i->second.first] * i->second.second; //i->dataElement.transformedFeatureValue
          alpha_sum += alpha[i->second.first]; //i->dataElement.OrigIndex
          
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= (uint)qBin) && ( index==(this->n-1-nrZeroIndices) ) ) {
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


void FastMinKernel::hikUpdateLookupTable(double * T, const double & alphaNew, const double & alphaOld, const int & idx, const Quantization & q, const ParameterizedFunction *pf ) const
{
  
  if (T == NULL)
  {
    fthrow(Exception, "FastMinKernel::hikUpdateLookupTable LUT not initialized, run FastMinKernel::hikPrepareLookupTable first!");
    return;
  }
  
  // number of quantization bins
  uint hmax = q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = pf->f ( 1, q.getPrototype(i) );
    } else {
      prototypes[i] = q.getPrototype(i);
    }
  
  double diffOfAlpha(alphaNew - alphaOld);
  
  // loop through all dimensions
  for (int dim = 0; dim < this->d; dim++)
  {  
    double x_i ( (X_sorted(dim,idx)) );
    
    //TODO we could also check wether x_i < tol, if we would store the tol explicitely
    if (x_i == 0.0) //nothing to do in this dimension
      continue;

    //TODO we could speed up this with first do a binary search for the position where the min changes, and then do two separate for-loops
    for (uint j = 0; j < hmax; j++)
    {
        double fval;
        int q_bin = q.quantize(x_i);
        if (q_bin > j)
          fval = prototypes[j];
        else
          fval = x_i;      
        
//       double fval = std::min(prototypes[j],x_i);      
      T[ dim*hmax + j ] += diffOfAlpha*fval;
    }
  }

  delete [] prototypes;
}


void FastMinKernel::hik_kernel_multiply(const NICE::VVector & A, const NICE::VVector & B, const NICE::Vector & alpha, NICE::Vector & beta) const
{
  beta.resize(n);
  beta.set(0.0);

  // runtime is O(n*d), we do no benefit from an additional lookup table here
  for (int dim = 0; dim < d; dim++)
  {
    // -- efficient sparse solution
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);

    if ( nrZeroIndices == n ) {
      // all values are zero in this dimension :) and we can simply ignore the feature
      continue;
    }

    int cnt(0);
    for ( multimap< double, SortedVectorSparse<double>::dataelement>::const_iterator i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, cnt++)
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      uint feat = de.first;
      int inversePosition = cnt; 
      double fval = de.second;

      // in which position was the element sorted in? actually we only care about the nonzero elements, so we have to subtract the number of zero elements. 
      //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not

      //we definitly know that this element exists in inversePermutation, so we have not to check wether find returns .end() or not
      //int inversePosition(inversePermutation.find(feat)->second - nrZeroIndices);
      // sum_{l \in L_k} \alpha_l x^l_k
      //
      // A is zero for zero feature values (x^l_k is zero for all l \in L_k)
      double firstPart( A[dim][inversePosition] );
      // sum_{u \in U_k} alpha_u
      // B is not zero for zero feature values, but we do not
      // have to care about them, because it is multiplied with
      // the feature value
      // DEBUG for Björns code
      if ( (uint)dim >= B.size() )
        fthrow(Exception, "dim exceeds B.size: " << dim << " " << B.size() );
      if ( B[dim].size() == 0 )
        fthrow(Exception, "B[dim] is empty");
      if ( (n-1-nrZeroIndices < 0)  || ((uint)(n-1-nrZeroIndices) >= B[dim].size() ) )
        fthrow(Exception, "n-1-nrZeroIndices is invalid: " << n << " " << nrZeroIndices << " " << B[dim].size() << " d: " << d);
      if ( inversePosition < 0 || (uint)inversePosition >= B[dim].size() )
        fthrow(Exception, "inverse position is invalid: " << inversePosition << " " << B[dim].size() );
      double secondPart( B[dim][n-1-nrZeroIndices] - B[dim][inversePosition]);

      beta[feat] += firstPart + fval * secondPart; // i->elementpointer->dataElement->Value
    }
  }
  
  //do we really want to considere noisy labels?
  for (int feat = 0; feat < n; feat++)
  {
    beta[feat] += noise*alpha[feat];
  }
}

void FastMinKernel::hik_kernel_multiply_fast(const double *Tlookup, const Quantization & q, const NICE::Vector & alpha, NICE::Vector & beta) const
{
  beta.resize(n);
  beta.set(0.0);

  // runtime is O(n*d), we do no benefit from an additional lookup table here
  for (int dim = 0; dim < d; dim++)
  {
    // -- efficient sparse solution
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();

    int cnt(0);
    for ( multimap< double, SortedVectorSparse<double>::dataelement>::const_iterator i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, cnt++)
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      uint feat = de.first;
      uint qBin = q.quantize(i->first);
      beta[feat] += Tlookup[dim*q.size() + qBin];
    }
  }
  
  //do we really want to considere noisy labels?
  for (int feat = 0; feat < n; feat++)
  {
    beta[feat] += noise*alpha[feat];
  }
}

void FastMinKernel::hik_kernel_sum(const NICE::VVector & A, const NICE::VVector & B, const NICE::SparseVector & xstar, double & beta, const ParameterizedFunction *pf) const
{
  // sparse version of hik_kernel_sum, no really significant changes,
  // we are just skipping zero elements
  beta = 0.0;
  for (SparseVector::const_iterator i = xstar.begin(); i != xstar.end(); i++)
  {
  
    int dim = i->first;
    double fval = i->second;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero and let us ignore it completely
      continue;
    }

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    //sum_{l \in L_k} \alpha_l x^l_k
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8
    if (position >= 0) 
      firstPart = (A[dim][position-nrZeroIndices]);
    
    // sum_{u \in U_k} alpha_u
    
    // sum_{u \in U_k} alpha_u
    // => double secondPart( B(dim, n-1) - B(dim, position));
    //TODO in the next line there occurs the following error
    // Invalid read of size 8      
    double secondPart( B[dim][n-1-nrZeroIndices]);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if (position >= 0) 
      secondPart-= B[dim][position-nrZeroIndices];
    
    if ( pf != NULL )
    {
      fval = pf->f ( dim, fval );
    }   
    
    // but apply using the transformed one
    beta += firstPart + secondPart* fval;
  }
}

void FastMinKernel::hik_kernel_sum(const NICE::VVector & A, const NICE::VVector & B, const NICE::Vector & xstar, double & beta, const ParameterizedFunction *pf) const
{
  beta = 0.0;
  int dim ( 0 );
  for (NICE::Vector::const_iterator i = xstar.begin(); i != xstar.end(); i++, dim++)
  {
  
    double fval = *i;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero and let us ignore it completely
      continue;
    }

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    //sum_{l \in L_k} \alpha_l x^l_k
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8
    if (position >= 0) 
      firstPart = (A[dim][position-nrZeroIndices]);
    
    // sum_{u \in U_k} alpha_u
    
    // sum_{u \in U_k} alpha_u
    // => double secondPart( B(dim, n-1) - B(dim, position));
    //TODO in the next line there occurs the following error
    // Invalid read of size 8      
    double secondPart( B[dim][n-1-nrZeroIndices]);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if (position >= 0) 
      secondPart-= B[dim][position-nrZeroIndices];
    
    if ( pf != NULL )
    {
      fval = pf->f ( dim, fval );
    }   
    
    // but apply using the transformed one
    beta += firstPart + secondPart* fval;
  }
}

void FastMinKernel::hik_kernel_sum_fast(const double *Tlookup, const Quantization & q, const NICE::Vector & xstar, double & beta) const
{
  beta = 0.0;
  if ((int) xstar.size() != d)
  {
    fthrow(Exception, "FastMinKernel::hik_kernel_sum_fast sizes of xstar and training data does not match!");
    return;
  }

  // runtime is O(d) if the quantizer is O(1)
  for (int dim = 0; dim < d; dim++)
  {
    double v = xstar[dim];
    uint qBin = q.quantize(v);
    
    beta += Tlookup[dim*q.size() + qBin];
  }
}

void FastMinKernel::hik_kernel_sum_fast(const double *Tlookup, const Quantization & q, const NICE::SparseVector & xstar, double & beta) const
{
  beta = 0.0;
  // sparse version of hik_kernel_sum_fast, no really significant changes,
  // we are just skipping zero elements
  // for additional comments see the non-sparse version of hik_kernel_sum_fast
  // runtime is O(d) if the quantizer is O(1)
  for (SparseVector::const_iterator i = xstar.begin(); i != xstar.end(); i++ )
  {
    int dim = i->first;
    double v = i->second;
    uint qBin = q.quantize(v);
    
    beta += Tlookup[dim*q.size() + qBin];
  }
}

double *FastMinKernel::solveLin(const NICE::Vector & y, NICE::Vector & alpha, const Quantization & q, const ParameterizedFunction *pf, const bool & useRandomSubsets, uint maxIterations, const int & _sizeOfRandomSubset, double minDelta, bool timeAnalysis) const
{ 
  int sizeOfRandomSubset(_sizeOfRandomSubset);
  bool verbose ( false );
  bool verboseMinimal ( false );
  
  // number of quantization bins
  uint hmax = q.size();
  
  NICE::Vector diagonalElements(y.size(),0.0);
  X_sorted.hikDiagonalElements(diagonalElements);
  diagonalElements += this->noise;
  
  NICE::Vector pseudoResidual (y.size(),0.0);
  NICE::Vector delta_alpha (y.size(),0.0);
  double alpha_old;
  double alpha_new;
  double x_i;
  
  // initialization
  if (alpha.size() != y.size())
    alpha.resize(y.size());
  alpha.set(0.0);
  
  double *Tlookup = new double [ hmax * this->d ];
  if ( (hmax*this->d) <= 0 ) return Tlookup;
  memset(Tlookup, 0, sizeof(Tlookup[0])*hmax*this->d);
  
  uint iter;
  Timer t;
  if ( timeAnalysis )
    t.start();
  
  if (useRandomSubsets)
  {
    std::vector<int> indices(y.size());
    for (uint i = 0; i < y.size(); i++)
      indices[i] = i;
    
    if (sizeOfRandomSubset <= 0) 
      sizeOfRandomSubset = y.size();
    
    for ( iter = 1; iter <= maxIterations; iter++ ) 
    {
      NICE::Vector perm;
      randomPermutation(perm,indices,sizeOfRandomSubset);
 
      if ( timeAnalysis )
      {
        t.stop();
        Vector r;
        this->hik_kernel_multiply_fast(Tlookup, q, alpha, r);
        r = r - y;
        
        double res = r.normL2();
        double resMax = r.normInf();

        cerr << "SimpleGradientDescent: TIME " << t.getSum() << " " << res << " " << resMax << endl;

        t.start();
      }
     
      for ( int i = 0; i < sizeOfRandomSubset; i++)  
      {

        pseudoResidual(perm[i]) = -y(perm[i]) + (this->noise*alpha(perm[i]));
        for (uint j = 0; j < (uint)this->d; j++)
        {
          x_i = X_sorted(j,perm[i]);
          pseudoResidual(perm[i]) += Tlookup[j*hmax + q.quantize(x_i)];
        }
      
        //NOTE: this threshhold could also be a parameter of the function call
        if ( fabs(pseudoResidual(perm[i])) > 1e-7 )
        {
          alpha_old = alpha(perm[i]);
          alpha_new = alpha_old - (pseudoResidual(perm[i])/diagonalElements(perm[i]));
          alpha(perm[i]) = alpha_new;


          delta_alpha(perm[i]) = alpha_old-alpha_new;
         
          this->hikUpdateLookupTable(Tlookup, alpha_new, alpha_old, perm[i], q, pf ); // works correctly
          
        } else
        {
          delta_alpha(perm[i]) = 0.0;
        }
        
      }
      // after this only residual(i) is the valid residual... we should
      // really update the whole vector somehow
      
      double delta = delta_alpha.normL2();
      if ( verbose ) {
        cerr << "FastMinKernel::solveLin: iteration " << iter << " / " << maxIterations << endl;     
        cerr << "FastMinKernel::solveLin: delta = " << delta << endl;
        cerr << "FastMinKernel::solveLin: pseudo residual = " << pseudoResidual.scalarProduct(pseudoResidual) << endl;
      }
      
      if ( delta < minDelta ) 
      {
        if ( verbose )
          cerr << "FastMinKernel::solveLin: small delta" << endl;
        break;
      }    
    }
  }
  else //don't use random subsets
  {   
    for ( iter = 1; iter <= maxIterations; iter++ ) 
    {
      
      for ( uint i = 0; i < y.size(); i++ )
      {
          
        pseudoResidual(i) = -y(i) + (this->noise*alpha(i));
        for (uint j = 0; j < (uint) this->d; j++)
        {
          x_i = X_sorted(j,i);
          pseudoResidual(i) += Tlookup[j*hmax + q.quantize(x_i)];
        }
      
        //NOTE: this threshhold could also be a parameter of the function call
        if ( fabs(pseudoResidual(i)) > 1e-7 )
        {
          alpha_old = alpha(i);
          alpha_new = alpha_old - (pseudoResidual(i)/diagonalElements(i));
          alpha(i) = alpha_new;
          delta_alpha(i) = alpha_old-alpha_new;
          
          this->hikUpdateLookupTable(Tlookup, alpha_new, alpha_old, i, q, pf ); // works correctly
          
        } else
        {
          delta_alpha(i) = 0.0;
        }
        
      }
      
      double delta = delta_alpha.normL2();
      if ( verbose ) {
        cerr << "FastMinKernel::solveLin: iteration " << iter << " / " << maxIterations << endl;     
        cerr << "FastMinKernel::solveLin: delta = " << delta << endl;
        cerr << "FastMinKernel::solveLin: pseudo residual = " << pseudoResidual.scalarProduct(pseudoResidual) << endl;
      }
      
      if ( delta < minDelta ) 
      {
        if ( verbose )
          cerr << "FastMinKernel::solveLin: small delta" << endl;
        break;
      }    
    }
  }
  
  if (verboseMinimal)
    std::cerr << "FastMinKernel::solveLin -- needed " << iter << " iterations" << std::endl;
  return Tlookup;
}

void FastMinKernel::randomPermutation(NICE::Vector & permutation, const std::vector<int> & oldIndices, const int & newSize) const
{
  std::vector<int> indices(oldIndices);
  
  int resultingSize (std::min((int) (oldIndices.size()),newSize) );
  permutation.resize(resultingSize);
  
  for (int i = 0; i < resultingSize; i++)
  {
    int newIndex(rand() % indices.size());
    permutation[i] = indices[newIndex ];
    indices.erase(indices.begin() + newIndex);
  }
}

double FastMinKernel::getFrobNormApprox()
{
  double frobNormApprox(0.0);
  
  switch (approxScheme)
  {
    case MEDIAN:
    {
      //\| K \|_F^1 ~ (n/2)^2 \left( \sum_k \median_k \right)^2
      //motivation: estimate half of the values in dim k to zero and half of them to the median (-> lower bound expectation)
      for (int i = 0; i < d; i++)
      {
        double median = this->X_sorted.getFeatureValues(i).getMedian();
        frobNormApprox += median;
      }
      
      frobNormApprox = fabs(frobNormApprox) * n/2.0;
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
      for (int i = 0; i < d; i++)
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
      secondTerm *= (pow(this->n,2) - this->n);
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
      approxScheme = MEDIAN;
      break;
    }
    case 1:
    {
      approxScheme = EXPECTATION;
      break;
    }
    default:
    {
      approxScheme = MEDIAN;
      break;
    }
  }
}

void FastMinKernel::hikPrepareKVNApproximation(NICE::VVector & A) const
{
  A.resize(d);

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
  for (int i = 0; i < d; i++)
  {
    uint numNonZero = X_sorted.getNumberOfNonZeroElementsPerDimension(i);
    A[i].resize( numNonZero );
  }
  //  for more information see hik_prepare_alpha_multiplications
  
  for (int dim = 0; dim < d; dim++)
  {
    double squared_sum(0.0);

    int cntNonzeroFeat(0);
    
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    // loop through all elements in sorted order
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++ )
    {
      const SortedVectorSparse<double>::dataelement & de = i->second;
      
      // de: first - index, second - transformed feature
      double elem( de.second );
                
      squared_sum += pow( elem, 2 );
      A[dim][cntNonzeroFeat] = squared_sum;

      cntNonzeroFeat++;
    }
  }
}

double * FastMinKernel::hikPrepareKVNApproximationFast(NICE::VVector & A, const Quantization & q, const ParameterizedFunction *pf ) const
{
  //NOTE keep in mind: for doing this, we already have precomputed A using hikPrepareSquaredKernelVector!
  
  // number of quantization bins
  uint hmax = q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = pf->f ( 1, q.getPrototype(i) );
    } else {
      prototypes[i] = q.getPrototype(i);
    }


  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->d ];

  // loop through all dimensions
  for (int dim = 0; dim < this->d; dim++)
  {
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
      
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    int index = 0;
    // we use the quantization of the original features! the transformed feature were
    // already used to calculate A and B, this of course assumes monotonic functions!!!
    int qBin = q.quantize ( i->first ); 

    // the next loop is linear in max(hmax, n)
    // REMARK: this could be changed to hmax*log(n), when
    // we use binary search
    
    for (int j = 0; j < (int)hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n 1
        t = pow( fval, 2 ) * (n-nrZeroIndices-index);
      } else {

         // move to next example, if necessary   
        while ( (j >= qBin) && ( index < (this->n-nrZeroIndices)) )
        {
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= (uint)qBin) && ( index==(this->n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // the second term vanishes, which is logical, since all elements are smaller than j!
          t = A[dim][index];
        } else {
          // standard case
          t =  A[dim][index-1] + pow( fval, 2 ) * (n-nrZeroIndices-(index) );
//           A[dim][index-1] + fval * (n-nrZeroIndices-(index) );//fval*fval * (n-nrZeroIndices-(index-1) );
          
        }
      }

      Tlookup[ dim*hmax + j ] = t;
    }
  }

  delete [] prototypes;

  return Tlookup;  
}

double* FastMinKernel::hikPrepareLookupTableForKVNApproximation(const Quantization & q, const ParameterizedFunction *pf ) const
{
  // number of quantization bins
  uint hmax = q.size();

  // store (transformed) prototypes
  double *prototypes = new double [ hmax ];
  for ( uint i = 0 ; i < hmax ; i++ )
    if ( pf != NULL ) {
      // FIXME: the transformed prototypes could change from dimension to another dimension
      // We skip this flexibility ...but it should be changed in the future
      prototypes[i] = pf->f ( 1, q.getPrototype(i) );
    } else {
      prototypes[i] = q.getPrototype(i);
    }

  // creating the lookup table as pure C, which might be beneficial
  // for fast evaluation
  double *Tlookup = new double [ hmax * this->d ];
  
  // loop through all dimensions
  for (int dim = 0; dim < this->d; dim++)
  {
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n )
      continue;

    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
         
    SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin();
    SortedVectorSparse<double>::const_elementpointer iPredecessor = nonzeroElements.begin();
    
    // index of the element, which is always bigger than the current value fval
    int index = 0;
    
    // we use the quantization of the original features! Nevetheless, the resulting lookupTable is computed using the transformed ones
    int qBin = q.quantize ( i->first ); 
    
    double sum(0.0);
    
    for (uint j = 0; j < hmax; j++)
    {
      double fval = prototypes[j];
      double t;

      if (  (index == 0) && (j < (uint)qBin) ) {
        // current element is smaller than everything else
        // resulting value = fval * sum_l=1^n 1
        t = pow( fval, 2 ) * (n-nrZeroIndices-index);
      } else {

         // move to next example, if necessary   
        while ( (j >= (uint)qBin) && ( index < (this->n-nrZeroIndices)) )
        {
          sum += pow( i->second.second, 2 ); //i->dataElement.transformedFeatureValue
          
          index++;
          iPredecessor = i;
          i++;

          if ( i->first !=  iPredecessor->first )
            qBin = q.quantize ( i->first );
        }
        // compute current element in the lookup table and keep in mind that
        // index is the next element and not the previous one
        //NOTE pay attention: this is only valid if we all entries are positiv! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
        if ( (j >= (uint)qBin) && ( index==(this->n-1-nrZeroIndices) ) ) {
          // the current element (fval) is equal or bigger to the element indexed by index
          // the second term vanishes, which is logical, since all elements are smaller than j!
          t = sum;
        } else {
          // standard case
          t = sum + pow( fval, 2 ) * (n-nrZeroIndices-(index) );
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

void FastMinKernel::hikComputeKVNApproximation(const NICE::VVector & A, const NICE::SparseVector & xstar, double & norm, const ParameterizedFunction *pf ) 
{
  norm = 0.0;
  for (SparseVector::const_iterator i = xstar.begin(); i != xstar.end(); i++)
  {
  
    int dim = i->first;
    double fval = i->second;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero so let us ignore them completely
      continue;
    }

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if (position >= 0) 
      firstPart = (A[dim][position-nrZeroIndices]);
    else
      firstPart = 0.0;
    
    double secondPart( 0.0);
      
    if ( pf != NULL )
      fval = pf->f ( dim, fval );
    
    fval = fval * fval;
    
    if (position >= 0) 
      secondPart = fval * (n-nrZeroIndices-(position+1));
    else //if x_d^* is smaller than every non-zero training example
      secondPart = fval * (n-nrZeroIndices);
    
    // but apply using the transformed one
    norm += firstPart + secondPart;
  }  
}

void FastMinKernel::hikComputeKVNApproximationFast(const double *Tlookup, const Quantization & q, const NICE::SparseVector & xstar, double & norm) const
{
  norm = 0.0;
  // runtime is O(d) if the quantizer is O(1)
  for (SparseVector::const_iterator i = xstar.begin(); i != xstar.end(); i++ )
  {
    int dim = i->first;
    double v = i->second;
    // we do not need a parameterized function here, since the quantizer works on the original feature values. 
    // nonetheless, the lookup table was created using the parameterized function    
    uint qBin = q.quantize(v);
    
    norm += Tlookup[dim*q.size() + qBin];
  }  
}

void FastMinKernel::hikComputeKernelVector ( const NICE::SparseVector& xstar, NICE::Vector & kstar ) const
{
  //init
  kstar.resize(this->n);
  kstar.set(0.0);
  
  //let's start :)
  for (SparseVector::const_iterator i = xstar.begin(); i != xstar.end(); i++)
  {
  
    int dim = i->first;
    double fval = i->second;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero so let us ignore them completely
      continue;
    }
    

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
    
    //get the non-zero elements for this dimension  
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    
    //run over the non-zero elements and add the corresponding entries to our kernel vector

    int count(nrZeroIndices);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, count++ )
    {
      int origIndex(i->second.first); //orig index (i->second.second would be the transformed feature value)
      if (count <= position)
        kstar[origIndex] += i->first; //orig feature value
      else
        kstar[origIndex] += fval;
    }
  }  
}

    //////////////////////////////////////////
    // variance computation: non-sparse inputs
    //////////////////////////////////////////  

void FastMinKernel::hikComputeKVNApproximation(const NICE::VVector & A, const NICE::Vector & xstar, double & norm, const ParameterizedFunction *pf ) 
{
  norm = 0.0;
  int dim ( 0 );
  for (Vector::const_iterator i = xstar.begin(); i != xstar.end(); i++, dim++)
  {
  
    double fval = *i;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero so let us ignore them completely
      continue;
    }

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
  
    //NOTE again - pay attention! This is only valid if all entries are NOT negative! - if not, ask wether the current feature is greater than zero. If so, subtract the nrZeroIndices, if not do not
    double firstPart(0.0);
    //TODO in the "overnext" line there occurs the following error
    // Invalid read of size 8    
    if (position >= 0) 
      firstPart = (A[dim][position-nrZeroIndices]);
    else
      firstPart = 0.0;
    
    double secondPart( 0.0);
      
    if ( pf != NULL )
      fval = pf->f ( dim, fval );
    
    fval = fval * fval;
    
    if (position >= 0) 
      secondPart = fval * (n-nrZeroIndices-(position+1));
    else //if x_d^* is smaller than every non-zero training example
      secondPart = fval * (n-nrZeroIndices);
    
    // but apply using the transformed one
    norm += firstPart + secondPart;
  }  
}

void FastMinKernel::hikComputeKVNApproximationFast(const double *Tlookup, const Quantization & q, const NICE::Vector & xstar, double & norm) const
{
  norm = 0.0;
  // runtime is O(d) if the quantizer is O(1)
  int dim ( 0 );
  for (Vector::const_iterator i = xstar.begin(); i != xstar.end(); i++, dim++ )
  {
    double v = *i;
    // we do not need a parameterized function here, since the quantizer works on the original feature values. 
    // nonetheless, the lookup table was created using the parameterized function    
    uint qBin = q.quantize(v);
    
    norm += Tlookup[dim*q.size() + qBin];
  }  
}


void FastMinKernel::hikComputeKernelVector( const NICE::Vector & xstar, NICE::Vector & kstar) const
{
  //init
  kstar.resize(this->n);
  kstar.set(0.0);
  
  //let's start :)
  int dim ( 0 );
  for (NICE::Vector::const_iterator i = xstar.begin(); i != xstar.end(); i++, dim++)
  {
  
    double fval = *i;
    
    int nrZeroIndices = X_sorted.getNumberOfZeroElementsPerDimension(dim);
    if ( nrZeroIndices == n ) {
      // all features are zero so let us ignore them completely
      continue;
    }
    

    int position;

    //where is the example x^z_i located in
    //the sorted array? -> perform binary search, runtime O(log(n))
    // search using the original value
    X_sorted.findFirstLargerInDimension(dim, fval, position);
    position--;
    
    //get the non-zero elements for this dimension  
    const multimap< double, SortedVectorSparse<double>::dataelement> & nonzeroElements = X_sorted.getFeatureValues(dim).nonzeroElements();
    
    //run over the non-zero elements and add the corresponding entries to our kernel vector

    int count(nrZeroIndices);
    for ( SortedVectorSparse<double>::const_elementpointer i = nonzeroElements.begin(); i != nonzeroElements.end(); i++, count++ )
    {
      int origIndex(i->second.first); //orig index (i->second.second would be the transformed feature value)
      if (count <= position)
        kstar[origIndex] += i->first; //orig feature value
      else
        kstar[origIndex] += fval;
    }
  }  
}

// ---------------------- STORE AND RESTORE FUNCTIONS ----------------------

void FastMinKernel::restore ( std::istream & is, int format )
{
  if (is.good())
  {
    std::cerr << "FastMinKernel::restore  " << std::endl;
    is.precision (numeric_limits<double>::digits10 + 1);  
    
    string tmp;
    is >> tmp; //class name
    
    is >> tmp;
    is >> n;
    
    is >> tmp;
    is >> d;
    
    is >> tmp;
    is >> noise;
    
    is >> tmp;
    int approxSchemeInt;
    is >> approxSchemeInt;
    setApproximationScheme(approxSchemeInt);
   
    std::cerr << "start restoring X_sorted  " << std::endl;
    X_sorted.restore(is,format);
    std::cerr << " done :) " << std::endl;
   }
  else
  {
    std::cerr << "FastMinKernel::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }  
  std::cerr << " FMK restore ended " << std::endl;
}
void FastMinKernel::store ( std::ostream & os, int format ) const
{
  if (os.good())
  {
    os.precision (numeric_limits<double>::digits10 + 1);
    os << "FastMinKernel" << std::endl;
    os << "n: " << n << std::endl;
    os << "d: " << d << std::endl;
    os << "noise: " << noise << std::endl;
    os << "approxScheme: " << approxScheme << std::endl;    
    X_sorted.store(os,format);  
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

void FastMinKernel::setVerbose( const bool & _verbose)
{
  verbose = _verbose;
}

bool FastMinKernel::getVerbose( )   const
{
  return verbose;
}

void FastMinKernel::setDebug( const bool & _debug)
{
  debug = _debug;
  X_sorted.setDebug( _debug );
}

bool FastMinKernel::getDebug( )   const
{
  return debug;
}
