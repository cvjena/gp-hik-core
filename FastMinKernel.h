/** 
* @file FastMinKernel.h
* @brief Efficient GPs with HIK for classification by regression (Interface)
* @author Alexander Freytag
* @date 06-12-2011 (dd-mm-yyyy)
*/
#ifndef FASTMINKERNELINCLUDE
#define FASTMINKERNELINCLUDE

// STL includes
#include <iostream>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Exception.h>
#include <core/basics/Persistent.h>
// 
// 
#include <core/vector/MatrixT.h>
#include <core/vector/SparseVectorT.h>
#include <core/vector/VectorT.h>
#include <core/vector/VVector.h>

// gp-hik-core includes
#include "gp-hik-core/FeatureMatrixT.h"
#include "gp-hik-core/OnlineLearnable.h"
// 
#include "gp-hik-core/quantization/Quantization.h"
#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"

namespace NICE {


/** 
 * @class FastMinKernel
 * @brief Efficient GPs with HIK for classification by regression
 * @author Alexander Freytag
 */  
  
  /** interface to FastMinKernel implementation*/
  class FastMinKernel : public NICE::Persistent, public OnlineLearnable
  {

    protected:
      /** number of examples */
      uint ui_n;

      /** dimension of feature vectors */
      uint ui_d; 

      /** noise added to the diagonal of the kernel matrix */
      double d_noise;
      
      /** sorted matrix of features (sorted along each dimension) */
      NICE::FeatureMatrixT<double> X_sorted;
      
      //! verbose flag for output after calling the restore-function
      bool b_verbose;
      //! debug flag for output during debugging
      bool b_debug;      

      /** 
      * @brief Set number of examples
      * @author Alexander Freytag
      * @date 07-12-2011 (dd-mm-yyyy)
      */
      void set_n(const uint & _n){this->ui_n = _n;};
      
      /** 
      * @brief Set number of dimensions
      * @author Alexander Freytag
      * @date 07-12-2011 (dd-mm-yyyy)
      */
      void set_d(const uint & _d){this->ui_d = _d;};     

      /** 
      * @brief Prepare the efficient HIK-computations part 1: order the features in each dimension and save the permutation. Pay attention: X is of dim n x d, where as X_sorted is of dimensionality d x n!
      * @author Alexander Freytag
      * @date 07-12-2011 (dd-mm-yyyy)
      */
      void hik_prepare_kernel_multiplications(const std::vector<std::vector<double> > & _X, 
                                              NICE::FeatureMatrixT<double> & _X_sorted, 
                                              const uint & _dim = 0
                                             );
      
      void hik_prepare_kernel_multiplications ( const std::vector< const NICE::SparseVector * > & _X, 
                                                NICE::FeatureMatrixT<double> & _X_sorted, 
                                                const bool & _dimensionsOverExamples, 
                                                const uint & _dim = 0
                                              );
      
      void randomPermutation(NICE::Vector & _permutation, 
                             const std::vector<uint> & _oldIndices, 
                             const uint & _newSize
                            ) const;
      
      enum ApproximationScheme{ MEDIAN = 0, EXPECTATION=1};
      ApproximationScheme approxScheme;

    public:

      //------------------------------------------------------
      // several constructors and destructors
      //------------------------------------------------------
      
      /** 
      * @brief default constructor
      * @author Alexander Freytag
      * @date 20-04-2012 (dd-mm-yyyy)
      */
      FastMinKernel();      
      
      /** 
      * @brief recommended constructor, initialize with some data
      * @author Alexander Freytag
      * @date 06-12-2011 (dd-mm-yyyy)
      */
      FastMinKernel( const std::vector<std::vector<double> > & _X, 
                     const double _noise ,
                     const bool _debug = false, 
                     const uint & _dim = 0
                   );

      
      /**
      * @brief recommended constructor, just another sparse data structure
      *
      * @param X vector of sparse vector pointers
      * @param noise GP noise
      */
      FastMinKernel( const std::vector< const NICE::SparseVector * > & _X, 
                     const double _noise, 
                     const bool _debug = false, 
                     const bool & dimensionsOverExamples=false, 
                     const uint & _dim = 0
                   );

#ifdef NICE_USELIB_MATIO
      /**
      * @brief recommended constructor, intialize with some data given in a matlab-sparse struct and restricted with an example index
      *
      * @param X matlab-struct containing the feature vectors
      * @param noise additional noise variance of the labels
      * @param examples set of indices to include
      */
      FastMinKernel ( const sparse_t & _X, 
                      const double _noise, 
                      const std::map<uint, uint> & _examples, 
                      const bool _debug = false , 
                      const uint & _dim = 0);
#endif

      /** 
      * @brief Default destructor
      * @author Alexander Freytag
      * @date 06-12-2011 (dd-mm-yyyy)
      */
      ~FastMinKernel();

    ///////////////////// ///////////////////// /////////////////////
    //                         GET / SET
    //                   INCLUDING ACCESS OPERATORS
    ///////////////////// ///////////////////// /////////////////////       
      
      
      void setApproximationScheme(const ApproximationScheme & _approxScheme = MEDIAN) {approxScheme = _approxScheme;};
      
      virtual void setApproximationScheme(const int & _approxScheme = 0);
      
      /** 
      * @brief Get number of examples
      * @author Alexander Freytag
      * @date 07-12-2011 (dd-mm-yyyy)
      */
      uint get_n() const;
      
      /** 
      * @brief Get number of dimensions
      * @author Alexander Freytag
      * @date 07-12-2011 (dd-mm-yyyy)
      */
      uint get_d() const;

      /** 
      * @brief Computes the ratio of sparsity across the matrix
      * @author Alexander Freytag
      * @date 11-01-2012 (dd-mm-yyyy)
      */
      double getSparsityRatio() const;
      
      /** set verbose flag used for restore-functionality*/
      void setVerbose( const bool & _verbose);
      bool getVerbose( ) const;  
      
      /** set debug flag used for debug output*/
      void setDebug( const bool & _debug);
      bool getDebug( ) const;        
      
      //------------------------------------------------------
      // high level methods
      //------------------------------------------------------
      
      /**
      * @brief apply a parameterized function to the feature matrix
      * @author Alexander Freytag
      * @date 04-05-2012 (dd-mm-yyyy)
      *
      * @param pf the parameterized function (optional), if not given, nothing will be done
      */         
      void applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *_pf = NULL );
          
      /** 
      * @brief  Prepare the efficient HIK-computations part 2: calculate the partial sum for each dimension. Explicitely exploiting sparsity!!! Pay attention: X_sorted is of dimensionality d x n!
      * @author Alexander Freytag
      * @date 17-01-2012 (dd-mm-yyyy)
      */
      void hik_prepare_alpha_multiplications(const NICE::Vector & _alpha, 
                                             NICE::VVector & _A, 
                                             NICE::VVector & _B
                                            ) const;
            
      /**
      * @brief Computing K*alpha with the minimum kernel trick, explicitely exploiting sparsity!!!
      * @author Alexander Freytag
      * @date 17-01-2012 (dd-mm-yyyy)
      */
      void hik_kernel_multiply(const NICE::VVector & _A, 
                               const NICE::VVector & _B, 
                               const NICE::Vector & _alpha, 
                               NICE::Vector & _beta
                              ) const;
      void hik_kernel_multiply_fast(const double *_Tlookup, 
                                    const Quantization * _q, 
                                    const NICE::Vector & _alpha, 
                                    NICE::Vector & _beta
                                   ) const;

      /**
      * @brief Computing k_{*}*alpha using the minimum kernel trick and exploiting sparsity of the feature vector given
      *
      * @author Alexander Freytag
      * @date 20-01-2012 (dd-mm-yyyy)
      * @param A pre-computation matrix (VVector) (use the prepare method) 
      * @param B pre-computation matrix (VVector)
      * @param xstar new feature vector (SparseVector)
      * @param beta result of the scalar product
      * @param pf optional feature transformation
      */
      void hik_kernel_sum(const NICE::VVector & _A, 
                          const NICE::VVector & _B, 
                          const NICE::SparseVector & _xstar, 
                          double & _beta, 
                          const ParameterizedFunction *_pf = NULL 
                         ) const;
      
      /**
      * @brief Computing k_{*}*alpha using the minimum kernel trick and exploiting sparsity of the feature vector given
      * NOTE: Whenever possible, you should use sparse features to obtain significantly smaller computation times!
      *
      * @author Alexander Freytag
      * @date 18-06-2013 (dd-mm-yyyy)
      * @param A pre-computation matrix (VVector) (use the prepare method) 
      * @param B pre-computation matrix (VVector)
      * @param xstar new feature vector (non-sparse Vector)
      * @param beta result of the scalar product
      * @param pf optional feature transformation
      */
      void hik_kernel_sum(const NICE::VVector & _A, 
                          const NICE::VVector & _B, 
                          const NICE::Vector & _xstar, 
                          double & _beta, 
                          const ParameterizedFunction *_pf = NULL 
                         ) const;      
      
      /**
      * @brief compute beta = k_*^T * alpha by using a large lookup table created by hik_prepare_alpha_multiplications_fast
      * NOTE: Whenever possible, you should use sparse features to obtain significantly smaller computation times!
      * @author Alexander Freytag
      * @date 18-06-2013 (dd-mm-yyyy)
      *
      * @param Tlookup large lookup table calculated by hik_prepare_alpha_multiplications_fast
      * @param q Quantization object
      * @param xstar feature vector (indirect k_*)
      * @param beta result of the calculation
      */
      void hik_kernel_sum_fast(const double* _Tlookup, 
                               const Quantization * _q, 
                               const NICE::Vector & _xstar, 
                               double & _beta
                              ) const;
      /**
      * @brief compute beta = k_*^T * alpha by using a large lookup table created by hik_prepare_alpha_multiplications_fast
      * NOTE: Whenever possible, you should use sparse features to obtain significantly smaller computation times!
      * @author Alexander Frytag
      *
      * @param Tlookup large lookup table calculated by hik_prepare_alpha_multiplications_fast
      * @param q Quantization object
      * @param xstar feature vector (indirect k_*)
      * @param beta result of the calculation
      */      

      void hik_kernel_sum_fast(const double *_Tlookup, 
                               const Quantization * _q, 
                               const NICE::SparseVector & _xstar, 
                               double & _beta
                              ) const;

      /**
      * @brief compute lookup table for HIK calculation using quantized signals and prepare for K*alpha or k_*^T * alpha computations
      * @author Erik Rodner, Alexander Freytag
      *
      * @param alpha coefficient vector
      * @param A pre-calculation array computed by hik_prepare_alpha_multiplications
      * @param B pre-calculation array computed by hik_prepare_alpha_multiplications
      * @param q Quantization
      *
      * @return C standard vector representing a q.size()*n double matrix and the lookup table T. Elements can be accessed with
      * T[dim*q.size() + j], where j is a bin entry corresponding to quantization q.
      */
      double *hik_prepare_alpha_multiplications_fast(const NICE::VVector & _A, 
                                                     const NICE::VVector & _B, 
                                                     const Quantization * _q, 
                                                     const ParameterizedFunction *_pf = NULL 
                                                    ) const;
      
      /**
      * @brief compute lookup table for HIK calculation using quantized signals and prepare for K*alpha or k_*^T * alpha computations
      * @author Alexander Freytag
      *
      * @param alpha coefficient vector
      * @param q Quantization
      * @param pf ParameterizedFunction to change the original feature values
      *
      * @return C standard vector representing a q.size()*n double matrix and the lookup table T. Elements can be accessed with
      * T[dim*q.size() + j], where j is a bin entry corresponding to quantization q.
      */
      double* hikPrepareLookupTable(const NICE::Vector & _alpha, 
                                    const Quantization * _q, 
                                    const ParameterizedFunction *_pf = NULL
                                   ) const;

      /**
      * @brief update the lookup table for HIK calculation using quantized signals and prepare for K*alpha or k_*^T * alpha computations
      * @author Alexander Freytag
      *
      * @param T previously computed LUT, that will be changed
      * @param alphaNew new value of alpha at index idx
      * @param alphaOld old value of alpha at index idx
      * @param idx index in which alpha changed
      * @param q Quantization
      * @param pf ParameterizedFunction to change the original feature values
      */
      void hikUpdateLookupTable(double * _T, 
                                const double & _alphaNew, 
                                const double & _alphaOld, 
                                const uint & _idx, 
                                const Quantization * _q, 
                                const ParameterizedFunction *pf 
                               ) const;

      /**
      * @brief return a reference to the sorted feature matrix
      */
      FeatureMatrix & featureMatrix(void) { return X_sorted; };
      const FeatureMatrix & featureMatrix(void) const { return X_sorted; };
      
      /**
       * @brief solve the linear system K*alpha = y with the minimum kernel trick based on the algorithm of Wu (Wu10_AFD)
       * @note method converges slowly for large scale problems and even for normal scale :(
       * @author Paul Bodesheim
       * 
       * @param y right hand side of linear system
       * @param alpha final solution of the linear system
       * @param q Quantization
       * @param pf ParameterizedFunction to change the original feature values
       * @param useRandomSubsets true, if the order of examples in each iteration should be randomly sampled
       * @param maxIterations maximum number of iterations
       * @param sizeOfRandomSubset nr of Elements that should be randomly considered in each iteration (max: y.size())
       * @param minDelta minimum difference between two solutions alpha_t and alpha_{t+1} (convergence criterion)
       * 
       * @return C standard vector representing a q.size()*n double matrix and the lookup table T. Elements can be accessed with
       * T[dim*q.size() + j], where j is a bin entry corresponding to quantization q.
       **/
      double *solveLin(const NICE::Vector & _y, 
                       NICE::Vector & _alpha, 
                       const Quantization * _q, 
                       const ParameterizedFunction *_pf = NULL, 
                       const bool & _useRandomSubsets = true, 
                       uint _maxIterations = 10000, 
                       const uint & _sizeOfRandomSubset = 0, 
                       double _minDelta = 1e-7, 
                       bool _timeAnalysis = false
                      ) const;


      //! set the noise parameter
      void setNoise ( double _noise ) { this->d_noise = _noise; }

      //! get the current noise parameter
      double getNoise (void) const { return this->d_noise; }
      
      double getFrobNormApprox();
      
      
      /** 
      * @brief  Prepare the efficient HIK-computations for the squared kernel vector |k_*|^2 : calculate the partial squared sums for each dimension.
      * @author Alexander Freytag
      * @date 10-04-2012 (dd-mm-yyyy)
      */
      void hikPrepareKVNApproximation(NICE::VVector & _A) const;
      
      /** 
      * @brief  Compute lookup table for HIK calculation of |k_*|^2 assuming quantized test samples. You have to run hikPrepareSquaredKernelVector before
      * @author Alexander Freytag
      * @date 10-04-2012 (dd-mm-yyyy)
      * 
      * @param A pre-calculation array computed by hikPrepareSquaredKernelVector
      * @param q Quantization
      * @param pf Parameterized Function to efficiently apply a function to the underlying data
      *
      * @return C standard vector representing a q.size()*d double matrix and the lookup table T. Elements can be accessed with
      * T[dim*q.size() + j], where j is a bin entry corresponding to quantization q.
      */
      double * hikPrepareKVNApproximationFast(NICE::VVector & _A, const Quantization * _q, const ParameterizedFunction *_pf = NULL ) const;
      
      /**
      * @brief Compute lookup table for HIK calculation of |k_*|^2 assuming quantized test samples ( equals hikPrepareSquaredKernelVector + hikPrepareSquaredKernelVectorFast, but is faster). Approximation does not considere mixed terms between dimensions.
      * @author Alexander Freytag
      * @date 10-04-2012 (dd-mm-yyyy)
      *
      * @param q Quantization
      * @param pf ParameterizedFunction to change the original feature values
      *
      * @return C standard vector representing a q.size()*d double matrix and the lookup table T. Elements can be accessed with
      * T[dim*q.size() + j], where j is a bin entry corresponding to quantization q.
      */
      double* hikPrepareLookupTableForKVNApproximation(const Quantization * _q, const ParameterizedFunction *_pf = NULL) const;
      
    //////////////////////////////////////////
    // variance computation: sparse inputs
    //////////////////////////////////////////      
      
      /**
      * @brief Approximate norm = |k_*|^2 using the minimum kernel trick and exploiting sparsity of the given feature vector. Approximation does not considere mixed terms between dimensions.
      * @author Alexander Freytag
      * @date 10-04-2012 (dd-mm-yyyy)
      * 
      * @param A pre-computation matrix (VVector) (use the prepare method) 
      * @param xstar new feature vector (SparseVector)
      * @param norm result of the squared norm approximation
      * @param pf optional feature transformation
      */
      void hikComputeKVNApproximation(const NICE::VVector & _A, const NICE::SparseVector & _xstar, double & _norm, const ParameterizedFunction *_pf = NULL ) ;
      
      /**
      * @brief Approximate norm = |k_*|^2 using a large lookup table created by hikPrepareSquaredKernelVector and hikPrepareSquaredKernelVectorFast or directly using hikPrepareLookupTableForSquaredKernelVector. Approximation does not considere mixed terms between dimensions.
      * @author Alexander Freytag
      * @date 10-04-2012 (dd-mm-yyyy)
      *
      * @param Tlookup large lookup table
      * @param q Quantization object
      * @param xstar feature vector (indirect k_*)
      * @param norm result of the calculation
      */
      void hikComputeKVNApproximationFast(const double *_Tlookup, const Quantization * _q, const NICE::SparseVector & _xstar, double & _norm ) const;

      /**
      * @brief Compute the kernel vector k_* between training examples and test example. Runtime. O(n \times D). Exploiting sparsity
      * @author Alexander Freytag
      * @date 13-04-2012 (dd-mm-yyyy)
      *
      * @param xstar feature vector
      * @param kstar kernel vector
      */      
      void hikComputeKernelVector( const NICE::SparseVector & _xstar, NICE::Vector & _kstar) const;
      
    //////////////////////////////////////////
    // variance computation: non-sparse inputs
    //////////////////////////////////////////     
      
      /**
      * @brief Approximate norm = |k_*|^2 using the minimum kernel trick and exploiting sparsity of the given feature vector. Approximation does not considere mixed terms between dimensions.
      * @author Alexander Freytag
      * @date 19-12-2013 (dd-mm-yyyy)
      * 
      * @param A pre-computation matrix (VVector) (use the prepare method) 
      * @param xstar new feature vector (Vector)
      * @param norm result of the squared norm approximation
      * @param pf optional feature transformation
      */
      void hikComputeKVNApproximation(const NICE::VVector & _A, const NICE::Vector & _xstar, double & _norm, const ParameterizedFunction *_pf = NULL ) ;
      
      /**
      * @brief Approximate norm = |k_*|^2 using a large lookup table created by hikPrepareSquaredKernelVector and hikPrepareSquaredKernelVectorFast or directly using hikPrepareLookupTableForSquaredKernelVector. Approximation does not considere mixed terms between dimensions.
      * @author Alexander Freytag
      * @date 19-12-2013 (dd-mm-yyyy)
      *
      * @param Tlookup large lookup table
      * @param q Quantization object
      * @param xstar feature vector (indirect k_*)
      * @param norm result of the calculation
      */
      void hikComputeKVNApproximationFast(const double *_Tlookup, const Quantization * _q, const NICE::Vector & _xstar, double & _norm ) const;      
      
      /**
      * @brief Compute the kernel vector k_* between training examples and test example. Runtime. O(n \times D). Does not exploit sparsity - deprecated!
      * @author Alexander Freytag
      * @date 19-12-2013 (dd-mm-yyyy)
      *
      * @param xstar feature vector
      * @param kstar kernel vector
      */      
      void hikComputeKernelVector( const NICE::Vector & _xstar, NICE::Vector & _kstar) const;      
      
      /** Persistent interface */
      virtual void restore ( std::istream & _is, int _format = 0 );
      virtual void store ( std::ostream & _os, int _format = 0 ) const; 
      virtual void clear ();
      
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
      
    virtual void addExample( const NICE::SparseVector * _example, 
                             const double & _label, 
                             const bool & _performOptimizationAfterIncrement = true
                           );

    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & _newExamples,
                                      const NICE::Vector & _newLabels,
                                      const bool & _performOptimizationAfterIncrement = true
                                    );  
    

      /**
      * @brief Add a new example to the feature-storage. You have to update the corresponding variables explicitely after that.
      * @author Alexander Freytag
      * @date 02-01-2014 (dd-mm-yyyy)
      *
      * @param example new feature vector
      */       
      void addExample(const NICE::SparseVector * _example, const NICE::ParameterizedFunction *_pf = NULL);
      
      /**
      * @brief Add multiple new example to the feature-storage. You have to update the corresponding variables explicitely after that.
      * @author Alexander Freytag
      * @date 02-01-2014 (dd-mm-yyyy)
      *
      * @param newExamples new feature vectors
      */       
      void addMultipleExamples(const std::vector<const NICE::SparseVector * > & _newExamples, const NICE::ParameterizedFunction *_pf = NULL);        
      
      
     

  };

} // namespace

#endif
