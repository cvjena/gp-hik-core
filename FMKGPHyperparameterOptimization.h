/** 
* @file FMKGPHyperparameterOptimization.h
* @brief Heart of the framework to set up everything, perform optimization, classification, and variance prediction (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 01/02/2012

*/
#ifndef _NICE_FMKGPHYPERPARAMETEROPTIMIZATIONINCLUDE
#define _NICE_FMKGPHYPERPARAMETEROPTIMIZATIONINCLUDE

// STL includes
#include <vector>
#include <set>
#include <map>

// NICE-core includes
#include <core/algebra/EigValues.h>
#include <core/algebra/IterativeLinearSolver.h>
#include <core/basics/Config.h>
#include <core/basics/Persistent.h>
#include <core/vector/VectorT.h>

#ifdef NICE_USELIB_MATIO
#include <core/matlabAccess/MatFileIO.h>
#endif

// gp-hik-core includes
#include "gp-hik-core/FastMinKernel.h"
#include "gp-hik-core/GPLikelihoodApprox.h"
#include "gp-hik-core/IKMLinearCombination.h"
#include "gp-hik-core/OnlineLearnable.h"
#include "gp-hik-core/Quantization.h"


#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"

namespace NICE {
  
  /** 
 * @class FMKGPHyperparameterOptimization
 * @brief Heart of the framework to set up everything, perform optimization, classification, and variance prediction
 * @author Erik Rodner, Alexander Freytag
 */
  
class FMKGPHyperparameterOptimization : public NICE::Persistent, public NICE::OnlineLearnable
{
  protected:
    enum {
      OPT_GREEDY = 0,
      OPT_DOWNHILLSIMPLEX,
      OPT_NONE,
      OPT_NUMBEROFMETHODS
    };

    /** optimization method used */
    int optimizationMethod;

    /** the parameterized function we use within the minimum kernel */
    ParameterizedFunction *pf;

    /** method computing eigenvalues */
    EigValues *eig;

    /** method for solving linear equation systems */
    IterativeLinearSolver *linsolver;

    /** object which stores our sorted data and provides fast hik functions */
    FastMinKernel *fmk;

    /** object which stores our quantization object */
    Quantization *q;

    /** verbose flag */
    bool verbose;    
    /** verbose flag for time measurement outputs */
    bool verboseTime;        
    /** debug flag for several outputs useful for debugging*/
    bool debug;    

    /** optimization parameters */
    double parameterUpperBound;
    double parameterLowerBound;
    double parameterStepSize;
    int ils_max_iterations;

    int downhillSimplexMaxIterations;
    double downhillSimplexTimeLimit;
    double downhillSimplexParamTol;

    /** whether to compute the likelihood with the usual method */
    bool verifyApproximation;
    
    /** number of Eigenvalues to consider in the approximation of |K|_F */
    int nrOfEigenvaluesToConsider;
    
    /** number of Eigenvalues to consider in the fine approximation of the predictive variance */
    int nrOfEigenvaluesToConsiderForVarApprox;

    typedef VVector PrecomputedType;

    /** precomputed arrays and lookup tables */
    std::map< int, PrecomputedType > precomputedA;
    std::map< int, PrecomputedType > precomputedB;
    std::map< int, double * > precomputedT;

    PrecomputedType precomputedAForVarEst;
    double * precomputedTForVarEst;

    //! optimize noise with the GP likelihood
    bool optimizeNoise;     
       
    //! k largest eigenvalues of the kernel matrix (k == nrOfEigenvaluesToConsider)
    NICE::Vector eigenMax;

    //! eigenvectors corresponding to k largest eigenvalues (k == nrOfEigenvaluesToConsider) -- format: nxk
    NICE::Matrix eigenMaxVectors;
    
    //! needed for optimization and variance approximation
    IKMLinearCombination * ikmsum;
    
    //! storing the labels is needed for Incremental Learning (re-optimization)
    NICE::Vector labels;
    

    //! calculate binary label vectors using a multi-class label vector
    int prepareBinaryLabels ( std::map<int, NICE::Vector> & binaryLabels, const NICE::Vector & y , std::set<int> & myClasses);     
    
    //! prepare the GPLike object for given binary labels and already given ikmsum-object
    inline void setupGPLikelihoodApprox( GPLikelihoodApprox * & gplike, const std::map<int, NICE::Vector> & binaryLabels, uint & parameterVectorSize);    
    
    //! update eigenvectors and eigenvalues for given ikmsum-objects and a method to compute eigenvalues
    inline void updateEigenDecomposition( const int & i_noEigenValues );
    
    //! core of the optimize-functions
    inline void performOptimization( GPLikelihoodApprox & gplike, const uint & parameterVectorSize);
    
    //! apply the optimized transformation values to the underlying features
    inline void transformFeaturesWithOptimalParameters(const GPLikelihoodApprox & gplike, const uint & parameterVectorSize);
    
    //! build the resulting matrices A and B as well as lookup tables T for fast evaluations using the optimized parameter settings
    inline void computeMatricesAndLUTs( const GPLikelihoodApprox & gplike);
    
     
    //! store the class number of the positive class (i.e., larger class no), only used in binary settings
    int binaryLabelPositive;
    //! store the class number of the negative class (i.e., smaller class no), only used in binary settings
    int binaryLabelNegative;
    
    //! contains all class numbers of the currently known classes
    std::set<int> knownClasses;
    
    bool b_usePreviousAlphas;
    
    //! we store the alpha vectors for good initializations in the IL setting
    std::map<int, NICE::Vector> previousAlphas;  

    //! Update matrices (A, B, LUTs) and optionally find optimal parameters after adding (a) new example(s).  
    void updateAfterIncrement (
      const std::set<int> newClasses,
      const bool & performOptimizationAfterIncrement = false
    );    
  

    
  public:  
    

    FMKGPHyperparameterOptimization();
    
    /**
    * @brief standard constructor
    *
    * @param pf pointer to a parameterized function used within the minimum kernel min(f(x_i), f(x_j)) (will not be deleted)
    * @param noise GP label noise
    * @param fmk pointer to a pre-initialized structure (will be deleted)
    */
    FMKGPHyperparameterOptimization( const Config *conf, ParameterizedFunction *pf, FastMinKernel *fmk = NULL, const std::string & confSection = "GPHIKClassifier" );
      
    /** simple destructor */
    virtual ~FMKGPHyperparameterOptimization();
    
    ///////////////////// ///////////////////// /////////////////////
    //                         GET / SET
    ///////////////////// ///////////////////// ///////////////////// 
    void setParameterUpperBound(const double & _parameterUpperBound);
    void setParameterLowerBound(const double & _parameterLowerBound);  
    
    std::set<int> getKnownClassNumbers ( ) const;
    
    ///////////////////// ///////////////////// /////////////////////
    //                      CLASSIFIER STUFF
    ///////////////////// ///////////////////// /////////////////////  
    
    void initialize( const Config *conf, ParameterizedFunction *pf, FastMinKernel *fmk = NULL, const std::string & confSection = "GPHIKClassifier" );
       
#ifdef NICE_USELIB_MATIO
    /**
    * @brief Perform hyperparameter optimization
    *
    * @param data MATLAB data structure, like a feature matrix loaded from ImageNet
    * @param y label vector (arbitrary), will be converted into a binary label vector
    * @param positives set of positive examples (indices)
    * @param negatives set of negative examples (indices)
    */
    void optimizeBinary ( const sparse_t & data, const NICE::Vector & y, const std::set<int> & positives, const std::set<int> & negatives, double noise );

    /**
    * @brief Perform hyperparameter optimization for GP multi-class or binary problems
    *
    * @param data MATLAB data structure, like a feature matrix loaded from ImageNet
    * @param y label vector with multi-class labels
    * @param examples mapping of example index to new index
    */
    void optimize ( const sparse_t & data, const NICE::Vector & y, const std::map<int, int> & examples, double noise );
#endif

    /**
    * @brief Perform hyperparameter optimization (multi-class or binary) assuming an already initialized fmk object
    *
    * @param y label vector (multi-class as well as binary labels supported)
    */
    void optimize ( const NICE::Vector & y );
    
    /**
    * @brief Perform hyperparameter optimization (multi-class or binary) assuming an already initialized fmk object
    *
    * @param binLabels vector of binary label vectors (1,-1) and corresponding class no.
    */
    void optimize ( std::map<int, NICE::Vector> & binaryLabels );    
    
    /**
    * @brief Compute the necessary variables for appxorimations of predictive variance (LUTs), assuming an already initialized fmk object
    * @author Alexander Freytag
    * @date 11-04-2012 (dd-mm-yyyy)
    */       
    void prepareVarianceApproximationRough();
    
    /**
    * @brief Compute the necessary variables for fine appxorimations of predictive variance (EVs), assuming an already initialized fmk object
    * @author Alexander Freytag
    * @date 11-04-2012 (dd-mm-yyyy)
    */       
    void prepareVarianceApproximationFine();    
    
    /**
    * @brief classify an example 
    *
    * @param x input example (sparse vector)
    * @param scores scores for each class number
    *
    * @return class number achieving the best score
    */
    int classify ( const NICE::SparseVector & x, SparseVector & scores ) const;
    
    /**
    * @brief classify an example that is given as non-sparse vector
    * NOTE: whenever possible, you should sparse vectors to obtain significantly smaller computation times
    * 
    * @date 18-06-2013 (dd-mm-yyyy)
    * @author Alexander Freytag
    *
    * @param x input example (non-sparse vector)
    * @param scores scores for each class number
    *
    * @return class number achieving the best score
    */
    int classify ( const NICE::Vector & x, SparseVector & scores ) const;    

    //////////////////////////////////////////
    // variance computation: sparse inputs
    //////////////////////////////////////////
    
    /**
    * @brief compute predictive variance for a given test example using a rough approximation: k_{**} -  k_*^T (K+\sigma I)^{-1} k_* <= k_{**} - |k_*|^2 * 1 / \lambda_max(K + \sigma I), where we approximate |k_*|^2 by neglecting the mixed terms
    * @author Alexander Freytag
    * @date 10-04-2012 (dd-mm-yyyy)
    * @param x input example
    * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceApproximateRough(const NICE::SparseVector & x, double & predVariance ) const;
    
    /**
    * @brief compute predictive variance for a given test example using a fine approximation  (k eigenvalues and eigenvectors to approximate the quadratic term)
    * @author Alexander Freytag
    * @date 18-04-2012 (dd-mm-yyyy)
    * @param x input example
     * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceApproximateFine(const NICE::SparseVector & x, double & predVariance ) const; 
    
    /**
    * @brief compute exact predictive variance for a given test example using ILS methods (exact, but more time consuming than approx versions)
    * @author Alexander Freytag
    * @date 10-04-2012 (dd-mm-yyyy)
    * @param x input example
     * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceExact(const NICE::SparseVector & x, double & predVariance ) const; 
    
    
    //////////////////////////////////////////
    // variance computation: non-sparse inputs
    //////////////////////////////////////////
    
    /**
    * @brief compute predictive variance for a given test example using a rough approximation: k_{**} -  k_*^T (K+\sigma I)^{-1} k_* <= k_{**} - |k_*|^2 * 1 / \lambda_max(K + \sigma I), where we approximate |k_*|^2 by neglecting the mixed terms
    * @author Alexander Freytag
    * @date 19-12-2013 (dd-mm-yyyy)
    * @param x input example
    * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceApproximateRough(const NICE::Vector & x, double & predVariance ) const;    

   
    
    /**
    * @brief compute predictive variance for a given test example using a fine approximation  (k eigenvalues and eigenvectors to approximate the quadratic term)
    * @author Alexander Freytag
    * @date 19-12-2013 (dd-mm-yyyy)
    * @param x input example
     * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceApproximateFine(const NICE::Vector & x, double & predVariance ) const;      
    

    
   /**
    * @brief compute exact predictive variance for a given test example using ILS methods (exact, but more time consuming than approx versions)
    * @author Alexander Freytag
    * @date 19-12-2013 (dd-mm-yyyy)
    * @param x input example
    * @param predVariance contains the approximation of the predictive variance
    *
    */    
    void computePredictiveVarianceExact(const NICE::Vector & x, double & predVariance ) const;  
    
    
    
    
    
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT ///////////////////// 
    
    void restore ( std::istream & is, int format = 0 );
    void store ( std::ostream & os, int format = 0 ) const;
    void clear ( ) ;
    
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////    
    
    virtual void addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement = true
			   );
			   
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement = true
				    );         
};

}

#endif
