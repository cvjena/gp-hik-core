/** 
* @file FMKGPHyperparameterOptimization.h
* @brief Heart of the framework to set up everything, perform optimization, classification, and variance prediction (Interface)
* @author Alexander Freytag, Erik Rodner
* @date 01-02-2012 (dd-mm-yyyy)
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
 * @author Alexander Freytag, Erik Rodner
 */
  
class FMKGPHyperparameterOptimization : public NICE::Persistent, public NICE::OnlineLearnable
{
  protected:
    
    /////////////////////////
    /////////////////////////
    // PROTECTED VARIABLES //
    /////////////////////////
    ///////////////////////// 
    
    ///////////////////////////////////
    // output/debug related settings //   
    ///////////////////////////////////
    
    /** verbose flag */
    bool verbose;    
    /** verbose flag for time measurement outputs */
    bool verboseTime;        
    /** debug flag for several outputs useful for debugging*/
    bool debug;    
    
    //////////////////////////////////////
    // classification related variables //
    //////////////////////////////////////
    
    /** per default, we perform classification, if not stated otherwise */
    bool b_performRegression;
    
    /** object storing sorted data and providing fast hik methods */
    NICE::FastMinKernel *fmk;

    /** object performing feature quantization */
    NICE::Quantization *q;
    
    /** the parameterized function we use within the minimum kernel */
    NICE::ParameterizedFunction *pf;

    /** method for solving linear equation systems - needed to compute K^-1 \times y */
    IterativeLinearSolver *linsolver;
    
    /** Max. number of iterations the iterative linear solver is allowed to run */
    int ils_max_iterations;    
    
    /** Simple type definition for precomputation matrices used for fast classification */
    typedef VVector PrecomputedType;

    /** precomputed arrays A (1 per class) needed for classification without quantization  */
    std::map< int, PrecomputedType > precomputedA;    
    /** precomputed arrays B (1 per class) needed for classification without quantization  */
    std::map< int, PrecomputedType > precomputedB;
    
    /** precomputed LUTs (1 per class) needed for classification with quantization  */
    std::map< int, double * > precomputedT;  
    
    //! storing the labels is needed for Incremental Learning (re-optimization)
    NICE::Vector labels; 
    
    //! store the class number of the positive class (i.e., larger class no), only used in binary settings
    int binaryLabelPositive;
    //! store the class number of the negative class (i.e., smaller class no), only used in binary settings
    int binaryLabelNegative;
    
    //! contains all class numbers of the currently known classes
    std::set<int> knownClasses;
    
    //! container for multiple kernel matrices (e.g., a data-containing kernel matrix (GMHIKernel) and a noise matrix (IKMNoise) )
    NICE::IKMLinearCombination * ikmsum;    
    
  
    /////////////////////////////////////
    // optimization related parameters //
    /////////////////////////////////////
    
    enum {
      OPT_GREEDY = 0,
      OPT_DOWNHILLSIMPLEX,
      OPT_NONE
    };

    /** specify the optimization method used (see corresponding enum) */
    int optimizationMethod;
    
    //! whether or not to optimize noise with the GP likelihood
    bool optimizeNoise;     
    
    /** upper bound for hyper parameters to optimize */
    double parameterUpperBound;
    
    /** lower bound for hyper parameters to optimize */
    double parameterLowerBound;
    
        // specific to greedy optimization
    /** step size used in grid based greedy optimization technique */
    double parameterStepSize;
    
        // specific to downhill simplex optimization
    /** Max. number of iterations the downhill simplex optimizer is allowed to run */
    int downhillSimplexMaxIterations;
    
    /** Max. time the downhill simplex optimizer is allowed to run */
    double downhillSimplexTimeLimit;
    
    /** Max. number of iterations the iterative linear solver is allowed to run */
    double downhillSimplexParamTol;
    
    
      // likelihood computation related variables

    /** whether to compute the exact likelihood by computing the exact kernel matrix (not recommended - only for debugging/comparison purpose) */
    bool verifyApproximation;

    /** method computing eigenvalues and eigenvectors*/
    NICE::EigValues *eig;
    
    /** number of Eigenvalues to consider in the approximation of |K|_F used for approximating the likelihood */
    int nrOfEigenvaluesToConsider;
    
    //! k largest eigenvalues of the kernel matrix (k == nrOfEigenvaluesToConsider)
    NICE::Vector eigenMax;

    //! eigenvectors corresponding to k largest eigenvalues (k == nrOfEigenvaluesToConsider) -- format: nxk
    NICE::Matrix eigenMaxVectors;
    

    ////////////////////////////////////////////
    // variance computation related variables //
    ////////////////////////////////////////////
    
    /** number of Eigenvalues to consider in the fine approximation of the predictive variance (fine approximation only) */
    int nrOfEigenvaluesToConsiderForVarApprox;
    
    /** precomputed array needed for rough variance approximation without quantization */ 
    PrecomputedType precomputedAForVarEst;
    
    /** precomputed LUT needed for rough variance approximation with quantization  */
    double * precomputedTForVarEst;    
    
    /////////////////////////////////////////////////////
    // online / incremental learning related variables //
    /////////////////////////////////////////////////////

    /** whether or not to use previous alpha solutions as initialization after adding new examples*/
    bool b_usePreviousAlphas;
    
    //! store alpha vectors for good initializations in the IL setting, if activated
    std::map<int, NICE::Vector> previousAlphas;     

    
    /////////////////////////
    /////////////////////////
    //  PROTECTED METHODS  //
    /////////////////////////
    /////////////////////////
    

    /**
    * @brief calculate binary label vectors using a multi-class label vector
    * @author Alexander Freytag
    */    
    int prepareBinaryLabels ( std::map<int, NICE::Vector> & binaryLabels, const NICE::Vector & y , std::set<int> & myClasses);     
    
    /**
    * @brief prepare the GPLike object for given binary labels and already given ikmsum-object
    * @author Alexander Freytag
    */
    inline void setupGPLikelihoodApprox( GPLikelihoodApprox * & gplike, const std::map<int, NICE::Vector> & binaryLabels, uint & parameterVectorSize);    
    
    /**
    * @brief update eigenvectors and eigenvalues for given ikmsum-objects and a method to compute eigenvalues
    * @author Alexander Freytag
    */
    inline void updateEigenDecomposition( const int & i_noEigenValues );
    
    /**
    * @brief core of the optimize-functions
    * @author Alexander Freytag
    */
    inline void performOptimization( GPLikelihoodApprox & gplike, const uint & parameterVectorSize);
    
    /**
    * @brief apply the optimized transformation values to the underlying features
    * @author Alexander Freytag
    */    
    inline void transformFeaturesWithOptimalParameters(const GPLikelihoodApprox & gplike, const uint & parameterVectorSize);
    
    /**
    * @brief build the resulting matrices A and B as well as lookup tables T for fast evaluations using the optimized parameter settings
    * @author Alexander Freytag
    */
    inline void computeMatricesAndLUTs( const GPLikelihoodApprox & gplike);
    
     

    /**
    * @brief Update matrices (A, B, LUTs) and optionally find optimal parameters after adding (a) new example(s).  
    * @author Alexander Freytag
    */           
    void updateAfterIncrement (
      const std::set<int> newClasses,
      const bool & performOptimizationAfterIncrement = false
    );    
  

    
  public:  
    
    /**
    * @brief simple constructor
    * @author Alexander Freytag
    */
    FMKGPHyperparameterOptimization( const bool & b_performRegression = false);
        
    /**
    * @brief standard constructor
    *
    * @param pf pointer to a parameterized function used within the minimum kernel min(f(x_i), f(x_j)) (will not be deleted)
    * @param noise GP label noise
    * @param fmk pointer to a pre-initialized structure (will be deleted)
    */
    FMKGPHyperparameterOptimization( const Config *conf, ParameterizedFunction *pf, FastMinKernel *fmk = NULL, const std::string & confSection = "GPHIKClassifier" );
      
    /**
    * @brief standard destructor
    * @author Alexander Freytag
    */
    virtual ~FMKGPHyperparameterOptimization();
    
    ///////////////////// ///////////////////// /////////////////////
    //                         GET / SET
    ///////////////////// ///////////////////// /////////////////////
    
    /**
    * @brief Set lower bound for hyper parameters to optimize
    * @author Alexander Freytag
    */    
    void setParameterUpperBound(const double & _parameterUpperBound);
    /**
    * @brief Set upper bound for hyper parameters to optimize
    * @author Alexander Freytag
    */    
    void setParameterLowerBound(const double & _parameterLowerBound);  
    
    /**
    * @brief Get the currently known class numbers
    * @author Alexander Freytag
    */    
    std::set<int> getKnownClassNumbers ( ) const;
    
    ///////////////////// ///////////////////// /////////////////////
    //                      CLASSIFIER STUFF
    ///////////////////// ///////////////////// /////////////////////  
    
    /**
    * @brief Set variables and parameters to default or config-specified values
    * @author Alexander Freytag
    */       
    void initialize( const Config *conf, ParameterizedFunction *pf, FastMinKernel *fmk = NULL, const std::string & confSection = "GPHIKClassifier" );
       
#ifdef NICE_USELIB_MATIO
    /**
    * @brief Perform hyperparameter optimization
    * @author Alexander Freytag
    * 
    * @param data MATLAB data structure, like a feature matrix loaded from ImageNet
    * @param y label vector (arbitrary), will be converted into a binary label vector
    * @param positives set of positive examples (indices)
    * @param negatives set of negative examples (indices)
    */
    void optimizeBinary ( const sparse_t & data, const NICE::Vector & y, const std::set<int> & positives, const std::set<int> & negatives, double noise );

    /**
    * @brief Perform hyperparameter optimization for GP multi-class or binary problems
    * @author Alexander Freytag
    * 
    * @param data MATLAB data structure, like a feature matrix loaded from ImageNet
    * @param y label vector with multi-class labels
    * @param examples mapping of example index to new index
    */
    void optimize ( const sparse_t & data, const NICE::Vector & y, const std::map<int, int> & examples, double noise );
#endif

    /**
    * @brief Perform hyperparameter optimization (multi-class or binary) assuming an already initialized fmk object
    * @author Alexander Freytag
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
    
    /** 
     * @brief Load current object from external file (stream)
     * @author Alexander Freytag
     */     
    void restore ( std::istream & is, int format = 0 );
    
    /** 
     * @brief Save current object to external file (stream)
     * @author Alexander Freytag
     */      
    void store ( std::ostream & os, int format = 0 ) const;
    
    /** 
     * @brief Clear current object
     * @author Alexander Freytag
     */      
    void clear ( ) ;
    
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////    
    
    /** 
     * @brief add a new example
     * @author Alexander Freytag
     */       
    virtual void addExample( const NICE::SparseVector * example, 
                             const double & label, 
                             const bool & performOptimizationAfterIncrement = true
                           );

    /** 
     * @brief add several new examples
     * @author Alexander Freytag
     */    
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
                                      const NICE::Vector & newLabels,
                                      const bool & performOptimizationAfterIncrement = true
                                    );         
};

}

#endif
