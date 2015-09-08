/** 
* @file GPLikelihoodApprox.h
* @brief GP likelihood approximation as a cost function (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 02/09/2012

*/
#ifndef _NICE_GPLIKELIHOODAPPROXINCLUDE
#define _NICE_GPLIKELIHOODAPPROXINCLUDE

// STL includes
#include <map>

// NICE-core includes
#include <core/algebra/EigValues.h>
#include <core/algebra/IterativeLinearSolver.h>
// 
#include <core/basics/Config.h>
#include <core/optimization/blackbox/CostFunction.h>
// 
#include <core/vector/VectorT.h>

// gp-hik-core includes
#include "gp-hik-core/FastMinKernel.h"
#include "gp-hik-core/ImplicitKernelMatrix.h"
#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"

namespace NICE {

 /** 
 * @class GPLikelihoodApprox
 * @brief GP likelihood approximation as a cost function
 * @author Erik Rodner, Alexander Freytag
 */
 
class GPLikelihoodApprox : public OPTIMIZATION::CostFunction
{

  protected:
    
    /** method computing eigenvalues */
    EigValues *eig;

    /** method for solving linear equation systems */
    IterativeLinearSolver *linsolver;

    /** object providing fast calculations */
    ImplicitKernelMatrix *ikm;

    /** set of binary label vectors */
    std::map<uint, Vector> binaryLabels;
   
    /** number of classes */
    uint nrOfClasses;
    
    /** To define how fine the approximation of the squared frobenius norm will be*/
    int nrOfEigenvaluesToConsider;
    
    //! only for debugging purposes, printing some statistics
    void calculateLikelihood ( double _mypara, 
                               const FeatureMatrix & _f, 
                               const std::map< uint, NICE::Vector > & _yset, 
                               double _noise, 
                               double _lambdaMax );

    //! last alpha vectors computed (from previous IL-step)
    std::map<uint, NICE::Vector> * initialAlphaGuess;
    
    //! alpha vectors of the best solution
    std::map<uint, Vector> min_alphas;

    //! minimal value of the likelihood
    double min_nlikelihood;

    //! best hyperparameter vector
    Vector min_parameter;

    //! function value pairs already visited
    std::map<unsigned long, double> alreadyVisited;

    //! to check whether the current solution of our optimization routine is too small
    double parameterLowerBound;
    //! to check whether the current solution of our optimization routine is too large
    double parameterUpperBound;

    //! Just for debugging to verify wheter the likelihood approximation is useful at all
    bool verifyApproximation;
    
    /** verbose flag */
    bool verbose;    
    /** debug flag for several outputs useful for debugging*/
    bool debug;  
    

  public:

    // ------ constructors and destructors ------
    /** simple constructor */
    GPLikelihoodApprox( const std::map<uint, Vector> & _binaryLabels, 
                        ImplicitKernelMatrix *_ikm,
                        IterativeLinearSolver *_linsolver,
                        EigValues *_eig,
                        bool _verifyApproximation = false,
                        int _nrOfEigenvaluesToConsider = 1
                      );
      
    /** simple destructor */
    virtual ~GPLikelihoodApprox();
     
    // ------ main methods ------
    
    /**
    * @brief Compute alpha vectors for given hyperparameters
    *
    * @param x vector with specified hyperparameters to evaluate their likelihood
    *
    * @return void
    */    
    void computeAlphaDirect(const OPTIMIZATION::matrix_type & _x, 
                            const NICE::Vector & _eigenValues
                           );
    
    /**
    * @brief Evaluate the likelihood for given hyperparameters
    *
    * @param x vector with specified hyperparameters to evaluate their likelihood
    *
    * @return likelihood 
    */
    virtual double evaluate(const OPTIMIZATION::matrix_type & x);
     
    
    // ------ get and set methods ------
    const NICE::Vector & getBestParameters () const { return min_parameter; };
    const std::map<uint, Vector> & getBestAlphas () const;
    
    void setParameterLowerBound(const double & _parameterLowerBound);
    void setParameterUpperBound(const double & _parameterUpperBound);
    
    void setInitialAlphaGuess(std::map<uint, NICE::Vector> * _initialAlphaGuess);
    void setBinaryLabels(const std::map<uint, Vector> & _binaryLabels);
    
    void setVerbose( const bool & _verbose );
    void setDebug( const bool & _debug );
    
    bool getVerbose ( ) { return verbose; } ;
    bool getDebug ( ) { return debug; } ;
};

}

#endif
