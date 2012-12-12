/** 
* @file GPLikelihoodApprox.h
* @brief GP likelihood approximation as a cost function (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 02/09/2012

*/
#ifndef _NICE_GPLIKELIHOODAPPROXINCLUDE
#define _NICE_GPLIKELIHOODAPPROXINCLUDE

#include <map>

#include <core/vector/VectorT.h>
#include <core/basics/Config.h>
#include <core/algebra/EigValues.h>
#include <core/algebra/IterativeLinearSolver.h>

#include <core/optimization/blackbox/CostFunction.h>

#include "FastMinKernel.h"
#include "ImplicitKernelMatrix.h"

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
    std::map<int, Vector> binaryLabels;
   
    /** number of classes */
    int nrOfClasses;
    
    /** To define how fine the approximation of the squared frobenius norm will be*/
    int nrOfEigenvaluesToConsider;
    
    //! only for debugging purposes, printing some statistics
    void calculateLikelihood ( double mypara, const FeatureMatrix & f, const std::map< int, NICE::Vector > & yset, double noise, double lambdaMax );

    //! last alpha vectors computed (from previous IL-step)
    std::map<int, Vector> * lastAlphas;
    
    //! alpha vectors of the best solution
    std::map<int, Vector> min_alphas;

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
    
    /** after adding new examples, shall the previous alpha solution be used as an initial guess?*/
    bool usePreviousAlphas;

  public:

    // ------ constructors and destructors ------
    /** simple constructor */
    GPLikelihoodApprox( const std::map<int, Vector> & binaryLabels, 
                        ImplicitKernelMatrix *ikm,
                        IterativeLinearSolver *linsolver,
                        EigValues *eig,
                        bool verifyApproximation = false,
                        int _nrOfEigenvaluesToConsider = 1
                      );
      
    /** simple destructor */
    virtual ~GPLikelihoodApprox();
     
    // ------ main methods ------
    /**
    * @brief Evaluate the likelihood for given hyperparameters
    *
    * @param x vector with specified hyperparameters to evaluate their likelihood
    *
    * @return likelihood 
    */
    virtual double evaluate(const OPTIMIZATION::matrix_type & x);
     
    
    // ------ get and set methods ------
    const Vector & getBestParameters () const { return min_parameter; };
    const std::map<int, Vector> & getBestAlphas () const { return min_alphas; };
    
    void setParameterLowerBound(const double & _parameterLowerBound);
    void setParameterUpperBound(const double & _parameterUpperBound);
    
    void setLastAlphas(std::map<int, NICE::Vector> * _lastAlphas);
    void setBinaryLabels(const std::map<int, Vector> & _binaryLabels);
    
    void setUsePreviousAlphas( const bool & _usePreviousAlphas );
    void setVerbose( const bool & _verbose );
    void setDebug( const bool & _debug );
};

}

#endif
