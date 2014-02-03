/** 
* @file GPHIKRegression.h
* @brief Main interface for our GP HIK regression implementation (Interface)
* @author Alexander Freytag
* @date 15-01-2014 (dd-mm-yyyy)
*/
#ifndef _NICE_GPHIKREGRESSIONINCLUDE
#define _NICE_GPHIKREGRESSIONINCLUDE

// STL includes
#include <string>
#include <limits>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Persistent.h>
// 
#include <core/vector/SparseVectorT.h>

// gp-hik-core includes
#include "gp-hik-core/FMKGPHyperparameterOptimization.h"
#include "gp-hik-core/OnlineLearnable.h"
#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class GPHIKRegression
 * @brief Main interface for our GP HIK regression implementation (Interface)
 * @author Alexander Freytag
 */
 
class GPHIKRegression : public NICE::Persistent, public NICE::OnlineLearnable
{

  protected:
    
    /////////////////////////
    /////////////////////////
    // PROTECTED VARIABLES //
    /////////////////////////
    /////////////////////////
    
    // output/debug related settings
    
    /** verbose flag for useful output*/
    bool verbose;
    /** debug flag for several outputs useful for debugging*/
    bool debug;
    
    // general specifications
    
    /** Header in configfile where variable settings are stored */
    std::string confSection;
    /** Configuration file specifying variable settings */
    NICE::Config *confCopy; 
    
    // internal objects 
    
    /** Main object doing all the jobs: training, regression, optimization, ... */
    NICE::FMKGPHyperparameterOptimization *gphyper;    
    
    /** Possibility for transforming feature values, parameters can be optimized */
    NICE::ParameterizedFunction *pf;    
    
    
    
    
    /** Gaussian label noise for model regularization */
    double noise;

    enum VarianceApproximation{
      APPROXIMATE_ROUGH,
      APPROXIMATE_FINE,
      EXACT,
      NONE
    };
    
    /** Which technique for variance approximations shall be used */
    VarianceApproximation varianceApproximation;
    
    /**compute the uncertainty prediction during regression?*/
    bool uncertaintyPredictionForRegression;
    
    /////////////////////////
    /////////////////////////
    //  PROTECTED METHODS  //
    /////////////////////////
    /////////////////////////
    
    /** 
    * @brief Setup internal variables and objects used
    * @author Alexander Freytag
    * @param conf Config file to specify variable settings
    * @param s_confSection
    */    
    void init(const NICE::Config *conf, const std::string & s_confSection);
       

  public:

    /** 
     * @brief standard constructor
     * @author Alexander Freytag
     */
    GPHIKRegression( const NICE::Config *conf = NULL, const std::string & s_confSection = "GPHIKRegression" );
      
    /**
     * @brief simple destructor
     * @author Alexander Freytag
     */
    ~GPHIKRegression();
    
    ///////////////////// ///////////////////// /////////////////////
    //                         GET / SET
    ///////////////////// ///////////////////// /////////////////////      
    
   
   
    ///////////////////// ///////////////////// /////////////////////
    //                      REGRESSION STUFF
    ///////////////////// ///////////////////// /////////////////////      
    
    /** 
     * @brief Estimate output of a given example with the previously learnt model
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) for which regression shall be performed, given in a sparse representation
     * @param result (double) regression result
     */        
    void estimate ( const NICE::SparseVector * example,  double & result ) const;
    
    /** 
     * @brief Estimate output of a given example with the previously learnt model
     ** @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) for which regression shall be performed, given in a sparse representation
     * @param result (double) regression result
     * @param uncertainty (double*) predictive variance of the regression result, if computed
     */    
    void estimate ( const NICE::SparseVector * example,  double & result, double & uncertainty ) const;
    
    /** 
     * @brief Estimate output of a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times* 
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) for which regression shall be performed, given in a non-sparse representation
     * @param result (double) regression result
     */        
    void estimate ( const NICE::Vector * example,  double & result ) const;
    
    /** 
     * @brief Estimate output of a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) for which regression shall be performed, given in a non-sparse representation
     * @param result (double)regression result
     * @param uncertainty (double*) predictive variance of the regression result, if computed
     */    
    void estimate ( const NICE::Vector * example,  double & result, double & uncertainty ) const;    

    /**
     * @brief train this regression method using a given set of examples and corresponding labels
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples (std::vector< NICE::SparseVector *>) training data given in a sparse representation
     * @param labels (Vector) labels
     */
    void train ( const std::vector< const NICE::SparseVector *> & examples, const NICE::Vector & labels );
    
    
    /**
     * @brief Clone regression object
     * @author Alexander Freytag
     */    
    GPHIKRegression *clone () const;

    /** 
     * @brief prediction of regression uncertainty
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the regression uncertainty shall be predicted, given in a sparse representation
     * @param uncertainty contains the resulting regression uncertainty
     */       
    void predictUncertainty( const NICE::SparseVector * example, double & uncertainty ) const;
    
    /** 
     * @brief prediction of regression uncertainty
     * @date 15-01-2014 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the regression uncertainty shall be predicted, given in a non-sparse representation
     * @param uncertainty contains the resulting regression uncertainty
     */       
    void predictUncertainty( const NICE::Vector * example, double & uncertainty ) const;    
    


    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////   
    
    /** 
     * @brief Load regression object from external file (stream)
     * @author Alexander Freytag
     */     
    void restore ( std::istream & is, int format = 0 );
    
    /** 
     * @brief Save regression object to external file (stream)
     * @author Alexander Freytag
     */     
    void store ( std::ostream & os, int format = 0 ) const;
    
    /** 
     * @brief Clear regression object
     * @author Alexander Freytag
     */     
    void clear ();
    
    
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
