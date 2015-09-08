/** 
* @file GPHIKClassifier.h
* @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning) (Interface)
* @author Alexander Freytag, Erik Rodner
* @date 01-02-2012 (dd-mm-yyyy)
*/
#ifndef _NICE_GPHIKCLASSIFIERINCLUDE
#define _NICE_GPHIKCLASSIFIERINCLUDE

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


namespace NICE {
  
 /** 
 * @class GPHIKClassifier
 * @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning)
 * @author Alexander Freytag, Erik Rodner
 */
 
class GPHIKClassifier : public NICE::Persistent, public NICE::OnlineLearnable
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
    
    /** verbose flag for useful output*/
    bool b_verbose;
    /** debug flag for several outputs useful for debugging*/
    bool b_debug;
    
    //////////////////////////////////////
    //      general specifications      //
    //////////////////////////////////////
    
    /** Header in configfile where variable settings are stored */
    std::string confSection;    
    
    //////////////////////////////////////
    // classification related variables //
    //////////////////////////////////////    
    /** memorize whether the classifier was already trained*/
    bool b_isTrained;
    
    
    /** Main object doing all the jobs: training, classification, optimization, ... */
    NICE::FMKGPHyperparameterOptimization *gphyper;    
    
    
    /** Gaussian label noise for model regularization */
    double d_noise;

    enum VarianceApproximation{
      APPROXIMATE_ROUGH,
      APPROXIMATE_FINE,
      EXACT,
      NONE
    };
    
    /** Which technique for variance approximations shall be used */
    VarianceApproximation varianceApproximation;
    
    /**compute the uncertainty prediction during classification?*/
    bool uncertaintyPredictionForClassification;
    
    /////////////////////////
    /////////////////////////
    //  PROTECTED METHODS  //
    /////////////////////////
    /////////////////////////
          

  public:

    /** 
     * @brief default constructor
     * @author Alexander Freytag
     * @date 05-02-2014 ( dd-mm-yyyy)
     */
    GPHIKClassifier( );
     
    
    /** 
     * @brief standard constructor
     * @author Alexander Freytag
     */
    GPHIKClassifier( const NICE::Config *_conf , 
                     const std::string & s_confSection = "GPHIKClassifier" 
                   );
      
    /**
     * @brief simple destructor
     * @author Alexander Freytag
     */
    ~GPHIKClassifier();
    
    /** 
    * @brief Setup internal variables and objects used
    * @author Alexander Freytag
    * @param conf Config file to specify variable settings
    * @param s_confSection
    */    
    void initFromConfig(const NICE::Config *_conf, 
                        const std::string & s_confSection
                       );    
    
    ///////////////////// ///////////////////// /////////////////////
    //                         GET / SET
    ///////////////////// ///////////////////// /////////////////////      
    
    /**
     * @brief Return currently known class numbers
     * @author Alexander Freytag
     */    
    std::set<uint> getKnownClassNumbers ( ) const;    
   
    ///////////////////// ///////////////////// /////////////////////
    //                      CLASSIFIER STUFF
    ///////////////////// ///////////////////// /////////////////////      
    
    /** 
     * @brief classify a given example with the previously learnt model
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) to be classified given in a sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     */        
    void classify ( const NICE::SparseVector * _example, 
                    uint & _result, 
                    NICE::SparseVector & _scores 
                  ) const;
    
    /** 
     * @brief classify a given example with the previously learnt model
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) to be classified given in a sparse representation
     * @param result (uint) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     * @param uncertainty (double*) predictive variance of the classification result, if computed
     */    
    void classify ( const NICE::SparseVector * _example,  
                    uint & _result, 
                    NICE::SparseVector & _scores, 
                    double & _uncertainty 
                  ) const;
    
    /** 
     * @brief classify a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times* 
     * @date 18-06-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) to be classified given in a non-sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     */        
    void classify ( const NICE::Vector * _example,  
                    uint & _result, 
                    NICE::SparseVector & _scores 
                  ) const;
    
    /** 
     * @brief classify a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times
     * @date 18-06-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) to be classified given in a non-sparse representation
     * @param result (uint) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     * @param uncertainty (double) predictive variance of the classification result, if computed
     */    
    void classify ( const NICE::Vector * _example,  
                    uint & _result, 
                    NICE::SparseVector & _scores, 
                    double & _uncertainty 
                  ) const;    

    /**
     * @brief train this classifier using a given set of examples and a given set of binary label vectors 
     * @date 18-10-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples (std::vector< NICE::SparseVector *>) training data given in a sparse representation
     * @param labels (Vector) class labels (multi-class)
     */
    void train ( const std::vector< const NICE::SparseVector *> & _examples, 
                 const NICE::Vector & _labels 
               );
    
    /** 
     * @brief train this classifier using a given set of examples and a given set of binary label vectors 
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples examples to use given in a sparse data structure
     * @param binLabels corresponding binary labels with class no. There is no need here that every examples has only on positive entry in this set (1,-1)
     */
    void train ( const std::vector< const NICE::SparseVector *> & _examples, 
                 std::map<uint, NICE::Vector> & _binLabels 
               );
    
    /**
     * @brief Clone classifier object
     * @author Alexander Freytag
     */    
    GPHIKClassifier *clone () const;

    /** 
     * @brief prediction of classification uncertainty
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the classification uncertainty shall be predicted, given in a sparse representation
     * @param uncertainty contains the resulting classification uncertainty
     */       
    void predictUncertainty( const NICE::SparseVector * _example, 
                             double & _uncertainty 
                           ) const;
    
    /** 
     * @brief prediction of classification uncertainty
     * @date 19-12-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the classification uncertainty shall be predicted, given in a non-sparse representation
     * @param uncertainty contains the resulting classification uncertainty
     */       
    void predictUncertainty( const NICE::Vector * _example, 
                             double & _uncertainty 
                           ) const;    
    


    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////   
    
    /** 
     * @brief Load classifier from external file (stream)
     * @author Alexander Freytag
     */     
    void restore ( std::istream & _is, 
                   int _format = 0 
                 );
    
    /** 
     * @brief Save classifier to external file (stream)
     * @author Alexander Freytag
     */     
    void store ( std::ostream & _os, 
                 int _format = 0 
               ) const;
    
    /** 
     * @brief Clear classifier object
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
    virtual void addExample( const NICE::SparseVector * _example, 
                             const double & _label, 
                             const bool & _performOptimizationAfterIncrement = true
                            );
                          
    /** 
     * @brief add several new examples
     * @author Alexander Freytag
     */    
    virtual void addMultipleExamples( const std::vector< const NICE::SparseVector * > & _newExamples,
                                      const NICE::Vector & _newLabels,
                                      const bool & _performOptimizationAfterIncrement = true
                                    );       
};

}

#endif
