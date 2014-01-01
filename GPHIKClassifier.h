/** 
* @file GPHIKClassifier.h
* @author Erik Rodner, Alexander Freytag
* @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning) (Interface)
* @date 02/01/2012

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
#include "FMKGPHyperparameterOptimization.h"
#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"

namespace NICE {
  
 /** 
 * @class GPHIKClassifier
 * @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning)
 * @author Erik Rodner, Alexander Freytag
 */
 
class GPHIKClassifier : NICE::Persistent
{

  protected:
    std::string confSection;
    double noise;

    enum VarianceApproximation{
      APPROXIMATE_ROUGH,
      APPROXIMATE_FINE,
      EXACT,
      NONE
    };
    
    VarianceApproximation varianceApproximation;
    
    /**compute the uncertainty prediction during classification?*/
    bool uncertaintyPredictionForClassification;
    
    NICE::Config *confCopy;
    NICE::ParameterizedFunction *pf;
    NICE::FMKGPHyperparameterOptimization *gphyper;
    
    /** verbose flag for useful output*/
    bool verbose;
    /** debug flag for several outputs useful for debugging*/
    bool debug;
    
    /** 
    * @brief classify a given example with the previously learnt model
    * @param pe example to be classified given in a sparse representation
    */    
    void init(const NICE::Config *conf, const std::string & s_confSection);
       

  public:

    /** simple constructor */
    GPHIKClassifier( const NICE::Config *conf = NULL, const std::string & s_confSection = "GPHIKClassifier" );
      
    /** simple destructor */
    ~GPHIKClassifier();
   

    /** 
     * @brief classify a given example with the previously learnt model
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) to be classified given in a sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     */        
    void classify ( const NICE::SparseVector * example,  int & result, NICE::SparseVector & scores );
    
    /** 
     * @brief classify a given example with the previously learnt model
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (SparseVector) to be classified given in a sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     * @param uncertainty (double*) predictive variance of the classification result, if computed
     */    
    void classify ( const NICE::SparseVector * example,  int & result, NICE::SparseVector & scores, double & uncertainty );
    
    /** 
     * @brief classify a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times* 
     * @date 18-06-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) to be classified given in a non-sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     */        
    void classify ( const NICE::Vector * example,  int & result, NICE::SparseVector & scores );
    
    /** 
     * @brief classify a given example with the previously learnt model
     * NOTE: whenever possible, you should the sparse version to obtain significantly smaller computation times
     * @date 18-06-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param example (non-sparse Vector) to be classified given in a non-sparse representation
     * @param result (int) class number of most likely class
     * @param scores (SparseVector) classification scores for known classes
     * @param uncertainty (double) predictive variance of the classification result, if computed
     */    
    void classify ( const NICE::Vector * example,  int & result, NICE::SparseVector & scores, double & uncertainty );    

    /**
     * @brief train this classifier using a given set of examples and a given set of binary label vectors 
     * @date 18-10-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples (std::vector< NICE::SparseVector *>) training data given in a sparse representation
     * @param labels (Vector) class labels (multi-class)
     */
    void train ( const std::vector< NICE::SparseVector *> & examples, const NICE::Vector & labels );
    
    /** 
     * @brief train this classifier using a given set of examples and a given set of binary label vectors 
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples examples to use given in a sparse data structure
     * @param binLabels corresponding binary labels with class no. There is no need here that every examples has only on positive entry in this set (1,-1)
     */
    void train ( const std::vector< NICE::SparseVector *> & examples, std::map<int, NICE::Vector> & binLabels );
    
    /** Persistent interface */
    void restore ( std::istream & is, int format = 0 );
    void store ( std::ostream & os, int format = 0 ) const;
    void clear ();

    GPHIKClassifier *clone () const;

    /** 
     * @brief prediction of classification uncertainty
     * @date 19-06-2012 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the classification uncertainty shall be predicted, given in a sparse representation
     * @param uncertainty contains the resulting classification uncertainty
     */       
    void predictUncertainty( const NICE::SparseVector * example, double & uncertainty );
    
    /** 
     * @brief prediction of classification uncertainty
     * @date 19-12-2013 (dd-mm-yyyy)
     * @author Alexander Freytag
     * @param examples example for which the classification uncertainty shall be predicted, given in a non-sparse representation
     * @param uncertainty contains the resulting classification uncertainty
     */       
    void predictUncertainty( const NICE::Vector * example, double & uncertainty );    
    
    std::set<int> getKnownClassNumbers ( ) const;

};

}

#endif
