/** 
* @file GeneralizedIntersectionKernelFunction.h
* @brief The generalized intersection kernel function as distance measure between two histograms interpreted as vectors (Interface)
* @author Alexander Freytag
* @date 23-12-2011 (dd-mm-yyyy)
*/
#ifndef _NICE_GENERALIZEDINTERSECTIONKERNELFUNCTION
#define _NICE_GENERALIZEDINTERSECTIONKERNELFUNCTION

#include <iostream>

#include <core/vector/MatrixT.h>

#include <gp-hik-core/FeatureMatrixT.h>


namespace NICE {

  /** 
 * @class GeneralizedIntersectionKernelFunction
 * @brief The generalized intersection kernel function as distance measure between two histograms interpreted as vectors
 * @author Alexander Freytag
 */

template<class T> class GeneralizedIntersectionKernelFunction
{

    protected:
    //TODO one could also use a separate function here, such as pow(,a) - this would be much more generalized but only power the inputs.
      double exponent;

    public:
  
    /** 
    * @brief Default constructor
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    GeneralizedIntersectionKernelFunction();
    
    /** 
    * @brief Recommended constructor
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    GeneralizedIntersectionKernelFunction(const double & _exponent);
      
    /** 
    * @brief Default desctructor
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    ~GeneralizedIntersectionKernelFunction();
    
    /** 
    * @brief Set exponent to specified value
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    void set_exponent(const double & _exponent);
    
    /** 
    * @brief Return currently used exponent
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    double get_exponent();

    /** 
    * @brief Measures the distance between tow vectors using the generalized histogram intersection distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    double measureDistance ( const std::vector<T> & a, const std::vector<T> & b  );
  
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the g-HIK distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X  );
    
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the g-HIK distance and add a given amount of noise on the main diagonal
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X , const double & noise);
    
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the g-HIK distance and add a given amount of noise on the main diagonal
    * @author Alexander Freytag
    * @date 03-02-2012 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const NICE::FeatureMatrixT<T>  & X , const double & noise);
    
    /** 
    * @brief Computes the similarity between the data and a new vector using the g-HIK distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    std::vector<double> computeKernelVector ( const std::vector<std::vector<T> > & X , const std::vector<T> & xstar);
    
    /** 
    * @brief Simply print the name of the class
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    void sayYourName();

};

  //! default definition for a GeneralizedIntersectionKernelFunction
  typedef GeneralizedIntersectionKernelFunction<double> GeneralizedIntersectionKernelFunctionDouble;

}

#ifdef __GNUC__
#include "gp-hik-core/kernels/GeneralizedIntersectionKernelFunction.tcc"
#endif

#endif
