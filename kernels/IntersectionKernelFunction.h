/** 
* @file IntersectionKernelFunction.h
* @brief The intersection kernel function as distance measure between two histograms interpreted as vectors (Interface)
* @author Alexander Freytag
* @date 08-12-2011 (dd-mm-yyyy)
*/
#ifndef _NICE_INTERSECTIONKERNELFUNCTION
#define _NICE_INTERSECTIONKERNELFUNCTION

#include <iostream>

#include <gp-hik-core/FeatureMatrixT.h>
#include <core/vector/SparseVectorT.h>


//TODO functions are not allowed to be virtual anymore due to the template usage - any idea how to treat this?
//maybe using type erasure: http://www.artima.com/cppsource/type_erasure2.html ?


namespace NICE {
  
 /** 
 * @class IntersectionKernelFunction
 * @brief The intersection kernel function as distance measure between two histograms interpreted as vectors
 * @author Alexander Freytag
 */
 
template<class T> class IntersectionKernelFunction 
{

    protected:

    public:
      
    /** 
    * @brief Default constructor
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    IntersectionKernelFunction();
      
    /** 
    * @brief Default desctructor
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    ~IntersectionKernelFunction();

    /** 
    * @brief Measures the distance between tow vectors using the histogram intersection distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    double measureDistance ( const std::vector<T> & a, const std::vector<T> & b  );
    
    double measureDistance ( const NICE::SparseVector & a, const NICE::SparseVector & b  );
    
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the HIK distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X  );
    
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the HIK distance and add a given amount of noise on the main diagonal
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X , const double & noise);
    
    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the HIK distance and add a given amount of noise on the main diagonal
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const std::vector<NICE::SparseVector > & X , const double & noise);

    /** 
    * @brief Computes the symmetric and positive semi-definite kernel matrix K based on the given examples using the HIK distance and add a given amount of noise on the main diagonal
    * @author Alexander Freytag
    * @date 03-02-2012 (dd-mm-yyyy)
    */
    NICE::Matrix computeKernelMatrix ( const NICE::FeatureMatrixT<T> & X , const double & noise);
    
    /** 
    * @brief Computes the similarity between the data and a new vector using the HIK distance
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    std::vector<double> computeKernelVector ( const std::vector<std::vector<T> > & X , const std::vector<T> & xstar);
    
    NICE::Vector computeKernelVector ( const std::vector<NICE::SparseVector> & X , const NICE::SparseVector & xstar);
    
    /** 
    * @brief Simply print the name of the class
    * @author Alexander Freytag
    * @date 23-12-2011 (dd-mm-yyyy)
    */
    void sayYourName();

};

  //! default definition for a IntersectionKernelFunction
  typedef IntersectionKernelFunction<double> IntersectionKernelFunctionDouble;

}

#ifdef __GNUC__
#include "gp-hik-core/kernels/IntersectionKernelFunction.tcc"
#endif

#endif
