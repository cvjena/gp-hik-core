/** 
* @file GenericKernelFunction.h
* @author Alexander Freytag
* @brief Abstract class for all kernels (Interface - abstract)
* @date 12/08/2011
*/
#ifndef _NICE_GENERICKERNELFUNCTION
#define _NICE_GENERICKERNELFUNCTION

#include <vector>
#include <core/vector/VectorT.h>

namespace NICE {
  
 /** 
 * @class GenericKernelFunction
 * @brief Abstract class for all kernels
 * @author Alexander Freytag
 */

template<class T> class GenericKernelFunction 
{

    protected:

    public:
  
    /** simple constructor */
    GenericKernelFunction(){};
      
    /** simple destructor */
    ~GenericKernelFunction(){};


    virtual double measureDistance ( const std::vector<T> & a, const std::vector<T> & b  )=0;
    virtual NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X )=0;
    virtual NICE::Matrix computeKernelMatrix ( const std::vector<std::vector<T> > & X , const double & noise)=0;
    virtual std::vector<double> computeKernelVector ( const std::vector<std::vector<T> > & X , const NICE::Vector & xstar)=0;
    virtual void sayYourName()=0;

};

}

#endif
