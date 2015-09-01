/** 
* @file ParameterizedFunction.h
* @brief Simple parameterized multi-dimensional function (Interface)
* @author Erik Rodner, Alexander Freytag
* @date 01/04/2012
*/
#ifndef _NICE_PARAMETERIZEDFUNCTIONINCLUDE
#define _NICE_PARAMETERIZEDFUNCTIONINCLUDE

// STL includes
#include <vector>
#include <limits>

// NICE-core includes
#include <core/basics/Persistent.h>
// 
#include <core/vector/VectorT.h>
#include <core/vector/SparseVectorT.h>

namespace NICE {
  
/** @class ParameterizedFunction
 * @brief
 * simple parameterized multi-dimensional function 
 * 
 * @description
 * current requirements: 
 * (1) f(0) = 0
 * (2) f is monotonically increasing
 *
 * @author Erik Rodner, Alexander Freytag
 */
class ParameterizedFunction : public NICE::Persistent
{

  protected:

    /** parameters of the function */
    NICE::Vector m_parameters;


  public:

    /**
    * @brief contructor taking the dimension of the parameter vector, initializes
    * the member variable
    *
    * @param dimension dimension of the parameter vector
    */
    ParameterizedFunction ( uint dimension );

    /** empty destructor */
    virtual ~ParameterizedFunction () {};

    /**
    * @brief Function evaluation
    *
    * @param index component of the vector
    * @param x function argument
    *
    * @return function value, which depends on the stored parameters
    */
    virtual double f ( uint index, double x ) const = 0;
    
    /**
    * @brief Tell whether this function is order-preserving in the sense that
    * a permutation of function values is invariant with respect to the parameter value.
    * Therefore, the function is either monotonically increasing or decreasing.
    *
    * @return boolean flag =true iff the function is order preserving
    */
    virtual bool isOrderPreserving () const = 0;

    /**
    * @brief get the lower bound for each parameter
    *
    * @return vector with lower bounds
    */
    virtual NICE::Vector getParameterLowerBounds() const = 0;
    
    /**
    * @brief set the lower bounds for each parameter
    *
    * @return vector with upper bounds
    */
    virtual void setParameterLowerBounds(const NICE::Vector & _newLowerBounds) = 0;


    /**
    * @brief get the upper bounds for each parameter
    *
    * @return vector with upper bounds
    */
    virtual NICE::Vector getParameterUpperBounds() const = 0;
    
    /**
    * @brief set the upper bounds for each parameter
    *
    * @return vector with upper bounds
    */
    virtual void setParameterUpperBounds(const NICE::Vector & _newUpperBounds) = 0;

    /**
    * @brief read-only access of the parameters
    *
    * @return const-reference to the stored parameter vector
    */
    const NICE::Vector & parameters() const { return m_parameters; }

    /**
    * @brief read-and-write access to the parameters
    *
    * @return reference to the stored parameter vector
    */
    Vector & parameters() { return m_parameters; }
   
    /**
    * @brief apply function to a data matrix ( feature vectors stored in rows )
    *
    * @param dataMatrix input matrix, e.g. featureMatrix[0][1..d] is the d-dimensional feature vector of example 0
    */
    void applyFunctionToDataMatrix ( std::vector< std::vector< double > > & dataMatrix ) const;
    
    /** Persistent interface */
    virtual void restore ( std::istream & is, int format = 0 );
    virtual void store ( std::ostream & os, int format = 0 ) const;
    virtual void clear () {};
    
    virtual std::string sayYourName() const = 0;
};

}

#endif