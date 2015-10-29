#ifdef NICE_USELIB_MEX
/** 
* @file ConverterMatlabToNICE.h
* @author Alexander Freytag
* @brief Several methods for converting Matlab data into NICE containers (Interface)
* @date 15-01-2014 ( dd-mm-yyyy)

*/
#ifndef _NICE_CONVERTERMATLABTONICEINCLUDE
#define _NICE_CONVERTERMATLABTONICEINCLUDE

// STL includes
#include "mex.h"

// NICE-core includes
#include <core/vector/MatrixT.h>
#include <core/vector/SparseVectorT.h>
#include <core/vector/VectorT.h>

namespace NICE {

 namespace MatlabConversion {

     /**
     * @author Alexander Freytag, Johannes Ruehle
     * @brief Several methods for converting Matlab data into NICE containers
     */

    /**
     * @brief Convert a sparse matlab matrix into an std::vector of NICE::SparseVectors *
     * @TODO could be also converted into VVector!
     * 
     * @param array_ptr Sparse MxD Matlab matrix
     * @return std::vector< NICE::SparseVector * >
     **/  
    std::vector< const NICE::SparseVector * > convertSparseMatrixToNice( const mxArray *array_ptr );

    /**
     * @brief Convert a sparse 1xD Matlab matrix into a SparseVector
     *
     * @param array_ptr Sparse 1xD Matlab matrix
     * @param b_adaptIndexMtoC if true, dim k will be inserted as k, not as k-1 (which would be the default for  M->C). Defaults to false.
     * @return NICE::SparseVector
     **/
    NICE::SparseVector convertSparseVectorToNice( const mxArray* array_ptr,  const bool & b_adaptIndexMtoC = false );

    /**
     * @brief Convert a MxD Matlab matrix into a NICE::Matrix
     *
     * @param matlabMatrix a matlab MxD matrix
     * @return NICE::Matrix
     **/
    NICE::Matrix convertDoubleMatrixToNice( const mxArray* matlabMatrix );
    
    /**
     * @brief Convert a 1xD Matlab matrix into a NICE::Vector
     *
     * @param matlabMatrix a matlab 1xD matrix
     * @return  NICE::Vector
     **/
    NICE::Vector convertDoubleVectorToNice( const mxArray* matlabMatrix );

    /**
     * @brief Convert a Matlab char array into an std::string
     *
     * @param matlabString a matlab char array variable
     * @return std::string
     **/
    std::string convertMatlabToString( const mxArray *matlabString );

    /**
     * @brief Convert a Matlab int32 scalar variable into an std::int
     *
     * @param matlabInt32 a matlab int32 variable
     * @return int
     **/
    int convertMatlabToInt32( const mxArray *matlabInt32 );
    
    /**
     * @brief Convert a Matlab double variable into an std::double
     *
     * @param matlabDouble a matlab double variable
     * @return double
     **/
    double convertMatlabToDouble( const mxArray *matlabDouble );
    
    /**
     * @brief Convert a Matlab bool variable into an std::bool
     *
     * @param matlabBool a matlab bool variable
     * @return bool
     **/    
    bool convertMatlabToBool( const mxArray *matlabBool );

    /**
     * @brief Checks whether a given sparse data structure is a matrix (or a vector instead)
     *
     * @param array_ptr Sparse MxD Matlab matrix
     * @return bool. false of either M or D equals to 1
     **/
    bool isSparseDataAMatrix( const mxArray *array_ptr );

} //ns MatlabConversion

}

#endif
#endif
