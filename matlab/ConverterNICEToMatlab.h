#ifdef NICE_USELIB_MEX
/** 
* @file ConverterNICEToMatlab.h
* @author Alexander Freytag
* @brief Several methods for converting NICE containers into Matlab data (Interface)
* @date 15-01-2014 ( dd-mm-yyyy)
*/
#ifndef _NICE_CONVERTERNICETOMATLABINCLUDE
#define _NICE_CONVERTERNICETOMATLABINCLUDE

// STL includes
#include "mex.h"

// NICE-core includes
#include <core/vector/MatrixT.h>
#include <core/vector/SparseVectorT.h>
#include <core/vector/VectorT.h>

namespace NICE {

/**
* @author Alexander Freytag, Johannes Ruehle
* @brief Several methods for converting Matlab data into NICE containers
*/
namespace MatlabConversion {

    /**
     * @brief Convert a SparseVector into a Matlab 1xD sparse matrix
     * @author Alexander Freytag
     * @date 15-01-2014 ( dd-mm-yyyy)
     *
     * @param niceSvec a NIC::SparseVector
     * @param b_adaptIndexCtoM if true, dim k will be inserted as k, not as k+1 (which would be the default for C->M) Defaults to false.
     * @return mxArray*
     **/
    mxArray* convertSparseVectorFromNice( const NICE::SparseVector & niceSvec, const bool & b_adaptIndexCtoM = false );

    /**
     * @brief Convert a NICE::Matrix into a full Matlab MxD matrix
     * @author Alexander Freytag
     * @date 15-01-2014 ( dd-mm-yyyy)
     *
     * @param niceMatrix a NICE::Matrix
     * @return mxArray*
     **/
    mxArray* convertMatrixFromNice( const NICE::Matrix & niceMatrix );
    
    /**
     * @brief Convert a NICE::Vector into a full Matlab 1xD matrix
     * @author Alexander Freytag
     * @date 15-01-2014 ( dd-mm-yyyy)
     *
     * @param niceVector a NICE::Vector
     * @return mxArray*
     **/
    
    mxArray* convertVectorFromNice( const NICE::Vector & niceVector );

} // ns MatlabConversion

}

#endif
#endif
