#ifdef NICE_USELIB_MEX

#include "ConverterNICEToMatlab.h"

using namespace NICE;
using namespace NICE::MatlabConversion;

// b_adaptIndexCtoM: if true, dim k will be inserted as k, not as k+1 (which would be the default for C->M)
mxArray* MatlabConversion::convertSparseVectorFromNice( const NICE::SparseVector & niceSvec, const bool & b_adaptIndexCtoM )
{
    mxArray * matlabSparseVec;
    
    if ( b_adaptIndexCtoM ) 
       matlabSparseVec = mxCreateSparse( niceSvec.getDim() -1 /*m*/, 1/*n*/, niceSvec.size() -1 /*nzmax*/, mxREAL);
    else
      matlabSparseVec = mxCreateSparse( niceSvec.getDim() /*m*/, 1/*n*/, niceSvec.size() /*nzmax*/, mxREAL);
    
    
    // To make the returned sparse mxArray useful, you must initialize the pr, ir, jc, and (if it exists) pi arrays.    
    // mxCreateSparse allocates space for:
    // 
    // A pr array of length nzmax.
    // A pi array of length nzmax, but only if ComplexFlag is mxCOMPLEX in C (1 in Fortran).
    // An ir array of length nzmax.
    // A jc array of length n+1.  
  
    double* prPtr = mxGetPr(matlabSparseVec);
    mwIndex * ir = mxGetIr( matlabSparseVec );
    
    mwIndex * jc = mxGetJc( matlabSparseVec );
    jc[1] = niceSvec.size(); jc[0] = 0; 
    
    
    mwSize cnt = 0;
        
    for ( NICE::SparseVector::const_iterator myIt = niceSvec.begin(); myIt != niceSvec.end(); myIt++, cnt++ )
    {
        // set index
        if ( b_adaptIndexCtoM ) 
            ir[cnt] = myIt->first-1;
        else
            ir[cnt] = myIt->first;
        
        // set value
        prPtr[cnt] = myIt->second;
    }
    
    return matlabSparseVec;
}

mxArray* MatlabConversion::convertMatrixFromNice( const NICE::Matrix & niceMatrix )
{
  mxArray *matlabMatrix = mxCreateDoubleMatrix( niceMatrix.rows(), niceMatrix.cols(), mxREAL );
  double* matlabMatrixPtr = mxGetPr( matlabMatrix );

  for( int i = 0; i < niceMatrix.rows(); i++ )
  {
    for( int j = 0; j < niceMatrix.cols(); j++ )
    {
      matlabMatrixPtr[i + j*niceMatrix.rows()] = niceMatrix(i,j);
    }
  }
  
  return matlabMatrix;
}

mxArray* MatlabConversion::convertVectorFromNice( const NICE::Vector & niceVector )
{
  mxArray *matlabVector = mxCreateDoubleMatrix( niceVector.size(), 1, mxREAL );
  double* matlabVectorPtr = mxGetPr( matlabVector );

  for( int i = 0; i < niceVector.size(); i++ )
  {
    matlabVectorPtr[i] = niceVector[i];
  }
  return matlabVector;
}
#endif
