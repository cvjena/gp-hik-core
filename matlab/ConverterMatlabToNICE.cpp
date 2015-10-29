#ifdef NICE_USELIB_MEX

#include "ConverterMatlabToNICE.h"

using namespace NICE;

/* Pass analyze_sparse a pointer to a sparse mxArray.  A sparse mxArray
   only stores its nonzero elements.  The values of the nonzero elements 
   are stored in the pr and pi arrays.  The tricky part of analyzing
   sparse mxArray's is figuring out the indices where the nonzero
   elements are stored.  (See the mxSetIr and mxSetJc reference pages
   for details. */  
std::vector< const NICE::SparseVector * > MatlabConversion::convertSparseMatrixToNice( const mxArray *array_ptr )
{
  double   *pr;//, *pi;
  mwIndex  *ir, *jc;
  mwSize   col, total=0;
  mwIndex  starting_row_index, stopping_row_index, current_row_index;
  mwSize   i_numExamples, i_numDim;
  
  /* Get the starting positions of all four data arrays. */ 
  pr = mxGetPr( array_ptr );
  // no complex data supported here
  // pi = mxGetPi(array_ptr);
  ir = mxGetIr( array_ptr );
  jc = mxGetJc( array_ptr );
  
  // dimenions of the matrix -> feature dimension and number of examples
  i_numExamples = mxGetM( array_ptr );  
  i_numDim = mxGetN( array_ptr );
    
  // initialize output variable -- don't use const pointers here since the content of the vectors will change 
  // in the following loop. We reinterprete the vector lateron into a const version
  std::vector< NICE::SparseVector * > sparseMatrix;
  sparseMatrix.resize ( i_numExamples );
    
  for ( std::vector< NICE::SparseVector * >::iterator matIt = sparseMatrix.begin(); 
        matIt != sparseMatrix.end(); matIt++)
  {
      *matIt = new NICE::SparseVector( i_numDim );
  }  
  
  // now copy the data
  for ( col = 0; col < i_numDim; col++ )
  { 
    starting_row_index = jc[col]; 
    stopping_row_index = jc[col+1]; 
    
    // empty column?
    if (starting_row_index == stopping_row_index)
      continue;
    else
    {
      for ( current_row_index = starting_row_index; 
            current_row_index < stopping_row_index; 
            current_row_index++
          )
      {
          //note: no complex data supported her
          sparseMatrix[ ir[current_row_index] ]->insert( std::pair<uint, double>( col, pr[total++] ) );
      } // for-loop
      
    }
  } // for-loop over columns
  
  //NOTE
  // Compiler doesn't know how to automatically convert
  // std::vector<T*> to std::vector<T const*> because the way
  // the template system works means that in theory the two may
  // be specialised differently.  This is an explicit conversion.
  return reinterpret_cast< std::vector< const NICE::SparseVector *> &>( sparseMatrix );
}

// b_adaptIndexMtoC: if true, dim k will be inserted as k, not as k-1 (which would be the default for  M->C)
NICE::SparseVector MatlabConversion::convertSparseVectorToNice(
               const mxArray* array_ptr,
               const bool & b_adaptIndexMtoC
    )
{
  double   *pr, *pi;
  mwIndex  *ir, *jc;
  mwSize   col, total=0;
  mwIndex  starting_row_index, stopping_row_index, current_row_index;
  mwSize   dimy, dimx;
  
  /* Get the starting positions of all four data arrays. */ 
  pr = mxGetPr( array_ptr );
  pi = mxGetPi( array_ptr );
  ir = mxGetIr( array_ptr );
  jc = mxGetJc( array_ptr );
  
  // dimenions of the matrix -> feature dimension and number of examples
  dimy = mxGetM( array_ptr );  
  dimx = mxGetN( array_ptr );
  
  double* ptr = mxGetPr( array_ptr );

  if( (dimx != 1) && (dimy != 1) )
    mexErrMsgIdAndTxt("mexnice:error","Vector expected");
  

  NICE::SparseVector svec( std::max(dimx, dimy) );
  
  
  if ( dimx > 1)
  {
    for ( mwSize row=0; row < dimx; row++)
    { 
        // empty column?
        if (jc[row] == jc[row+1])
        {
          continue;
        }
        else
        {
          //note: no complex data supported her
            double value ( pr[total++] );
            if ( b_adaptIndexMtoC )
                svec.insert( std::pair<uint, double>( row+1,  value ) );
            else
                svec.insert( std::pair<uint, double>( row,  value ) );
        }
    } // for loop over cols      
  }
  else
  {
    mwSize numNonZero = jc[1]-jc[0];
    
    for ( mwSize colNonZero=0; colNonZero < numNonZero; colNonZero++)
    {
        //note: no complex data supported her
        double value ( pr[total++] );
        if ( b_adaptIndexMtoC )
            svec.insert( std::pair<uint, double>( ir[colNonZero]+1, value  ) );
        else
            svec.insert( std::pair<uint, double>( ir[colNonZero], value  ) );
    }          
  }

  return svec;
}

NICE::Matrix MatlabConversion::convertDoubleMatrixToNice( const mxArray* matlabMatrix )
{
  if( !mxIsDouble( matlabMatrix ) )
    mexErrMsgIdAndTxt( "mexnice:error","Expected double in convertDoubleMatrixToNice" );

  const mwSize *dims;
  int dimx, dimy, numdims;
  
  //figure out dimensions
  dims = mxGetDimensions( matlabMatrix );
  numdims = mxGetNumberOfDimensions( matlabMatrix );
  dimy = (int)dims[0];
  dimx = (int)dims[1];
  
  double* ptr = mxGetPr( matlabMatrix );

  NICE::Matrix niceMatrix(ptr, dimy, dimx, NICE::Matrix::external); 

  return niceMatrix;
}


NICE::Vector MatlabConversion::convertDoubleVectorToNice( const mxArray* matlabMatrix )
{
  if( !mxIsDouble( matlabMatrix ) )
    mexErrMsgIdAndTxt( "mexnice:error","Expected double in convertDoubleVectorToNice" );

  const mwSize *dims;
  int dimx, dimy, numdims;
  
  //figure out dimensions
  dims = mxGetDimensions( matlabMatrix );
  numdims = mxGetNumberOfDimensions( matlabMatrix );
  dimy = (int)dims[0];
  dimx = (int)dims[1];
  
  double* ptr = mxGetPr( matlabMatrix );

  if( (dimx != 1) && (dimy != 1) )
    mexErrMsgIdAndTxt("mexnice:error","Vector expected");

  int dim = std::max(dimx, dimy);    

  NICE::Vector niceVector( dim, 0.0 );
  
  for( int i = 0; i < dim; i++ )
  {
      niceVector(i) = ptr[i];
  }

  return niceVector;
}



std::string MatlabConversion::convertMatlabToString( const mxArray *matlabString )
{
  if( !mxIsChar( matlabString ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected string");

  char *cstring = mxArrayToString( matlabString );
  std::string s( cstring );
  mxFree(cstring);
  return s;
}


int MatlabConversion::convertMatlabToInt32( const mxArray *matlabInt32 )
{
  if( !mxIsInt32( matlabInt32 ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected int32");

  int* ptr = (int*) mxGetData( matlabInt32 );
  return ptr[0];
}

double MatlabConversion::convertMatlabToDouble( const mxArray *matlabDouble )
{
  if( !mxIsDouble(matlabDouble) )
    mexErrMsgIdAndTxt("mexnice:error","Expected double");

  double* ptr = (double*) mxGetData( matlabDouble );
  return ptr[0];
}

bool MatlabConversion::convertMatlabToBool( const mxArray *matlabBool )
{
  if( !mxIsLogical( matlabBool ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected bool");

  bool* ptr = (bool*) mxGetData( matlabBool );
  return ptr[0];
}

bool MatlabConversion::isSparseDataAMatrix( const mxArray *array_ptr )
{
     mwSize   i_numExamples, i_numDim;

     // dimenions of the matrix -> feature dimension and number of examples
     i_numExamples = mxGetM( array_ptr );
     i_numDim = mxGetN( array_ptr );

     if ( ( i_numExamples == 1) || ( i_numDim == 1) )
     {
         return false;
     }
     else
     {
         return true;
     }
 }
#endif
