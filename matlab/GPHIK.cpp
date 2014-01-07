/** 
* @file GPHIK.cpp
* @author Alexander Freytag
* @date 07-01-2014 (dd-mm-yyyy)
* @brief Matlab-Interface of our GPHIKClassifier, allowing for training, classification, optimization, variance prediction, incremental learning, and  storing/re-storing.
*/

// STL includes
#include <math.h>
#include <matrix.h>
#include <mex.h>

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Timer.h>
#include <core/vector/MatrixT.h>
#include <core/vector/VectorT.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKClassifier.h"

// Interface for conversion between Matlab and C objects
#include "classHandleMtoC.h"


using namespace std; //C basics
using namespace NICE;  // nice-core

/* A sparse mxArray only stores its nonzero elements.
 * The values of the nonzero elements are stored in 
 * the pr and pi arrays.  The tricky part of analyzing
 * sparse mxArray's is figuring out the indices where 
 * the nonzero elements are stored.
 * (See the mxSetIr and mxSetJc reference pages for details. */  
std::vector< const NICE::SparseVector * > convertSparseMatrixToNice(const mxArray *array_ptr)
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
          sparseMatrix[ ir[current_row_index] ]->insert( std::pair<int, double>( col, pr[total++] ) );
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
NICE::SparseVector convertSparseVectorToNice(const mxArray* array_ptr, const bool & b_adaptIndexMtoC = false )
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
                svec.insert( std::pair<int, double>( row+1,  value ) );
            else
                svec.insert( std::pair<int, double>( row,  value ) );
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
            svec.insert( std::pair<int, double>( ir[colNonZero]+1, value  ) );
        else
            svec.insert( std::pair<int, double>( ir[colNonZero], value  ) );
    }          
  }

  return svec;
}

// b_adaptIndexCtoM: if true, dim k will be inserted as k, not as k+1 (which would be the default for C->M)
mxArray* convertSparseVectorFromNice( const NICE::SparseVector & scores, const bool & b_adaptIndexCtoM = false)
{
    mxArray * matlabSparseVec = mxCreateSparse( scores.getDim() /*m*/, 1/*n*/, scores.size()/*nzmax*/, mxREAL);
    
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
    jc[1] = scores.size(); jc[0] = 0; 
    
    
    mwSize cnt = 0;
        
    for ( NICE::SparseVector::const_iterator myIt = scores.begin(); myIt != scores.end(); myIt++, cnt++ )
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


mxArray* convertMatrixFromNice(NICE::Matrix & niceMatrix)
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

NICE::Matrix convertDoubleMatrixToNice(const mxArray* matlabMatrix)
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

mxArray* convertVectorFromNice(NICE::Vector & niceVector)
{
  mxArray *matlabVector = mxCreateDoubleMatrix( niceVector.size(), 1, mxREAL );
  double* matlabVectorPtr = mxGetPr( matlabVector );

  for( int i = 0; i < niceVector.size(); i++ )
  {
    matlabVectorPtr[i] = niceVector[i];
  }
  return matlabVector;
}

NICE::Vector convertDoubleVectorToNice( const mxArray* matlabMatrix )
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



std::string convertMatlabToString( const mxArray *matlabString )
{
  if( !mxIsChar( matlabString ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected string");

  char *cstring = mxArrayToString( matlabString );
  std::string s( cstring );
  mxFree(cstring);
  return s;
}


int convertMatlabToInt32( const mxArray *matlabInt32 )
{
  if( !mxIsInt32( matlabInt32 ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected int32");

  int* ptr = (int*) mxGetData( matlabInt32 );
  return ptr[0];
}

double convertMatlabToDouble( const mxArray *matlabDouble )
{
  if( !mxIsDouble(matlabDouble) )
    mexErrMsgIdAndTxt("mexnice:error","Expected double");

  double* ptr = (double*) mxGetData( matlabDouble );
  return ptr[0];
}

bool convertMatlabToBool(const mxArray *matlabBool)
{
  if( !mxIsLogical( matlabBool ) )
    mexErrMsgIdAndTxt("mexnice:error","Expected bool");

  bool* ptr = (bool*) mxGetData( matlabBool );
  return ptr[0];
}

NICE::Config parseParameters(const mxArray *prhs[], int nrhs)
{
  NICE::Config conf;
  
  // if first argument is the filename of an existing config file,
  // read the config accordingly
  
  int i_start ( 0 );
  std::string variable = convertMatlabToString(prhs[i_start]);
  if(variable == "conf")
  {
      conf = NICE::Config ( convertMatlabToString( prhs[i_start+1] )  );
      i_start = i_start+2;
  }
  
  // now run over all given parameter specifications
  // and add them to the config
  for( int i=i_start; i < nrhs; i+=2 )
  {
    std::string variable = convertMatlabToString(prhs[i]);
    
    /////////////////////////////////////////
    // READ STANDARD BOOLEAN VARIABLES
    /////////////////////////////////////////
    if( (variable == "verboseTime") || (variable == "verbose") ||
        (variable == "optimize_noise") || (variable == "uncertaintyPredictionForClassification") ||
        (variable == "use_quantization") || (variable == "ils_verbose")
      )
    {
      if ( mxIsChar( prhs[i+1] ) )
      {
        string value = convertMatlabToString( prhs[i+1] );
        if ( (value != "true") && (value != "false") )
        {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. In string modus, \'true\' or \'false\' expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );
        }
        
        if( value == "true" )
          conf.sB("GPHIKClassifier", variable, true);
        else
          conf.sB("GPHIKClassifier", variable, false);
      }
      else if ( mxIsLogical( prhs[i+1] ) )
      {
        bool value = convertMatlabToBool( prhs[i+1] );
        conf.sB("GPHIKClassifier", variable, value);
      }
      else
      {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. \'true\', \'false\', or logical expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );        
      }
    }
    
    /////////////////////////////////////////
    // READ STANDARD INT VARIABLES
    /////////////////////////////////////////
    if ( (variable == "nrOfEigenvaluesToConsiderForVarApprox")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = convertMatlabToDouble(prhs[i+1]);
        conf.sI("GPHIKClassifier", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = convertMatlabToInt32(prhs[i+1]);
        conf.sI("GPHIKClassifier", variable, value);          
      }
      else
      {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. Int32 or Double expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );         
      }     
    }
    
    /////////////////////////////////////////
    // READ STRICT POSITIVE INT VARIABLES
    /////////////////////////////////////////
    if ( (variable == "num_bins") || (variable == "ils_max_iterations")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = convertMatlabToDouble(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }
        conf.sI("GPHIKClassifier", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = convertMatlabToInt32(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }        
        conf.sI("GPHIKClassifier", variable, value);          
      }
      else
      {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. Int32 or Double expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );         
      }     
    }
    
    /////////////////////////////////////////
    // READ POSITIVE DOUBLE VARIABLES
    /////////////////////////////////////////
    if ( (variable == "ils_min_delta") || (variable == "ils_min_residual") ||
         (variable == "noise")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = convertMatlabToDouble(prhs[i+1]);
        if( value < 0.0 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }
        conf.sD("GPHIKClassifier", variable, value);        
      }
      else
      {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. Double expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );         
      }     
    }    
    
    /////////////////////////////////////////
    // READ REMAINING SPECIFIC VARIABLES
    /////////////////////////////////////////  

    if(variable == "ils_method")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "CG" && value != "CGL" && value != "SYMMLQ" && value != "MINRES")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'ils_method\'. \'CG\', \'CGL\', \'SYMMLQ\' or \'MINRES\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }


    if(variable == "optimization_method")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "greedy" && value != "downhillsimplex" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'optimization_method\'. \'greedy\', \'downhillsimplex\' or \'none\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

    if(variable == "transform")
    {
      string value = convertMatlabToString( prhs[i+1] );
      if(value != "absexp" && value != "exp" && value != "MKL" && value != "WeightedDim")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'transform\'. \'absexp\', \'exp\' , \'MKL\' or \'WeightedDim\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

  
    if(variable == "varianceApproximation")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "approximate_fine" && value != "approximate_rough" && value != "exact" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'varianceApproximation\'. \'approximate_fine\', \'approximate_rough\', \'none\' or \'exact\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }
    

    
  }


  return conf;
}

// MAIN MATLAB FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    // get the command string specifying what to do
    if (nrhs < 1)
        mexErrMsgTxt("No commands and options passed... Aborting!");        
    
    if( !mxIsChar( prhs[0] ) )
        mexErrMsgTxt("First argument needs to be the command, ie.e, the class method to call... Aborting!");        
    
    std::string cmd = convertMatlabToString( prhs[0] );
      
        
    // create object
    if ( !strcmp("new", cmd.c_str() ) )
    {
        // check output variable
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        
        // read config settings
        NICE::Config conf = parseParameters(prhs+1,nrhs-1);
        
        // create class instance
        NICE::GPHIKClassifier * classifier = new NICE::GPHIKClassifier ( &conf, "GPHIKClassifier" /*sectionName in config*/ );
        
         
        // handle to the C++ instance
        plhs[0] = convertPtr2Mat<NICE::GPHIKClassifier>( classifier );
        return;
    }
    
    // in all other cases, there should be a second input,
    // which the be the class instance handle
    if (nrhs < 2)
      mexErrMsgTxt("Second input should be a class instance handle.");
    
    // delete object
    if ( !strcmp("delete", cmd.c_str() ) )
    {
        // Destroy the C++ object
        destroyObject<NICE::GPHIKClassifier>(prhs[1]);
        return;
    }
    
    // get the class instance pointer from the second input
    // every following function needs the classifier object
    NICE::GPHIKClassifier * classifier = convertMat2Ptr<NICE::GPHIKClassifier>(prhs[1]);
    
    
    ////////////////////////////////////////
    //  Check which class method to call  //
    ////////////////////////////////////////
    
    
    // standard train - assumes initialized object
    if (!strcmp("train", cmd.c_str() ))
    {
        // Check parameters
        if (nlhs < 0 || nrhs < 4)
        {
            mexErrMsgTxt("Train: Unexpected arguments.");
        }
        
        //------------- read the data --------------
          
        std::vector< const NICE::SparseVector *> examplesTrain;
        NICE::Vector yMultiTrain;  

        if ( mxIsSparse( prhs[2] ) )
        {
            examplesTrain = convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix dataTrain;
            dataTrain = convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            examplesTrain.resize( dataTrain.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
            for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
            }            
        }
          
        yMultiTrain = convertDoubleVectorToNice(prhs[3]);

        //----------------- train our classifier -------------
        classifier->train ( examplesTrain , yMultiTrain );

        //----------------- clean up -------------
        for(int i=0;i<examplesTrain.size();i++)
            delete examplesTrain[i];
        
        return;
    }
    
    
    // Classify    
    if ( !strcmp("classify", cmd.c_str() ) )
    {
        // Check parameters
        if ( (nlhs < 0) || (nrhs < 2) )
        {
            mexErrMsgTxt("Test: Unexpected arguments.");
        }
        
        //------------- read the data --------------

        int result;
        NICE::SparseVector scores;
        double uncertainty;        

        if ( mxIsSparse( prhs[2] ) )
        {
            NICE::SparseVector * example;
            example = new NICE::SparseVector ( convertSparseVectorToNice( prhs[2] ) );
            classifier->classify ( example,  result, scores, uncertainty );
            
            //----------------- clean up -------------
            delete example;
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( convertDoubleVectorToNice(prhs[2]) ); 
            classifier->classify ( example,  result, scores, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
          
          

          // output
          plhs[0] = mxCreateDoubleScalar( result ); 
          
          
          if(nlhs >= 2)
          {
            plhs[1] = convertSparseVectorFromNice( scores, true  /*b_adaptIndex*/);
          }
          if(nlhs >= 3)
          {
            plhs[2] = mxCreateDoubleScalar( uncertainty );          
          }
          return;
    }
    
    // Uncertainty prediction    
    if ( !strcmp("uncertainty", cmd.c_str() ) )
    {
        // Check parameters
        if ( (nlhs < 0) || (nrhs < 2) )
        {
            mexErrMsgTxt("Test: Unexpected arguments.");
        }
        
        double uncertainty;        
        
        //------------- read the data --------------

        if ( mxIsSparse( prhs[2] ) )
        {
            NICE::SparseVector * example;
            example = new NICE::SparseVector ( convertSparseVectorToNice( prhs[2] ) );
            classifier->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( convertDoubleVectorToNice(prhs[2]) ); 
            classifier->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        
       

          // output
          plhs[0] = mxCreateDoubleScalar( uncertainty );                    
          return;
    }    
    
    
    // Test    
    if ( !strcmp("test", cmd.c_str() ) )
    {        
        // Check parameters
        if (nlhs < 0 || nrhs < 4)
            mexErrMsgTxt("Test: Unexpected arguments.");
        //------------- read the data --------------
        
        
        bool dataIsSparse ( mxIsSparse( prhs[2] ) );
        
        std::vector< const NICE::SparseVector *> dataTest_sparse;
        NICE::Matrix dataTest_dense;

        if ( dataIsSparse )
        {
            dataTest_sparse = convertSparseMatrixToNice( prhs[2] );
        }
        else
        {    
            dataTest_dense = convertDoubleMatrixToNice(prhs[2]);          
        }        

        NICE::Vector yMultiTest;
        yMultiTest = convertDoubleVectorToNice(prhs[3]);

        
        // ------------------------------------------
        // ------------- PREPARATION --------------
        // ------------------------------------------   
        
        // determine classes known during training and corresponding mapping
        // thereby allow for non-continous class labels
        std::set<int> classesKnownTraining = classifier->getKnownClassNumbers();
        
        int noClassesKnownTraining ( classesKnownTraining.size() );
        std::map<int,int> mapClNoToIdxTrain;
        std::set<int>::const_iterator clTrIt = classesKnownTraining.begin();
        for ( int i=0; i < noClassesKnownTraining; i++, clTrIt++ )
            mapClNoToIdxTrain.insert ( std::pair<int,int> ( *clTrIt, i )  );
        
        // determine classes known during testing and corresponding mapping
        // thereby allow for non-continous class labels
        std::set<int> classesKnownTest;
        classesKnownTest.clear();
        

        // determine which classes we have in our label vector
        // -> MATLAB: myClasses = unique(y);
        for ( NICE::Vector::const_iterator it = yMultiTest.begin(); it != yMultiTest.end(); it++ )
        {
          if ( classesKnownTest.find ( *it ) == classesKnownTest.end() )
          {
            classesKnownTest.insert ( *it );
          }
        }          
        
        int noClassesKnownTest ( classesKnownTest.size() );  
        std::map<int,int> mapClNoToIdxTest;
        std::set<int>::const_iterator clTestIt = classesKnownTest.begin();
        for ( int i=0; i < noClassesKnownTest; i++, clTestIt++ )
            mapClNoToIdxTest.insert ( std::pair<int,int> ( *clTestIt, i )  );          
        


        int i_numTestSamples;
        
        if ( dataIsSparse ) 
            i_numTestSamples = dataTest_sparse.size();
        else
            i_numTestSamples = (int) dataTest_dense.rows();
        
        NICE::Matrix confusionMatrix( noClassesKnownTraining, noClassesKnownTest, 0.0);
        NICE::Matrix scores( i_numTestSamples, noClassesKnownTraining, 0.0);
          
          

        // ------------------------------------------
        // ------------- CLASSIFICATION --------------
        // ------------------------------------------          
        
        NICE::Timer t;
        double testTime (0.0);
        


        for (int i = 0; i < i_numTestSamples; i++)
        {
            //----------------- convert data to sparse data structures ---------
          

            int result;
            NICE::SparseVector exampleScoresSparse;

            if ( dataIsSparse )
            {                
              // and classify
              t.start();
              classifier->classify( dataTest_sparse[ i ], result, exampleScoresSparse );
              t.stop();
              testTime += t.getLast();
            }
            else
            {
                NICE::Vector example ( dataTest_dense.getRow(i) );
              // and classify
              t.start();
              classifier->classify( &example, result, exampleScoresSparse );
              t.stop();
              testTime += t.getLast();                
            }

            confusionMatrix(  mapClNoToIdxTrain.find(result)->second, mapClNoToIdxTest.find(yMultiTest[i])->second ) += 1.0;
            int scoreCnt ( 0 );
            for ( NICE::SparseVector::const_iterator scoreIt = exampleScoresSparse.begin(); scoreIt != exampleScoresSparse.end(); scoreIt++, scoreCnt++ )
            {
              scores(i,scoreCnt) = scoreIt->second;
            }
              
        }
        
        std::cerr << "Time for testing: " << testTime << std::endl;          
        
        // clean up
        if ( dataIsSparse )
        {
            for ( std::vector<const NICE::SparseVector *>::iterator it = dataTest_sparse.begin(); it != dataTest_sparse.end(); it++) 
                delete *it;
        }
        


        confusionMatrix.normalizeColumnsL1();

        double recRate = confusionMatrix.trace()/confusionMatrix.cols();

        
        plhs[0] = mxCreateDoubleScalar( recRate );

        if(nlhs >= 2)
          plhs[1] = convertMatrixFromNice(confusionMatrix);
        if(nlhs >= 3)
          plhs[2] = convertMatrixFromNice(scores);          
          
          
        return;
    }
    
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
    // interface specific methods for incremental extensions
    ///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////      
    
    // addExample    
    if ( !strcmp("addExample", cmd.c_str() ) )
    {
        // Check parameters
        if ( (nlhs < 0) || (nrhs < 4) )
        {
            mexErrMsgTxt("Test: Unexpected arguments.");
        }
        
        //------------- read the data --------------

        NICE::SparseVector * newExample;
        double newLabel;        

        if ( mxIsSparse( prhs[2] ) )
        {
            newExample = new NICE::SparseVector ( convertSparseVectorToNice( prhs[2] ) );
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( convertDoubleVectorToNice(prhs[2]) ); 
            newExample = new NICE::SparseVector ( *example );
            //----------------- clean up -------------
            delete example;            
        }
        
        newLabel = convertMatlabToDouble( prhs[3] );
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = convertMatlabToBool( prhs[4] );
          
          classifier->addExample ( newExample,  newLabel, performOptimizationAfterIncrement );
        }
        else
        {
          classifier->addExample ( newExample,  newLabel );
        }
          
        
        //----------------- clean up -------------
        delete newExample;        

        return;
    }
    
    // addExample    
    if ( !strcmp("addMultipleExamples", cmd.c_str() ) )
    {
        // Check parameters
        if ( (nlhs < 0) || (nrhs < 4) )
        {
            mexErrMsgTxt("Test: Unexpected arguments.");
        }
        
        //------------- read the data --------------

        std::vector< const NICE::SparseVector *> newExamples;
        NICE::Vector newLabels;

        if ( mxIsSparse( prhs[2] ) )
        {
            newExamples = convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix newData;
            newData = convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            newExamples.resize( newData.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = newExamples.begin();
            for (int i = 0; i < (int)newData.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( newData.getRow(i) );
            }            
        }
          
        newLabels = convertDoubleVectorToNice(prhs[3]);
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = convertMatlabToBool( prhs[4] );
          
          classifier->addMultipleExamples ( newExamples,  newLabels, performOptimizationAfterIncrement );
        }
        else
        {
          classifier->addMultipleExamples ( newExamples,  newLabels );
        }
          
        
        //----------------- clean up -------------
        for ( std::vector< const NICE::SparseVector *>::iterator exIt = newExamples.begin();
              exIt != newExamples.end(); exIt++
            ) 
        {
          delete *exIt;
        }

        return;
    }    
    

    
    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////    
    
  
    
    // store the classifier    
    if ( !strcmp("store", cmd.c_str() ) || !strcmp("save", cmd.c_str() ) )
    {
        // Check parameters
        if ( nrhs < 3 )
            mexErrMsgTxt("store: no destination given.");        
               
        std::string s_destination = convertMatlabToString( prhs[2] );
          
        std::filebuf fb;
        fb.open ( s_destination.c_str(), ios::out );
        std::ostream os(&fb);
        //
        classifier->store( os );
        //   
        fb.close();        
            
        return;
    }
    
    // load classifier from external file    
    if ( !strcmp("restore", cmd.c_str() ) || !strcmp("load", cmd.c_str() ) )
    {
        // Check parameters
        if ( nrhs < 3 )
            mexErrMsgTxt("restore: no destination given.");        
               
        std::string s_destination = convertMatlabToString( prhs[2] );
        
        std::cerr << " aim at restoring the classifier from " << s_destination << std::endl;
          
        std::filebuf fbIn;
        fbIn.open ( s_destination.c_str(), ios::in );
        std::istream is (&fbIn);
        //
        classifier->restore( is );
        //   
        fbIn.close();        
            
        return;
    }    
    
    
    // Got here, so command not recognized
    
    std::string errorMsg (cmd.c_str() );
    errorMsg += " -- command not recognized.";
    mexErrMsgTxt( errorMsg.c_str() );

}
