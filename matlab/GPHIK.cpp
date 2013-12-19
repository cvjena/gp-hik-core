#include <math.h>
#include <matrix.h>
#include "mex.h"
#include "classHandleMtoC.h"

// NICE-core includes
#include <core/basics/Config.h>
#include <core/basics/Timer.h>
#include <core/vector/MatrixT.h>
#include <core/vector/VectorT.h>

// gp-hik-core includes
#include "gp-hik-core/GPHIKClassifier.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

/* Pass analyze_sparse a pointer to a sparse mxArray.  A sparse mxArray
   only stores its nonzero elements.  The values of the nonzero elements 
   are stored in the pr and pi arrays.  The tricky part of analyzing
   sparse mxArray's is figuring out the indices where the nonzero
   elements are stored.  (See the mxSetIr and mxSetJc reference pages
   for details. */  
std::vector< NICE::SparseVector * > convertSparseMatrixToNice(const mxArray *array_ptr)
{
  double  *pr;//, *pi;
  mwIndex  *ir, *jc;
  mwSize      col, total=0;
  mwIndex   starting_row_index, stopping_row_index, current_row_index;
  mwSize      i_numExamples, i_numDim;
  
  /* Get the starting positions of all four data arrays. */ 
  pr = mxGetPr(array_ptr);
//   pi = mxGetPi(array_ptr);
  ir = mxGetIr(array_ptr);
  jc = mxGetJc(array_ptr);
  
  // dimenions of the matrix -> feature dimension and number of examples
  i_numExamples = mxGetM(array_ptr);  
  i_numDim = mxGetN(array_ptr);
    
  // initialize output variable
  std::vector< NICE::SparseVector * > sparseMatrix;
  sparseMatrix.resize ( i_numExamples );
    
  for ( std::vector< NICE::SparseVector * >::iterator matIt = sparseMatrix.begin(); 
        matIt != sparseMatrix.end(); matIt++)
  {
      *matIt = new NICE::SparseVector( i_numDim );
  }  
  
  // now copy the data
  for (col=0; col < i_numDim; col++)
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
	        current_row_index++)
      {
          //note: no complex data supported her
          sparseMatrix[ ir[current_row_index] ]->insert( std::pair<int, double>( col, pr[total++] ) );
      } // for-loop
      
    }
  } // for-loop over columns
  
  return sparseMatrix;
}


// b_adaptIndexMtoC: if true, dim k will be inserted as k, not as k-1 (which would be the default for  M->C)
NICE::SparseVector convertSparseVectorToNice(const mxArray* array_ptr, const bool & b_adaptIndexMtoC = false )
{
  double  *pr, *pi;
  mwIndex  *ir, *jc;
  mwSize      col, total=0;
  mwIndex   starting_row_index, stopping_row_index, current_row_index;
  mwSize      dimy, dimx;
  
  /* Get the starting positions of all four data arrays. */ 
  pr = mxGetPr(array_ptr);
  pi = mxGetPi(array_ptr);
  ir = mxGetIr(array_ptr);
  jc = mxGetJc(array_ptr);
  
  // dimenions of the matrix -> feature dimension and number of examples
  dimy = mxGetM(array_ptr);  
  dimx = mxGetN(array_ptr);
  
  double* ptr = mxGetPr(array_ptr);

  if(dimx != 1 && dimy != 1)
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
	mxArray *matlabMatrix = mxCreateDoubleMatrix(niceMatrix.rows(),niceMatrix.cols(),mxREAL);
	double* matlabMatrixPtr = mxGetPr(matlabMatrix);

	for(int i=0; i<niceMatrix.rows(); i++)
    {
		for(int j=0; j<niceMatrix.cols(); j++)
		{
			matlabMatrixPtr[i + j*niceMatrix.rows()] = niceMatrix(i,j);
		}
    }
	return matlabMatrix;
}

NICE::Matrix convertMatrixToNice(const mxArray* matlabMatrix)
{
	//todo: do not assume double

  const mwSize *dims;
  int dimx, dimy, numdims;
    //figure out dimensions
  dims = mxGetDimensions(matlabMatrix);
  numdims = mxGetNumberOfDimensions(matlabMatrix);
  dimy = (int)dims[0]; dimx = (int)dims[1];
  double* ptr = mxGetPr(matlabMatrix);

  NICE::Matrix niceMatrix(ptr, dimy, dimx, NICE::Matrix::external); 

  return niceMatrix;
}

mxArray* convertVectorFromNice(NICE::Vector & niceVector)
{
	//cout << "start convertVectorFromNice" << endl;
	mxArray *matlabVector = mxCreateDoubleMatrix(niceVector.size(), 1, mxREAL);
	double* matlabVectorPtr = mxGetPr(matlabVector);

	for(int i=0;i<niceVector.size(); i++)
    {
        matlabVectorPtr[i] = niceVector[i];
    }
	return matlabVector;
}

NICE::Vector convertVectorToNice(const mxArray* matlabMatrix)
{
	//todo: do not assume double

  const mwSize *dims;
  int dimx, dimy, numdims;
    //figure out dimensions
  dims = mxGetDimensions(matlabMatrix);
  numdims = mxGetNumberOfDimensions(matlabMatrix);
  dimy = (int)dims[0]; dimx = (int)dims[1];
  double* ptr = mxGetPr(matlabMatrix);

  if(dimx != 1 && dimy != 1)
    mexErrMsgIdAndTxt("mexnice:error","Vector expected");

  int dim = max(dimx, dimy);    

  NICE::Vector niceVector(dim, 0.0);
  
  for(int i=0;i<dim;i++)
  {
      niceVector(i) = ptr[i];
  }

  return niceVector;
}



std::string convertMatlabToString(const mxArray *matlabString)
{
  if(!mxIsChar(matlabString))
    mexErrMsgIdAndTxt("mexnice:error","Expected string");

  char *cstring = mxArrayToString(matlabString);
  std::string s(cstring);
  mxFree(cstring);
  return s;
}


int convertMatlabToInt32(const mxArray *matlabInt32)
{
  if(!mxIsInt32(matlabInt32))
    mexErrMsgIdAndTxt("mexnice:error","Expected int32");

  int* ptr = (int*)mxGetData(matlabInt32);
  return ptr[0];
}

double convertMatlabToDouble(const mxArray *matlabDouble)
{
  if(!mxIsDouble(matlabDouble))
    mexErrMsgIdAndTxt("mexnice:error","Expected double");

  double* ptr = (double*)mxGetData(matlabDouble);
  return ptr[0];
}

Config parseParameters(const mxArray *prhs[], int nrhs)
{
  Config conf;
  for(int i=0;i<nrhs;i+=2)
  {
    string variable = convertMatlabToString(prhs[i]);
    if(variable == "ils_verbose")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'ils_verbose\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }

    if(variable == "ils_max_iterations")
    {
      int value = convertMatlabToInt32(prhs[i+1]);
      if(value < 1)
        mexErrMsgIdAndTxt("mexnice:error","Expected parameter value larger than 0 for \'ils_max_iterations\'.");
      conf.sI("GPHIKClassifier", variable, value);
    }

    if(variable == "ils_method")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "CG" && value != "CGL" && value != "SYMMLQ" && value != "MINRES")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'ils_method\'. \'CG\', \'CGL\', \'SYMMLQ\' or \'MINRES\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

    if(variable == "ils_min_delta")
    {
      double value = convertMatlabToDouble(prhs[i+1]);
      if(value < 0.0)
        mexErrMsgIdAndTxt("mexnice:error","Expected parameter value larger than 0 for \'ils_min_delta\'.");
      conf.sD("GPHIKClassifier", variable, value);
    }

    if(variable == "ils_min_residual")
    {
      double value = convertMatlabToDouble(prhs[i+1]);
      if(value < 0.0)
        mexErrMsgIdAndTxt("mexnice:error","Expected parameter value larger than 0 for \'ils_min_residual\'.");
      conf.sD("GPHIKClassifier", variable, value);
    }


    if(variable == "optimization_method")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "greedy" && value != "downhillsimplex" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'optimization_method\'. \'greedy\', \'downhillsimplex\' or \'none\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

    if(variable == "use_quantization")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'use_quantization\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }

    if(variable == "num_bins")
    {
      int value = convertMatlabToInt32(prhs[i+1]);
      if(value < 1)
        mexErrMsgIdAndTxt("mexnice:error","Expected parameter value larger than 0 for \'num_bins\'.");
      conf.sI("GPHIKClassifier", variable, value);
    }

    if(variable == "transform")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "absexp" && value != "exp" && value != "MKL" && value != "WeightedDim")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'transform\'. \'absexp\', \'exp\' , \'MKL\' or \'WeightedDim\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

    if(variable == "verboseTime")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'verboseTime\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }

    if(variable == "verbose")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'verbose\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }

    if(variable == "noise")
    {
      double value = convertMatlabToDouble(prhs[i+1]);
      if(value < 0.0)
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value larger than 0 for \'noise\'.");
      conf.sD("GPHIKClassifier", variable, value);
    }

    if(variable == "learn_balanced")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'learn_balanced\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }

    if(variable == "optimize_noise")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "true" && value != "false")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'optimize_noise\'. \'true\' or \'false\' expected.");
      if(value == "true")
        conf.sB("GPHIKClassifier", variable, true);
      else
        conf.sB("GPHIKClassifier", variable, false);
    }
    
    if(variable == "varianceApproximation")
    {
      string value = convertMatlabToString(prhs[i+1]);
      if(value != "approximate_fine" && value != "approximate_rough" && value != "exact" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'varianceApproximation\'. \'approximate_fine\', \'approximate_rough\', \'none\' or \'exact\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }
    
    if(variable == "nrOfEigenvaluesToConsiderForVarApprox")
    {
      double value = convertMatlabToDouble(prhs[i+1]);
      conf.sI("GPHIKClassifier", variable, (int) value);
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
        NICE::GPHIKClassifier * classifier = new NICE::GPHIKClassifier ( &conf );
        
         
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
          
        std::vector< NICE::SparseVector *> examplesTrain;
        NICE::Vector yMultiTrain;  

        if ( mxIsSparse( prhs[2] ) )
        {
            examplesTrain = convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix dataTrain;
            dataTrain = convertMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            examplesTrain.resize( dataTrain.rows() );

                    
            std::vector< NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
            for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
            }            
        }
          
          yMultiTrain = convertVectorToNice(prhs[3]);
          
//           std::cerr << " DATA AFTER CONVERSION: \n" << std::endl;
//           int lineIdx(0);
//           for ( std::vector< NICE::SparseVector *>::const_iterator exTrainIt = examplesTrain.begin();
//                 exTrainIt != examplesTrain.end(); exTrainIt++, lineIdx++)
//           {
//               std::cerr << "\n lineIdx: " << lineIdx << std::endl;
//               (*exTrainIt)->store( std::cerr );
//               
//           }

          // test assumption
          {
            if( yMultiTrain.Min() < 0)
              mexErrMsgIdAndTxt("mexnice:error","Class labels smaller 0 are not allowed");
          }


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
            example = new NICE::Vector ( convertVectorToNice(prhs[2]) ); 
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
    
    // Classify    
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
            example = new NICE::Vector ( convertVectorToNice(prhs[2]) ); 
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
        
        std::vector< NICE::SparseVector *> dataTest_sparse;
        NICE::Matrix dataTest_dense;

        if ( dataIsSparse )
        {
            dataTest_sparse = convertSparseMatrixToNice( prhs[2] );
        }
        else
        {    
            dataTest_dense = convertMatrixToNice(prhs[2]);          
        }        

          NICE::Vector yMultiTest;
          yMultiTest = convertVectorToNice(prhs[3]);

          
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
                scores(i,scoreCnt) = scoreIt->second;
                
          }
          
          std::cerr << "Time for testing: " << testTime << std::endl;          
          
          // clean up
          if ( dataIsSparse )
          {
              for ( std::vector<NICE::SparseVector *>::iterator it = dataTest_sparse.begin(); it != dataTest_sparse.end(); it++) 
                  delete *it;
          }
          


          confusionMatrix.normalizeColumnsL1();
          //std::cerr << confusionMatrix << std::endl;

          double recRate = confusionMatrix.trace()/confusionMatrix.rows();
          //std::cerr << "average recognition rate: " << recRate << std::endl;

          
          plhs[0] = mxCreateDoubleScalar( recRate );

          if(nlhs >= 2)
            plhs[1] = convertMatrixFromNice(confusionMatrix);
          if(nlhs >= 3)
            plhs[2] = convertMatrixFromNice(scores);          
          
          
        return;
    }
    
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
