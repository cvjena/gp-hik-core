/** 
* @file GPHIKRegressionMex.cpp
* @author Alexander Freytag
* @date 17-01-2014 (dd-mm-yyyy)
* @brief Matlab-Interface of our GPHIKRegression, allowing for training, regression, optimization, variance prediction, incremental learning, and  storing/re-storing.
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
#include "gp-hik-core/GPHIKRegression.h"


// Interface for conversion between Matlab and C objects
#include "gp-hik-core/matlab/classHandleMtoC.h"
#include "gp-hik-core/matlab/ConverterMatlabToNICE.h"
#include "gp-hik-core/matlab/ConverterNICEToMatlab.h"

const NICE::ConverterMatlabToNICE converterMtoNICE;
const NICE::ConverterNICEToMatlab converterNICEtoM;


using namespace std; //C basics
using namespace NICE;  // nice-core


NICE::Config parseParametersGPHIKRegression(const mxArray *prhs[], int nrhs)
{
  NICE::Config conf;
  
  // if first argument is the filename of an existing config file,
  // read the config accordingly
  
  int i_start ( 0 );
  std::string variable = converterMtoNICE.convertMatlabToString(prhs[i_start]);
  if(variable == "conf")
  {
      conf = NICE::Config ( converterMtoNICE.convertMatlabToString( prhs[i_start+1] )  );
      i_start = i_start+2;
  }
  
  // now run over all given parameter specifications
  // and add them to the config
  for( int i=i_start; i < nrhs; i+=2 )
  {
    std::string variable = converterMtoNICE.convertMatlabToString(prhs[i]);
    
    /////////////////////////////////////////
    // READ STANDARD BOOLEAN VARIABLES
    /////////////////////////////////////////
    if( (variable == "verboseTime") || (variable == "verbose") ||
        (variable == "optimize_noise") || (variable == "uncertaintyPredictionForRegression") ||
        (variable == "use_quantization") || (variable == "ils_verbose")
      )
    {
      if ( mxIsChar( prhs[i+1] ) )
      {
        string value = converterMtoNICE.convertMatlabToString( prhs[i+1] );
        if ( (value != "true") && (value != "false") )
        {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. In string modus, \'true\' or \'false\' expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );
        }
        
        if( value == "true" )
          conf.sB("GPHIKRegression", variable, true);
        else
          conf.sB("GPHIKRegression", variable, false);
      }
      else if ( mxIsLogical( prhs[i+1] ) )
      {
        bool value = converterMtoNICE.convertMatlabToBool( prhs[i+1] );
        conf.sB("GPHIKRegression", variable, value);
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
        double value = converterMtoNICE.convertMatlabToDouble(prhs[i+1]);
        conf.sI("GPHIKRegression", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = converterMtoNICE.convertMatlabToInt32(prhs[i+1]);
        conf.sI("GPHIKRegression", variable, value);          
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
        double value = converterMtoNICE.convertMatlabToDouble(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }
        conf.sI("GPHIKRegression", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = converterMtoNICE.convertMatlabToInt32(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }        
        conf.sI("GPHIKRegression", variable, value);          
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
        double value = converterMtoNICE.convertMatlabToDouble(prhs[i+1]);
        if( value < 0.0 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }
        conf.sD("GPHIKRegression", variable, value);        
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
      string value = converterMtoNICE.convertMatlabToString(prhs[i+1]);
      if(value != "CG" && value != "CGL" && value != "SYMMLQ" && value != "MINRES")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'ils_method\'. \'CG\', \'CGL\', \'SYMMLQ\' or \'MINRES\' expected.");
        conf.sS("GPHIKRegression", variable, value);
    }


    if(variable == "optimization_method")
    {
      string value = converterMtoNICE.convertMatlabToString(prhs[i+1]);
      if(value != "greedy" && value != "downhillsimplex" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'optimization_method\'. \'greedy\', \'downhillsimplex\' or \'none\' expected.");
        conf.sS("GPHIKRegression", variable, value);
    }

    if(variable == "transform")
    {
      string value = converterMtoNICE.convertMatlabToString( prhs[i+1] );
      if(value != "absexp" && value != "exp" && value != "MKL" && value != "WeightedDim")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'transform\'. \'absexp\', \'exp\' , \'MKL\' or \'WeightedDim\' expected.");
        conf.sS("GPHIKRegression", variable, value);
    }

  
    if(variable == "varianceApproximation")
    {
      string value = converterMtoNICE.convertMatlabToString(prhs[i+1]);
      if(value != "approximate_fine" && value != "approximate_rough" && value != "exact" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'varianceApproximation\'. \'approximate_fine\', \'approximate_rough\', \'none\' or \'exact\' expected.");
        conf.sS("GPHIKRegression", variable, value);
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
    
    std::string cmd = converterMtoNICE.convertMatlabToString( prhs[0] );
      
        
    // create object
    if ( !strcmp("new", cmd.c_str() ) )
    {
        // check output variable
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        
        // read config settings
        NICE::Config conf = parseParametersGPHIKRegression(prhs+1,nrhs-1);
        
        // create class instance
        NICE::GPHIKRegression * regressor = new NICE::GPHIKRegression ( &conf, "GPHIKRegression" /*sectionName in config*/ );
        
         
        // handle to the C++ instance
        plhs[0] = convertPtr2Mat<NICE::GPHIKRegression>( regressor );
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
        destroyObject<NICE::GPHIKRegression>(prhs[1]);
        return;
    }
    
    // get the class instance pointer from the second input
    // every following function needs the regressor object
    NICE::GPHIKRegression * regressor = convertMat2Ptr<NICE::GPHIKRegression>(prhs[1]);
    
    
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
        NICE::Vector yValuesTrain;  

        if ( mxIsSparse( prhs[2] ) )
        {
            examplesTrain = converterMtoNICE.convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix dataTrain;
            dataTrain = converterMtoNICE.convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            examplesTrain.resize( dataTrain.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
            for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
            }            
        }
          
        yValuesTrain = converterMtoNICE.convertDoubleVectorToNice(prhs[3]);

        //----------------- train our regressor -------------
        regressor->train ( examplesTrain , yValuesTrain );

        //----------------- clean up -------------
        for(int i=0;i<examplesTrain.size();i++)
            delete examplesTrain[i];
        
        return;
    }
    
    
    // perform regression    
    if ( !strcmp("estimate", cmd.c_str() ) )
    {
        // Check parameters
        if ( (nlhs < 0) || (nrhs < 2) )
        {
            mexErrMsgTxt("Test: Unexpected arguments.");
        }
        
        //------------- read the data --------------

        double result;
        double uncertainty;        

        if ( mxIsSparse( prhs[2] ) )
        {
            NICE::SparseVector * example;
            example = new NICE::SparseVector ( converterMtoNICE.convertSparseVectorToNice( prhs[2] ) );
            regressor->estimate ( example,  result, uncertainty );
            
            //----------------- clean up -------------
            delete example;
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( converterMtoNICE.convertDoubleVectorToNice(prhs[2]) ); 
            regressor->estimate ( example,  result, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
          
          

          // output
          plhs[0] = mxCreateDoubleScalar( result ); 
          
          
          if(nlhs >= 2)
          {
            plhs[1] = mxCreateDoubleScalar( uncertainty );          
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
            example = new NICE::SparseVector ( converterMtoNICE.convertSparseVectorToNice( prhs[2] ) );
            regressor->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( converterMtoNICE.convertDoubleVectorToNice(prhs[2]) ); 
            regressor->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        
       

          // output
          plhs[0] = mxCreateDoubleScalar( uncertainty );                    
          return;
    }    
    
    
    // Test - evaluate regressor on whole test set  
    if ( !strcmp("testL2loss", cmd.c_str() ) )
    {        
        // Check parameters
        if (nlhs < 0 || nrhs < 3)
            mexErrMsgTxt("Test: Unexpected arguments.");
        //------------- read the data --------------
        
        
        bool dataIsSparse ( mxIsSparse( prhs[2] ) );
        
        std::vector< const NICE::SparseVector *> dataTest_sparse;
        NICE::Matrix dataTest_dense;

        if ( dataIsSparse )
        {
            dataTest_sparse = converterMtoNICE.convertSparseMatrixToNice( prhs[2] );
        }
        else
        {    
            dataTest_dense = converterMtoNICE.convertDoubleMatrixToNice(prhs[2]);          
        }        

        NICE::Vector yValuesTest;
        yValuesTest = converterMtoNICE.convertDoubleVectorToNice(prhs[3]);
	
        int i_numTestSamples ( yValuesTest.size() );
        
	double l2loss ( 0.0 );
	
	NICE::Vector scores;
	NICE::Vector::iterator itScores;
	if ( nlhs >= 2 )
	{
	  scores.resize( i_numTestSamples );
	  itScores = scores.begin();
	}
          
          

        // ------------------------------------------
        // ------------- REGRESSION --------------
        // ------------------------------------------          
        
        NICE::Timer t;
        double testTime (0.0);
        


        for (int i = 0; i < i_numTestSamples; i++)
        {
            //----------------- convert data to sparse data structures ---------
          

            double result;

            if ( dataIsSparse )
            {                
              // and perform regression
              t.start();
              regressor->estimate( dataTest_sparse[ i ], result);
              t.stop();
              testTime += t.getLast();
            }
            else
            {
                NICE::Vector example ( dataTest_dense.getRow(i) );
              // and perform regression
              t.start();
              regressor->estimate( &example, result );
              t.stop();
              testTime += t.getLast();                
            }

            l2loss += pow ( yValuesTest[i] - result, 2); 
	    
	    if ( nlhs >= 2 )
	    {
	      *itScores = result;
	      itScores++;
	    }	    
        }
        
        std::cerr << "Time for testing: " << testTime << std::endl;          
        
        // clean up
        if ( dataIsSparse )
        {
            for ( std::vector<const NICE::SparseVector *>::iterator it = dataTest_sparse.begin(); it != dataTest_sparse.end(); it++) 
                delete *it;
        }
        


        plhs[0] = mxCreateDoubleScalar( l2loss );

        if(nlhs >= 2)
          plhs[1] = converterNICEtoM.convertVectorFromNice(scores);          
          
          
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
            newExample = new NICE::SparseVector ( converterMtoNICE.convertSparseVectorToNice( prhs[2] ) );
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( converterMtoNICE.convertDoubleVectorToNice(prhs[2]) ); 
            newExample = new NICE::SparseVector ( *example );
            //----------------- clean up -------------
            delete example;            
        }
        
        newLabel = converterMtoNICE.convertMatlabToDouble( prhs[3] );
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = converterMtoNICE.convertMatlabToBool( prhs[4] );
          
          regressor->addExample ( newExample,  newLabel, performOptimizationAfterIncrement );
        }
        else
        {
          regressor->addExample ( newExample,  newLabel );
        }
          
        
        //----------------- clean up -------------
        delete newExample;        

        return;
    }
    
    // addMultipleExamples    
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
            newExamples = converterMtoNICE.convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix newData;
            newData = converterMtoNICE.convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            newExamples.resize( newData.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = newExamples.begin();
            for (int i = 0; i < (int)newData.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( newData.getRow(i) );
            }            
        }
          
        newLabels = converterMtoNICE.convertDoubleVectorToNice(prhs[3]);
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = converterMtoNICE.convertMatlabToBool( prhs[4] );
          
          regressor->addMultipleExamples ( newExamples,  newLabels, performOptimizationAfterIncrement );
        }
        else
        {
          regressor->addMultipleExamples ( newExamples,  newLabels );
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
    
  
    
    // store the regressor  to an external file
    if ( !strcmp("store", cmd.c_str() ) || !strcmp("save", cmd.c_str() ) )
    {
        // Check parameters
        if ( nrhs < 3 )
            mexErrMsgTxt("store: no destination given.");        
               
        std::string s_destination = converterMtoNICE.convertMatlabToString( prhs[2] );
          
        std::filebuf fb;
        fb.open ( s_destination.c_str(), ios::out );
        std::ostream os(&fb);
        //
        regressor->store( os );
        //   
        fb.close();        
            
        return;
    }
    
    // load regressor from external file    
    if ( !strcmp("restore", cmd.c_str() ) || !strcmp("load", cmd.c_str() ) )
    {
        // Check parameters
        if ( nrhs < 3 )
            mexErrMsgTxt("restore: no destination given.");        
               
        std::string s_destination = converterMtoNICE.convertMatlabToString( prhs[2] );
        
        std::cerr << " aim at restoring the regressor from " << s_destination << std::endl;
          
        std::filebuf fbIn;
        fbIn.open ( s_destination.c_str(), ios::in );
        std::istream is (&fbIn);
        //
        regressor->restore( is );
        //   
        fbIn.close();        
            
        return;
    }    
    
    
    // Got here, so command not recognized
    
    std::string errorMsg (cmd.c_str() );
    errorMsg += " -- command not recognized.";
    mexErrMsgTxt( errorMsg.c_str() );

}
