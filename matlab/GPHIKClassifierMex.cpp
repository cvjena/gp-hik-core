#ifdef NICE_USELIB_MEX
/** 
* @file GPHIKClassifierMex.cpp
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
#include "gp-hik-core/matlab/classHandleMtoC.h"
#include "gp-hik-core/matlab/ConverterMatlabToNICE.h"
#include "gp-hik-core/matlab/ConverterNICEToMatlab.h"

using namespace std; //C basics
using namespace NICE;  // nice-core


NICE::Config parseParametersGPHIKClassifier(const mxArray *prhs[], int nrhs)
{
  NICE::Config conf;
  
  // if first argument is the filename of an existing config file,
  // read the config accordingly
  
  int i_start ( 0 );
  std::string variable = MatlabConversion::convertMatlabToString(prhs[i_start]);
  if(variable == "conf")
  {
      conf = NICE::Config ( MatlabConversion::convertMatlabToString( prhs[i_start+1] )  );
      i_start = i_start+2;
  }
  
  // now run over all given parameter specifications
  // and add them to the config
  for( int i=i_start; i < nrhs; i+=2 )
  {
    std::string variable = MatlabConversion::convertMatlabToString(prhs[i]);
    
    /////////////////////////////////////////
    // READ STANDARD BOOLEAN VARIABLES
    /////////////////////////////////////////
    if( (variable == "verboseTime") || 
        (variable == "verbose") ||
        (variable == "debug") ||            
        (variable == "optimize_noise") || 
        (variable == "uncertaintyPredictionForClassification") ||
        (variable == "use_quantization") || 
        (variable == "ils_verbose")
      )
    {
      if ( mxIsChar( prhs[i+1] ) )
      {
        string value = MatlabConversion::convertMatlabToString( prhs[i+1] );
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
        bool value = MatlabConversion::convertMatlabToBool( prhs[i+1] );
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
        double value = MatlabConversion::convertMatlabToDouble(prhs[i+1]);
        conf.sI("GPHIKClassifier", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = MatlabConversion::convertMatlabToInt32(prhs[i+1]);
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
    if ( (variable == "num_bins") || 
         (variable == "ils_max_iterations")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = MatlabConversion::convertMatlabToDouble(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );     
        }
        conf.sI("GPHIKClassifier", variable, (int) value);        
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = MatlabConversion::convertMatlabToInt32(prhs[i+1]);
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
    // READ STANDARD DOUBLE VARIABLES
    /////////////////////////////////////////
    if ( (variable == "parameter_upper_bound") || 
         (variable == "parameter_lower_bound")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = MatlabConversion::convertMatlabToDouble(prhs[i+1]);

        conf.sD("GPHIKClassifier", variable, value);        
      }
      else
      {
          std::string errorMsg = "Unexpected parameter value for \'" +  variable + "\'. Double expected.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );         
      }     
    }        
    
    /////////////////////////////////////////
    // READ POSITIVE DOUBLE VARIABLES
    /////////////////////////////////////////
    if ( (variable == "ils_min_delta") || 
         (variable == "ils_min_residual") ||
         (variable == "noise")
       )
    {
      if ( mxIsDouble( prhs[i+1] ) )
      {
        double value = MatlabConversion::convertMatlabToDouble(prhs[i+1]);
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
      string value = MatlabConversion::convertMatlabToString(prhs[i+1]);
      if(value != "CG" && value != "CGL" && value != "SYMMLQ" && value != "MINRES")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'ils_method\'. \'CG\', \'CGL\', \'SYMMLQ\' or \'MINRES\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }


    if(variable == "optimization_method")
    {
      string value = MatlabConversion::convertMatlabToString(prhs[i+1]);
      if(value != "greedy" && value != "downhillsimplex" && value != "none")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'optimization_method\'. \'greedy\', \'downhillsimplex\' or \'none\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

    if(variable == "s_quantType")
    {
      string value = MatlabConversion::convertMatlabToString( prhs[i+1] );
      if( value != "1d-aequi-0-1" && value != "1d-aequi-0-max" && value != "nd-aequi-0-max" )
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'s_quantType\'. \'1d-aequi-0-1\' , \'1d-aequi-0-max\' or \'nd-aequi-0-max\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }
    
    if(variable == "transform")
    {
      string value = MatlabConversion::convertMatlabToString( prhs[i+1] );
      if( value != "identity" && value != "absexp" && value != "exp" && value != "MKL" && value != "WeightedDim")
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'transform\'. \'identity\', \'absexp\', \'exp\' , \'MKL\' or \'WeightedDim\' expected.");
        conf.sS("GPHIKClassifier", variable, value);
    }

  
    if(variable == "varianceApproximation")
    {
      string value = MatlabConversion::convertMatlabToString(prhs[i+1]);
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
    
    std::string cmd = MatlabConversion::convertMatlabToString( prhs[0] );
      
        
    // create object
    if ( !strcmp("new", cmd.c_str() ) )
    {
        // check output variable
        if (nlhs != 1)
            mexErrMsgTxt("New: One output expected.");
        
        // read config settings
        NICE::Config conf = parseParametersGPHIKClassifier(prhs+1,nrhs-1);
        
        // create class instance
        NICE::GPHIKClassifier * classifier = new NICE::GPHIKClassifier ( &conf, "GPHIKClassifier" /*sectionName in config*/ );
        
         
        // handle to the C++ instance
        plhs[0] = MatlabConversion::convertPtr2Mat<NICE::GPHIKClassifier>( classifier );
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
        MatlabConversion::destroyObject<NICE::GPHIKClassifier>(prhs[1]);
        return;
    }
    
    // get the class instance pointer from the second input
    // every following function needs the classifier object
    NICE::GPHIKClassifier * classifier = MatlabConversion::convertMat2Ptr<NICE::GPHIKClassifier>(prhs[1]);
    
    
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
            examplesTrain = MatlabConversion::convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix dataTrain;
            dataTrain = MatlabConversion::convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            examplesTrain.resize( dataTrain.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = examplesTrain.begin();
            for (int i = 0; i < (int)dataTrain.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( dataTrain.getRow(i) );
            }            
        }
          
        yMultiTrain = MatlabConversion::convertDoubleVectorToNice(prhs[3]);

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

        uint result;
        NICE::SparseVector scores;
        double uncertainty;        

        if ( mxIsSparse( prhs[2] ) )
        {
            NICE::SparseVector * example;
            example = new NICE::SparseVector ( MatlabConversion::convertSparseVectorToNice( prhs[2] ) );
            classifier->classify ( example,  result, scores, uncertainty );
            
            //----------------- clean up -------------
            delete example;
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( MatlabConversion::convertDoubleVectorToNice(prhs[2]) );
            classifier->classify ( example,  result, scores, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
          
          

          // output
          plhs[0] = mxCreateDoubleScalar( result ); 
                    
          if(nlhs >= 2)
          {
            plhs[1] = MatlabConversion::convertSparseVectorFromNice( scores, true  /*b_adaptIndex*/);
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
            example = new NICE::SparseVector ( MatlabConversion::convertSparseVectorToNice( prhs[2] ) );
            classifier->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( MatlabConversion::convertDoubleVectorToNice(prhs[2]) );
            classifier->predictUncertainty( example, uncertainty );
            
            //----------------- clean up -------------
            delete example;            
        }
        
       

          // output
          plhs[0] = mxCreateDoubleScalar( uncertainty );                    
          return;
    }    
    
    
    // Test - evaluate classifier on whole test set  
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
            dataTest_sparse = MatlabConversion::convertSparseMatrixToNice( prhs[2] );
        }
        else
        {    
            dataTest_dense = MatlabConversion::convertDoubleMatrixToNice(prhs[2]);
        }        

        NICE::Vector yMultiTest;
        yMultiTest = MatlabConversion::convertDoubleVectorToNice(prhs[3]);

        
        // ------------------------------------------
        // ------------- PREPARATION --------------
        // ------------------------------------------   
        
        // determine classes known during training and corresponding mapping
        // thereby allow for non-continous class labels
        std::set< uint > classesKnownTraining = classifier->getKnownClassNumbers();
        
        uint noClassesKnownTraining ( classesKnownTraining.size() );
        std::map< uint, uint > mapClNoToIdxTrain;
        std::set< uint >::const_iterator clTrIt = classesKnownTraining.begin();
        for ( uint i=0; i < noClassesKnownTraining; i++, clTrIt++ )
            mapClNoToIdxTrain.insert ( std::pair< uint, uint > ( *clTrIt, i )  );
        
        // determine classes known during testing and corresponding mapping
        // thereby allow for non-continous class labels
        std::set< uint > classesKnownTest;
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
        std::map< uint, uint> mapClNoToIdxTest;
        std::set< uint >::const_iterator clTestIt = classesKnownTest.begin();
        for ( uint i=0; i < noClassesKnownTest; i++, clTestIt++ )
            mapClNoToIdxTest.insert ( std::pair< uint, uint > ( *clTestIt, i )  );          
        


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
          

            uint result;
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
          plhs[1] = MatlabConversion::convertMatrixFromNice(confusionMatrix);
        if(nlhs >= 3)
          plhs[2] = MatlabConversion::convertMatrixFromNice(scores);
          
          
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
            newExample = new NICE::SparseVector ( MatlabConversion::convertSparseVectorToNice( prhs[2] ) );
        }
        else
        {
            NICE::Vector * example;
            example = new NICE::Vector ( MatlabConversion::convertDoubleVectorToNice(prhs[2]) );
            newExample = new NICE::SparseVector ( *example );
            //----------------- clean up -------------
            delete example;            
        }
        
        newLabel = MatlabConversion::convertMatlabToDouble( prhs[3] );
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = MatlabConversion::convertMatlabToBool( prhs[4] );
          
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
            newExamples = MatlabConversion::convertSparseMatrixToNice( prhs[2] );
        }
        else
        {
            NICE::Matrix newData;
            newData = MatlabConversion::convertDoubleMatrixToNice(prhs[2]);
            
            //----------------- convert data to sparse data structures ---------
            newExamples.resize( newData.rows() );

                    
            std::vector< const NICE::SparseVector *>::iterator exTrainIt = newExamples.begin();
            for (int i = 0; i < (int)newData.rows(); i++, exTrainIt++)
            {
                *exTrainIt =  new NICE::SparseVector( newData.getRow(i) );
            }            
        }
          
        newLabels = MatlabConversion::convertDoubleVectorToNice(prhs[3]);
        
        // setting performOptimizationAfterIncrement is optional
        if ( nrhs > 4 )
        {
          bool performOptimizationAfterIncrement;          
          performOptimizationAfterIncrement = MatlabConversion::convertMatlabToBool( prhs[4] );
          
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
    
  
    
    // store the classifier  to an external file
    if ( !strcmp("store", cmd.c_str() ) || !strcmp("save", cmd.c_str() ) )
    {
        // Check parameters
        if ( nrhs < 3 )
            mexErrMsgTxt("store: no destination given.");        
               
        std::string s_destination = MatlabConversion::convertMatlabToString( prhs[2] );
          
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
               
        std::string s_destination = MatlabConversion::convertMatlabToString( prhs[2] );
        
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
#endif
