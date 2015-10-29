#ifdef NICE_USELIB_MEX
/**
* @file GPHIKRawClassifierMex.cpp
* @author Alexander Freytag
* @date 21-09-2015 (dd-mm-yyyy)
* @brief Matlab-Interface of our GPHIKRawClassifier, allowing for training and classification without more advanced methods.
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
#include "gp-hik-core/GPHIKRawClassifier.h"


// Interface for conversion between Matlab and C objects
#include "gp-hik-core/matlab/classHandleMtoC.h"
#include "gp-hik-core/matlab/ConverterMatlabToNICE.h"
#include "gp-hik-core/matlab/ConverterNICEToMatlab.h"

using namespace std; //C basics
using namespace NICE;  // nice-core


NICE::Config parseParametersGPHIKRawClassifier(const mxArray *prhs[], int nrhs)
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
    if( (variable == "verbose") ||
        (variable == "debug") ||
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
          conf.sB("GPHIKRawClassifier", variable, true);
        else
          conf.sB("GPHIKRawClassifier", variable, false);
      }
      else if ( mxIsLogical( prhs[i+1] ) )
      {
        bool value = MatlabConversion::convertMatlabToBool( prhs[i+1] );
        conf.sB("GPHIKRawClassifier", variable, value);
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

    /////////////////////////////////////////
    // READ STRICT POSITIVE INT VARIABLES
    /////////////////////////////////////////
    if ( (variable == "num_bins") ||
         (variable == "ils_max_iterations" )
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
        conf.sI("GPHIKRawClassifier", variable, (int) value);
      }
      else if ( mxIsInt32( prhs[i+1] ) )
      {
        int value = MatlabConversion::convertMatlabToInt32(prhs[i+1]);
        if( value < 1 )
        {
          std::string errorMsg = "Expected parameter value larger than 0 for \'" +  variable + "\'.";
          mexErrMsgIdAndTxt( "mexnice:error", errorMsg.c_str() );
        }
        conf.sI("GPHIKRawClassifier", variable, value);
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


    /////////////////////////////////////////
    // READ POSITIVE DOUBLE VARIABLES
    /////////////////////////////////////////
    if ( (variable == "f_tolerance") ||
         (variable == "ils_min_delta") ||
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
        conf.sD("GPHIKRawClassifier", variable, value);
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
        conf.sS("GPHIKRawClassifier", variable, value);
    }

    if(variable == "s_quantType")
    {
      string value = MatlabConversion::convertMatlabToString( prhs[i+1] );
      if( value != "1d-aequi-0-1" && value != "1d-aequi-0-max" && value != "nd-aequi-0-max" )
        mexErrMsgIdAndTxt("mexnice:error","Unexpected parameter value for \'s_quantType\'. \'1d-aequi-0-1\' , \'1d-aequi-0-max\' or \'nd-aequi-0-max\' expected.");
        conf.sS("GPHIKRawClassifier", variable, value);
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
        NICE::Config conf = parseParametersGPHIKRawClassifier(prhs+1,nrhs-1);

        // create class instance
        NICE::GPHIKRawClassifier * classifier = new NICE::GPHIKRawClassifier ( &conf, "GPHIKRawClassifier" /*sectionName in config*/ );


        // handle to the C++ instance
        plhs[0] = MatlabConversion::convertPtr2Mat<NICE::GPHIKRawClassifier>( classifier );
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
        MatlabConversion::destroyObject<NICE::GPHIKRawClassifier>(prhs[1]);
        return;
    }

    // get the class instance pointer from the second input
    // every following function needs the classifier object
    NICE::GPHIKRawClassifier * classifier = MatlabConversion::convertMat2Ptr<NICE::GPHIKRawClassifier>(prhs[1]);


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

        if ( mxIsSparse( prhs[2] ) )
        {
          if ( MatlabConversion::isSparseDataAMatrix( prhs[2] ) )
          {
            //----------------- conversion -------------
            std::vector< const NICE::SparseVector *> examplesTest;
            examplesTest = MatlabConversion::convertSparseMatrixToNice( prhs[2] );
            
            //----------------- classification -------------
            NICE::Vector results;
            NICE::Matrix scores;            
            classifier->classify ( examplesTest,  results, scores );
            
            //----------------- clean up -------------
            for ( std::vector< const NICE::SparseVector *>::iterator exIt = examplesTest.begin();
                 exIt != examplesTest.end();
                 exIt++
            )
            {
              delete *exIt;
            }
            
            //----------------- output -------------
            plhs[0] = MatlabConversion::convertVectorFromNice( results );

            if(nlhs >= 2)
            {
              plhs[1] = MatlabConversion::convertMatrixFromNice( scores );
            }
            return;            
          }
          else
          { 
            //----------------- conversion -------------
            NICE::SparseVector * example;
            example = new NICE::SparseVector ( MatlabConversion::convertSparseVectorToNice( prhs[2] ) );
            
            //----------------- classification -------------
            uint result;
            NICE::SparseVector scores;
            classifier->classify ( example,  result, scores );

            //----------------- clean up -------------
            delete example;
            
            //----------------- output -------------
            plhs[0] = mxCreateDoubleScalar( result );

            if(nlhs >= 2)
            {
              plhs[1] = MatlabConversion::convertSparseVectorFromNice( scores, true  /*b_adaptIndex*/);
            }
            return;            
          }
        }
        else
        {
            //----------------- conversion -------------          
            NICE::Vector * example;
            example = new NICE::Vector ( MatlabConversion::convertDoubleVectorToNice(prhs[2]) );
            NICE::SparseVector * svec  = new NICE::SparseVector( *example );
            delete example;

            //----------------- classification -------------
            uint result;
            NICE::SparseVector scores;            
            classifier->classify ( svec,  result, scores );

            //----------------- clean up -------------
            delete svec;
            
            
            //----------------- output -------------
            plhs[0] = mxCreateDoubleScalar( result );

            if(nlhs >= 2)
            {
              plhs[1] = MatlabConversion::convertSparseVectorFromNice( scores, true  /*b_adaptIndex*/);
            }
            return;
        }
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
                NICE::SparseVector * svec = new NICE::SparseVector ( example );
              // and classify
              t.start();
              classifier->classify( svec, result, exampleScoresSparse );
              t.stop();
              testTime += t.getLast();
              delete svec;
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

   // not supported here

    ///////////////////// INTERFACE PERSISTENT /////////////////////
    // interface specific methods for store and restore
    ///////////////////// INTERFACE PERSISTENT /////////////////////

    // not supported here



    // Got here, so command not recognized

    std::string errorMsg (cmd.c_str() );
    errorMsg += " -- command not recognized.";
    mexErrMsgTxt( errorMsg.c_str() );

}
#endif

