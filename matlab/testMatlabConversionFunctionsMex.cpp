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
#include <core/vector/MatrixT.h>
#include <core/vector/VectorT.h>


// Interface for conversion between Matlab and C objects
#include "gp-hik-core/matlab/classHandleMtoC.h"
#include "gp-hik-core/matlab/ConverterMatlabToNICE.h"
#include "gp-hik-core/matlab/ConverterNICEToMatlab.h"

using namespace std; //C basics
using namespace NICE;  // nice-core

// MAIN MATLAB FUNCTION
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    
    // get the command string specifying what to do
    if (nrhs < 1)
        mexErrMsgTxt("No commands and options passed... Aborting!");        
    
    if( !mxIsChar( prhs[0] ) )
        mexErrMsgTxt("First argument needs to be the command, ie.e, the unit test method to call... Aborting!");
    
    std::string cmd = MatlabConversion::convertMatlabToString( prhs[0] );
    
    // in all other cases, there should be a second input,
    // which the be the class instance handle
    if (nrhs < 2)
    {
        mexErrMsgTxt("Second input should be some kind of matrix variable");
        return;
    }

    if (nlhs < 1)
    {
        mexErrMsgTxt("No return value defined, possible loss of data... Aborting!");
    }

    const mxArray *t_pData = prhs[1];
    
    ////////////////////////////////////////
    //  Check which method to call  //
    ////////////////////////////////////////


    if ( !strcmp("convertInt32", cmd.c_str() ) )
    {

        int t_iTest = MatlabConversion::convertMatlabToInt32( t_pData );
        std::cerr << "convertInt32: " << t_iTest << std::endl;

        // output
        plhs[0] = mxCreateDoubleScalar( t_iTest );

        return;
    }
    else if ( !strcmp("convertLogical", cmd.c_str() ) )
    {

        bool t_iTest = MatlabConversion::convertMatlabToBool( t_pData );
        std::cerr << "convertLogical: " << t_iTest << std::endl;

        // output
        plhs[0] = mxCreateLogicalScalar( t_iTest );

        return;
    }
    else if ( !strcmp("convertDouble", cmd.c_str() ) )
    {

        double t_dTest = MatlabConversion::convertMatlabToDouble( t_pData );
        std::cerr << "convertDouble: " << t_dTest << std::endl;

        // output
        plhs[0] = mxCreateDoubleScalar( t_dTest );

        return;
    }
    /// Matrix/vector functions
    else if ( !strcmp("convertDoubleVector", cmd.c_str() ) )
    {

        NICE::Vector t_vecTest = MatlabConversion::convertDoubleVectorToNice( t_pData );
        std::cerr << "convertDoubleVector: " << t_vecTest << std::endl;

        // output
        plhs[0] = MatlabConversion::convertVectorFromNice( t_vecTest );

        return;
    }
    else if ( !strcmp("convertDoubleMatrix", cmd.c_str() ) )
    {

        NICE::Matrix t_matTest = MatlabConversion::convertDoubleMatrixToNice( t_pData );
        std::cerr << "convertDoubleMatrix: " << t_matTest << std::endl;

        // output
        plhs[0] = MatlabConversion::convertMatrixFromNice( t_matTest );

        return;
    }
    else if ( !strcmp("convertDoubleSparseVector", cmd.c_str() ) )
    {

        NICE::SparseVector t_vecTest = MatlabConversion::convertSparseVectorToNice( t_pData );
	
	NICE::Vector t_fullVector;
	t_vecTest.convertToVectorT( t_fullVector );
        std::cerr << "convertDoubleSparseVector: full version:" << t_fullVector << std::endl;

        // output
        plhs[0] = MatlabConversion::convertVectorFromNice( t_fullVector );

        return;
    }
      
    
    
    // Got here, so command not recognized
    
    std::string errorMsg (cmd.c_str() );
    errorMsg += " -- command not recognized.";
    mexErrMsgTxt( errorMsg.c_str() );

}
