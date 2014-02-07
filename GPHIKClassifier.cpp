/** 
* @file GPHIKClassifier.cpp
* @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning) (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/01/2012

*/

// STL includes
#include <iostream>

// NICE-core includes
#include <core/basics/numerictools.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "GPHIKClassifier.h"

using namespace std;
using namespace NICE;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////



/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
GPHIKClassifier::GPHIKClassifier( ) 
{  
  this->b_isTrained = false;  
  this->confSection = "";
  
  this->gphyper = new NICE::FMKGPHyperparameterOptimization();
  
  // in order to be sure about all necessary variables be setup with default values, we
  // run initFromConfig with an empty config
  NICE::Config tmpConfEmpty ;
  this->initFromConfig ( &tmpConfEmpty, this->confSection );  
  

}

GPHIKClassifier::GPHIKClassifier( const Config *conf, const string & s_confSection )
{
  ///////////
  // same code as in empty constructor - duplication can be avoided with C++11 allowing for constructor delegation
  ///////////
  
  this->b_isTrained = false;  
  this->confSection = "";
  
  this->gphyper = new NICE::FMKGPHyperparameterOptimization();
  
  ///////////
  // here comes the new code part different from the empty constructor
  ///////////
  
  this->confSection = s_confSection;  
  
  // if no config file was given, we either restore the classifier from an external file, or run ::init with 
  // an emtpy config (using default values thereby) when calling the train-method
  if ( conf != NULL )
  {
    this->initFromConfig( conf, confSection );
  }
  else
  {
    // if no config was given, we create an empty one
    NICE::Config tmpConfEmpty ;
    this->initFromConfig ( &tmpConfEmpty, this->confSection );      
  }
}

GPHIKClassifier::~GPHIKClassifier()
{
  if ( gphyper != NULL )
    delete gphyper;
}

void GPHIKClassifier::initFromConfig(const Config *conf, const string & s_confSection)
{ 
  this->noise = conf->gD(confSection, "noise", 0.01);

  this->confSection = confSection;
  this->verbose = conf->gB(confSection, "verbose", false);
  this->debug = conf->gB(confSection, "debug", false);
  this->uncertaintyPredictionForClassification = conf->gB( confSection, "uncertaintyPredictionForClassification", false );
  

   
  //how do we approximate the predictive variance for classification uncertainty?
  string s_varianceApproximation = conf->gS(confSection, "varianceApproximation", "approximate_fine"); //default: fine approximative uncertainty prediction
  if ( (s_varianceApproximation.compare("approximate_rough") == 0) || ((s_varianceApproximation.compare("1") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_ROUGH;
    
    //no additional eigenvalue is needed here at all.
    this->gphyper->setNrOfEigenvaluesToConsiderForVarApprox ( 0 );    
  }
  else if ( (s_varianceApproximation.compare("approximate_fine") == 0) || ((s_varianceApproximation.compare("2") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_FINE;    
    
    //security check - compute at least one eigenvalue for this approximation strategy
    this->gphyper->setNrOfEigenvaluesToConsiderForVarApprox ( std::max( conf->gI(confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 ), 1) );
  }
  else if ( (s_varianceApproximation.compare("exact") == 0)  || ((s_varianceApproximation.compare("3") == 0)) )
  {
    this->varianceApproximation = EXACT;
    
    //no additional eigenvalue is needed here at all.
    this->gphyper->setNrOfEigenvaluesToConsiderForVarApprox ( 0 );
  }
  else
  {
    this->varianceApproximation = NONE;
    
    //no additional eigenvalue is needed here at all.
    this->gphyper->setNrOfEigenvaluesToConsiderForVarApprox ( 0 );
  } 
  
  if ( this->verbose )
    std::cerr << "varianceApproximationStrategy: " << s_varianceApproximation  << std::endl;
  
  //NOTE init all member pointer variables here as well
  this->gphyper->initFromConfig ( conf, confSection /*possibly delete the handing of confSection*/);
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// ///////////////////// 

std::set<int> GPHIKClassifier::getKnownClassNumbers ( ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  return gphyper->getKnownClassNumbers();
}


///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores ) const
{
  double tmpUncertainty;
  this->classify( example, result, scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const NICE::Vector * example,  int & result, SparseVector & scores ) const
{
  double tmpUncertainty;
  this->classify( example, result, scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores, double & uncertainty ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );
  
  scores.clear();
  
  result = gphyper->classify ( *example, scores );

  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
  
  if (uncertaintyPredictionForClassification)
  {
    if (varianceApproximation != NONE)
    {
      this->predictUncertainty( example, uncertainty );
    }  
    else
    {
      //do nothing
      uncertainty = std::numeric_limits<double>::max();
    }
  }
  else
  {
    //do nothing
    uncertainty = std::numeric_limits<double>::max();
  }    
}

void GPHIKClassifier::classify ( const NICE::Vector * example,  int & result, SparseVector & scores, double & uncertainty ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  scores.clear();
  
  result = gphyper->classify ( *example, scores );

  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
    
  if (uncertaintyPredictionForClassification)
  {
    if (varianceApproximation != NONE)
    {
      this->predictUncertainty( example, uncertainty );
    }  
    else
    {
      //do nothing
      uncertainty = std::numeric_limits<double>::max();
    }
  }
  else
  {
    //do nothing
    uncertainty = std::numeric_limits<double>::max();
  }  
}

/** training process */
void GPHIKClassifier::train ( const std::vector< const NICE::SparseVector *> & examples, const NICE::Vector & labels )
{
  // security-check: examples and labels have to be of same size
  if ( examples.size() != labels.size() ) 
  {
    fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );  
  }  
  
  if (verbose)
  {
    std::cerr << "GPHIKClassifier::train" << std::endl;
  }
  
  Timer t;
  t.start();
  
  FastMinKernel *fmk = new FastMinKernel ( examples, noise, this->debug );
  gphyper->setFastMinKernel ( fmk ); 
  
  t.stop();
  if (verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  
 

  if (verbose)
    cerr << "Learning ..." << endl;

  // go go go
  gphyper->optimize ( labels );
  if (verbose)
    std::cerr << "optimization done" << std::endl;
  
  if ( ( varianceApproximation != NONE ) )
  {    
    switch (varianceApproximation)    
    {
      case APPROXIMATE_ROUGH:
      {
        gphyper->prepareVarianceApproximationRough();
        break;
      }
      case APPROXIMATE_FINE:
      {
        gphyper->prepareVarianceApproximationFine();
        break;
      }    
      case EXACT:
      {
       //nothing to prepare
        break;
      }
      default:
      {
       //nothing to prepare
      }
    }
  }

  //indicate that we finished training successfully
  this->b_isTrained = true;

  // clean up all examples ??
  if (verbose)
    std::cerr << "Learning finished" << std::endl;
}

/** training process */
void GPHIKClassifier::train ( const std::vector< const NICE::SparseVector *> & examples, std::map<int, NICE::Vector> & binLabels )
{ 
  // security-check: examples and labels have to be of same size
  for ( std::map< int, NICE::Vector >::const_iterator binLabIt = binLabels.begin();
        binLabIt != binLabels.end();
        binLabIt++ 
      )
  {
    if ( examples.size() != binLabIt->second.size() ) 
    {
      fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );  
    }
  }
  
  if (verbose)
    std::cerr << "GPHIKClassifier::train" << std::endl;
 
  Timer t;
  t.start();
  
  FastMinKernel *fmk = new FastMinKernel ( examples, noise, this->debug );
  gphyper->setFastMinKernel ( fmk );  
  
  t.stop();
  if (verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  



  if (verbose)
    cerr << "Learning ..." << endl;
  // go go go
  gphyper->optimize ( binLabels );
  if (verbose)
    std::cerr << "optimization done, now prepare for the uncertainty prediction" << std::endl;
  
  if ( ( varianceApproximation != NONE ) )
  {    
    switch (varianceApproximation)    
    {
      case APPROXIMATE_ROUGH:
      {
        gphyper->prepareVarianceApproximationRough();
        break;
      }
      case APPROXIMATE_FINE:
      {
        gphyper->prepareVarianceApproximationFine();
        break;
      }    
      case EXACT:
      {
       //nothing to prepare
        break;
      }
      default:
      {
       //nothing to prepare
      }
    }
  }

  //indicate that we finished training successfully
  this->b_isTrained = true;

  // clean up all examples ??
  if (verbose)
    std::cerr << "Learning finished" << std::endl;
}

GPHIKClassifier *GPHIKClassifier::clone () const
{
  fthrow(Exception, "GPHIKClassifier: clone() not yet implemented" );

  return NULL;
}
  
void GPHIKClassifier::predictUncertainty( const NICE::SparseVector * example, double & uncertainty ) const
{  
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  //we directly store the predictive variances in the vector, that contains the classification uncertainties lateron to save storage
  switch (varianceApproximation)    
  {
    case APPROXIMATE_ROUGH:
    {
      gphyper->computePredictiveVarianceApproximateRough( *example, uncertainty );
      break;
    }
    case APPROXIMATE_FINE:
    {
      gphyper->computePredictiveVarianceApproximateFine( *example, uncertainty );
      break;
    }    
    case EXACT:
    {
      gphyper->computePredictiveVarianceExact( *example, uncertainty );
      break;
    }
    default:
    {
      fthrow(Exception, "GPHIKClassifier - your settings disabled the variance approximation needed for uncertainty prediction.");
    }
  }
}

void GPHIKClassifier::predictUncertainty( const NICE::Vector * example, double & uncertainty ) const
{  
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  //we directly store the predictive variances in the vector, that contains the classification uncertainties lateron to save storage
  switch (varianceApproximation)    
  {
    case APPROXIMATE_ROUGH:
    {
      gphyper->computePredictiveVarianceApproximateRough( *example, uncertainty );
      break;
    }
    case APPROXIMATE_FINE:
    {
      gphyper->computePredictiveVarianceApproximateFine( *example, uncertainty );
      break;
    }    
    case EXACT:
    {
      gphyper->computePredictiveVarianceExact( *example, uncertainty );
      break;
    }
    default:
    {
      fthrow(Exception, "GPHIKClassifier - your settings disabled the variance approximation needed for uncertainty prediction.");
    }
  }
}

///////////////////// INTERFACE PERSISTENT /////////////////////
// interface specific methods for store and restore
///////////////////// INTERFACE PERSISTENT ///////////////////// 

void GPHIKClassifier::restore ( std::istream & is, int format )
{
  //delete everything we knew so far...
  this->clear();
  
  bool b_restoreVerbose ( false );
#ifdef B_RESTOREVERBOSE
  b_restoreVerbose = true;
#endif  
  
  if ( is.good() )
  {
    if ( b_restoreVerbose ) 
      std::cerr << " restore GPHIKClassifier" << std::endl;
    
    std::string tmp;
    is >> tmp; //class name 
    
    if ( ! this->isStartTag( tmp, "GPHIKClassifier" ) )
    {
      std::cerr << " WARNING - attempt to restore GPHIKClassifier, but start flag " << tmp << " does not match! Aborting... " << std::endl;
      throw;
    }   
    
    if (gphyper != NULL)
    {
      delete gphyper;
      gphyper = NULL;
    }    
    
    is.precision (numeric_limits<double>::digits10 + 1);
    
    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "GPHIKClassifier" ) )
      {
        b_endOfBlock = true;
        continue;
      }      
      
      tmp = this->removeStartTag ( tmp );
      
      if ( b_restoreVerbose )
        std::cerr << " currently restore section " << tmp << " in GPHIKClassifier" << std::endl;
      
      if ( tmp.compare("confSection") == 0 )
      {
        is >> confSection;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("gphyper") == 0 )
      {
        if ( gphyper == NULL )
          gphyper = new NICE::FMKGPHyperparameterOptimization();
        
        //then, load everything that we stored explicitely,
        // including precomputed matrices, LUTs, eigenvalues, ... and all that stuff
        gphyper->restore(is, format);  
          
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }   
      else if ( tmp.compare("b_isTrained") == 0 )
      {
        is >> b_isTrained;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("noise") == 0 )
      {
        is >> noise;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("verbose") == 0 )
      {
        is >> verbose;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("debug") == 0 )
      {
        is >> debug;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("uncertaintyPredictionForClassification") == 0 )
      {
        is >> uncertaintyPredictionForClassification;        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("varianceApproximation") == 0 )
      {
        unsigned int ui_varianceApproximation;
        is >> ui_varianceApproximation;        
        varianceApproximation = static_cast<VarianceApproximation> ( ui_varianceApproximation );
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else
      {
      std::cerr << "WARNING -- unexpected GPHIKClassifier object -- " << tmp << " -- for restoration... aborting" << std::endl;
      throw;
      }
    }
  }
  else
  {
    std::cerr << "GPHIKClassifier::restore -- InStream not initialized - restoring not possible!" << std::endl;
    throw;
  }
}

void GPHIKClassifier::store ( std::ostream & os, int format ) const
{ 
  if (os.good())
  {
    // show starting point
    os << this->createStartTag( "GPHIKClassifier" ) << std::endl;    
    
    os.precision (numeric_limits<double>::digits10 + 1);
    
    os << this->createStartTag( "confSection" ) << std::endl;
    os << confSection << std::endl;
    os << this->createEndTag( "confSection" ) << std::endl; 
   
    os << this->createStartTag( "gphyper" ) << std::endl;
    //store the underlying data
    //will be done in gphyper->store(of,format)
    //store the optimized parameter values and all that stuff
    gphyper->store(os, format);
    os << this->createEndTag( "gphyper" ) << std::endl; 
    
    
    /////////////////////////////////////////////////////////
    // store variables which we previously set via config    
    /////////////////////////////////////////////////////////
    os << this->createStartTag( "b_isTrained" ) << std::endl;
    os << b_isTrained << std::endl;
    os << this->createEndTag( "b_isTrained" ) << std::endl; 
    
    os << this->createStartTag( "noise" ) << std::endl;
    os << noise << std::endl;
    os << this->createEndTag( "noise" ) << std::endl;
    
    
    os << this->createStartTag( "verbose" ) << std::endl;
    os << verbose << std::endl;
    os << this->createEndTag( "verbose" ) << std::endl; 
    
    os << this->createStartTag( "debug" ) << std::endl;
    os << debug << std::endl;
    os << this->createEndTag( "debug" ) << std::endl; 
    
    os << this->createStartTag( "uncertaintyPredictionForClassification" ) << std::endl;
    os << uncertaintyPredictionForClassification << std::endl;
    os << this->createEndTag( "uncertaintyPredictionForClassification" ) << std::endl;
    
    os << this->createStartTag( "varianceApproximation" ) << std::endl;
    os << varianceApproximation << std::endl;
    os << this->createEndTag( "varianceApproximation" ) << std::endl;     
  
    
    
    // done
    os << this->createEndTag( "GPHIKClassifier" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void GPHIKClassifier::clear ()
{
  if ( gphyper != NULL )
  {
    delete gphyper;
    gphyper = NULL;
  }
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void GPHIKClassifier::addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement
			   )
{
  
  if ( ! this->b_isTrained )
  {
    //call train method instead
    std::cerr << "Classifier not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
     
    std::vector< const NICE::SparseVector *> examplesVec;
    examplesVec.push_back ( example );
    
    NICE::Vector labelsVec ( 1 , label );
    
    this->train ( examplesVec, labelsVec );
  }
  else
  {
    this->gphyper->addExample( example, label, performOptimizationAfterIncrement );  
  }
}

void GPHIKClassifier::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
				      const NICE::Vector & newLabels,
				      const bool & performOptimizationAfterIncrement
				    )
{
  //are new examples available? If not, nothing has to be done
  if ( newExamples.size() < 1)
    return;

  if ( ! this->b_isTrained )
  {
    //call train method instead
    std::cerr << "Classifier not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
    
    this->train ( newExamples, newLabels );    
  }
  else
  {
    this->gphyper->addMultipleExamples( newExamples, newLabels, performOptimizationAfterIncrement );     
  }
}