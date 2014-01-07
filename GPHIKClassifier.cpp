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
#include "gp-hik-core/parameterizedFunctions/PFAbsExp.h"
#include "gp-hik-core/parameterizedFunctions/PFExp.h"
#include "gp-hik-core/parameterizedFunctions/PFMKL.h"

using namespace std;
using namespace NICE;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PROTECTED METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

void GPHIKClassifier::init(const Config *conf, const string & s_confSection)
{
  //copy the given config to have it accessible lateron
  if ( this->confCopy != conf )
  {
    if ( this->confCopy != NULL )
      delete this->confCopy;
    
    this->confCopy = new Config ( *conf );
    //we do not want to read until end of file for restoring    
    this->confCopy->setIoUntilEndOfFile(false);        
  }
  

  
  double parameterUpperBound = confCopy->gD(confSection, "parameter_upper_bound", 5.0 );
  double parameterLowerBound = confCopy->gD(confSection, "parameter_lower_bound", 1.0 );  

  this->noise = confCopy->gD(confSection, "noise", 0.01);

  string transform = confCopy->gS(confSection, "transform", "absexp" );
  
  if (pf == NULL)
  {
    if ( transform == "absexp" )
    {
      this->pf = new PFAbsExp( 1.0, parameterLowerBound, parameterUpperBound );
    } else if ( transform == "exp" ) {
      this->pf = new PFExp( 1.0, parameterLowerBound, parameterUpperBound );
    }else if ( transform == "MKL" ) {
      //TODO generic, please :) load from a separate file or something like this!
      std::set<int> steps; steps.insert(4000); steps.insert(6000); //specific for VISAPP
      this->pf = new PFMKL( steps, parameterLowerBound, parameterUpperBound );
    } else {
      fthrow(Exception, "Transformation type is unknown " << transform);
    }
  }
  else
  {
    //we already know the pf from the restore-function
  }
  this->confSection = confSection;
  this->verbose = confCopy->gB(confSection, "verbose", false);
  this->debug = confCopy->gB(confSection, "debug", false);
  this->uncertaintyPredictionForClassification = confCopy->gB( confSection, "uncertaintyPredictionForClassification", false );
  

   
  //how do we approximate the predictive variance for classification uncertainty?
  string s_varianceApproximation = confCopy->gS(confSection, "varianceApproximation", "approximate_fine"); //default: fine approximative uncertainty prediction
  if ( (s_varianceApproximation.compare("approximate_rough") == 0) || ((s_varianceApproximation.compare("1") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_ROUGH;
    
    //no additional eigenvalue is needed here at all.
    this->confCopy->sI ( confSection, "nrOfEigenvaluesToConsiderForVarApprox", 0 );
  }
  else if ( (s_varianceApproximation.compare("approximate_fine") == 0) || ((s_varianceApproximation.compare("2") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_FINE;
    
    //security check - compute at least one eigenvalue for this approximation strategy
    this->confCopy->sI ( confSection, "nrOfEigenvaluesToConsiderForVarApprox", std::max( confCopy->gI(confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 ), 1) );
  }
  else if ( (s_varianceApproximation.compare("exact") == 0)  || ((s_varianceApproximation.compare("3") == 0)) )
  {
    this->varianceApproximation = EXACT;
    
    //no additional eigenvalue is needed here at all.
    this->confCopy->sI ( confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 );    
  }
  else
  {
    this->varianceApproximation = NONE;
    
    //no additional eigenvalue is needed here at all.
    this->confCopy->sI ( confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 );
  } 
  
  if ( this->verbose )
    std::cerr << "varianceApproximationStrategy: " << s_varianceApproximation  << std::endl;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
//                 PUBLIC METHODS
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
GPHIKClassifier::GPHIKClassifier( const Config *conf, const string & s_confSection ) 
{
  //default settings, may be overwritten lateron
  gphyper = NULL;
  pf = NULL;
  confCopy = NULL;
  //just a default value
  uncertaintyPredictionForClassification = false;
  
  this->confSection = s_confSection;
  
  // if no config file was given, we either restore the classifier from an external file, or run ::init with 
  // an emtpy config (using default values thereby) when calling the train-method
  if ( conf != NULL )
  {
    this->init(conf, confSection);
  }
}

GPHIKClassifier::~GPHIKClassifier()
{
  if ( gphyper != NULL )
    delete gphyper;
  
  if (pf != NULL)
    delete pf;

  if ( confCopy != NULL )
    delete confCopy;
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// ///////////////////// 

std::set<int> GPHIKClassifier::getKnownClassNumbers ( ) const
{
  if (gphyper == NULL)
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
  if (gphyper == NULL)
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
  if (gphyper == NULL)
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
  if (verbose)
  {
    std::cerr << "GPHIKClassifier::train" << std::endl;
  }
  
  if ( this->confCopy == NULL )
  {
    std::cerr << "WARNING -- No config used so far, initialize values with empty config file now..." << std::endl;
    NICE::Config tmpConfEmpty ;
    this->init ( &tmpConfEmpty, this->confSection );
  }

  Timer t;
  t.start();
  FastMinKernel *fmk = new FastMinKernel ( examples, noise, this->debug );
  
  t.stop();
  if (verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  
  
  if (gphyper != NULL)
     delete gphyper;
  
  
  if ( ( varianceApproximation != APPROXIMATE_FINE) )
    confCopy->sI ( confSection, "nrOfEigenvaluesToConsiderForVarApprox", 0);
  
  gphyper = new FMKGPHyperparameterOptimization ( confCopy, pf, fmk, confSection ); 

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


  // clean up all examples ??
  if (verbose)
    std::cerr << "Learning finished" << std::endl;
}

/** training process */
void GPHIKClassifier::train ( const std::vector< const NICE::SparseVector *> & examples, std::map<int, NICE::Vector> & binLabels )
{ 
  if (verbose)
    std::cerr << "GPHIKClassifier::train" << std::endl;
  
  if ( this->confCopy == NULL )
  {
    std::cerr << "WARNING -- No config used so far, initialize values with empty config file now..." << std::endl;
    NICE::Config tmpConfEmpty ;
    this->init ( &tmpConfEmpty, this->confSection );
  }  

  Timer t;
  t.start();
  FastMinKernel *fmk = new FastMinKernel ( examples, noise, this->debug );
  t.stop();
  if (verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  
  
  if (gphyper != NULL)
     delete gphyper;
  gphyper = new FMKGPHyperparameterOptimization ( confCopy, pf, fmk, confSection ); 

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
        std::cerr << "predict uncertainty fine" << std::endl;
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
//       uncertainty = numeric_limits<double>::max();
//       break;
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
//       uncertainty = numeric_limits<double>::max();
//       break;
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
    
    if (pf != NULL)
    {
      delete pf;
      pf = NULL;
    }
    if ( confCopy != NULL )
    {
      delete confCopy;
      confCopy = NULL;
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
      else if ( tmp.compare("pf") == 0 )
      {
      
        is >> tmp; // start of block 
        if ( this->isEndTag( tmp, "pf" ) )
        {
          std::cerr << " ParameterizedFunction object can not be restored. Aborting..." << std::endl;
          throw;
        } 
        
        std::string transform = this->removeStartTag ( tmp );
        

        if ( transform == "PFAbsExp" )
        {
          this->pf = new PFAbsExp ();
        } else if ( transform == "PFExp" ) {
          this->pf = new PFExp ();
        } else {
          fthrow(Exception, "Transformation type is unknown " << transform);
        }
        
        pf->restore(is, format);
        
        is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      } 
      else if ( tmp.compare("ConfigCopy") == 0 )
      {
        // possibly obsolete safety checks
        if ( confCopy == NULL )
          confCopy = new Config;
        confCopy->clear();
        
        
        //we do not want to read until the end of the file
        confCopy->setIoUntilEndOfFile( false );
        //load every options we determined explicitely
        confCopy->restore(is, format);
        
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
      else
      {
      std::cerr << "WARNING -- unexpected GPHIKClassifier object -- " << tmp << " -- for restoration... aborting" << std::endl;
      throw;
      }
    }

    //load every settings as well as default options
    std::cerr << "run this->init" << std::endl;
    this->init(confCopy, confSection);    
    std::cerr << "run gphyper->initialize" << std::endl;
    gphyper->initialize ( confCopy, pf, NULL, confSection );
  }
  else
  {
    std::cerr << "GPHIKClassifier::restore -- InStream not initialized - restoring not possible!" << std::endl;
    throw;
  }
}

void GPHIKClassifier::store ( std::ostream & os, int format ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  if (os.good())
  {
    // show starting point
    os << this->createStartTag( "GPHIKClassifier" ) << std::endl;    
    
    os.precision (numeric_limits<double>::digits10 + 1);
    
    os << this->createStartTag( "confSection" ) << std::endl;
    os << confSection << std::endl;
    os << this->createEndTag( "confSection" ) << std::endl; 
    
    os << this->createStartTag( "pf" ) << std::endl;
    pf->store(os, format);
    os << this->createEndTag( "pf" ) << std::endl; 

    os << this->createStartTag( "ConfigCopy" ) << std::endl;
    //we do not want to read until end of file for restoring    
    confCopy->setIoUntilEndOfFile(false);
    confCopy->store(os,format);
    os << this->createEndTag( "ConfigCopy" ) << std::endl; 
    
    os << this->createStartTag( "gphyper" ) << std::endl;
    //store the underlying data
    //will be done in gphyper->store(of,format)
    //store the optimized parameter values and all that stuff
    gphyper->store(os, format);
    os << this->createEndTag( "gphyper" ) << std::endl;   
    
    
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
  
  if (pf != NULL)
  {
    delete pf;
    pf = NULL;
  }

  if ( confCopy != NULL )
  {
    delete confCopy; 
    confCopy = NULL;
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
  
  if ( this->gphyper == NULL )
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

  if ( this->gphyper == NULL )
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