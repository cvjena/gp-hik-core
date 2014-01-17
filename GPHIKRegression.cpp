/** 
* @file GPHIKRegression.cpp
* @brief Main interface for our GP HIK regression implementation (Implementation)
* @author Alexander Freytag
* @date 15-01-2014 (dd-mm-yyyy)
*/

// STL includes
#include <iostream>

// NICE-core includes
#include <core/basics/numerictools.h>
#include <core/basics/Timer.h>

// gp-hik-core includes
#include "GPHIKRegression.h"
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

void GPHIKRegression::init(const Config *conf, const string & s_confSection)
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
  this->uncertaintyPredictionForRegression = confCopy->gB( confSection, "uncertaintyPredictionForRegression", false );
  

   
  //how do we approximate the predictive variance for regression uncertainty?
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
GPHIKRegression::GPHIKRegression( const Config *conf, const string & s_confSection ) 
{
  //default settings, may be overwritten lateron
  gphyper = NULL;
  pf = NULL;
  confCopy = NULL;
  //just a default value
  uncertaintyPredictionForRegression = false;
  
  this->confSection = s_confSection;
  
  // if no config file was given, we either restore the classifier from an external file, or run ::init with 
  // an emtpy config (using default values thereby) when calling the train-method
  if ( conf != NULL )
  {
    this->init(conf, confSection);
  }
}

GPHIKRegression::~GPHIKRegression()
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



///////////////////// ///////////////////// /////////////////////
//                      REGRESSION STUFF
///////////////////// ///////////////////// /////////////////////

void GPHIKRegression::estimate ( const SparseVector * example,  double & result ) const
{
  double tmpUncertainty;
  this->estimate( example, result, tmpUncertainty );
}

void GPHIKRegression::estimate ( const NICE::Vector * example,  double & result ) const
{
  double tmpUncertainty;
  this->estimate( example, result, tmpUncertainty );
}

void GPHIKRegression::estimate ( const SparseVector * example,  double & result, double & uncertainty ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Regression object not trained yet -- aborting!" );
  
  NICE::SparseVector scores;
  scores.clear();
  
  gphyper->classify ( *example, scores );
  
  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
  
  // the internal gphyper object returns for regression a sparse vector with a single entry only
  result = scores.begin()->second;
  
  if (uncertaintyPredictionForRegression)
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

void GPHIKRegression::estimate ( const NICE::Vector * example,  double & result, double & uncertainty ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Regression object not trained yet -- aborting!" );  
  
  NICE::SparseVector scores;
  scores.clear();
  
  gphyper->classify ( *example, scores );

  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
  
  // the internal gphyper object returns for regression a sparse vector with a single entry only  
  result = scores.begin()->second;
    
  if (uncertaintyPredictionForRegression)
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
void GPHIKRegression::train ( const std::vector< const NICE::SparseVector *> & examples, const NICE::Vector & labels )
{
  // security-check: examples and labels have to be of same size
  if ( examples.size() != labels.size() ) 
  {
    fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );  
  }  
  
  if (verbose)
  {
    std::cerr << "GPHIKRegression::train" << std::endl;
  }
  
  //TODO add flag fpr gphyper that only regression is performed, or add a new method for this.
  // thereby, all the binary-label-stuff should be skipped :)
  // also think about the internal stuff, like initialization of alpha vectors and stuff like that .... :(
  // in the worst case, stuff has to be re-written...
  
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
  
  confCopy->sB ( confSection, "b_performRegression", true );
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


GPHIKRegression *GPHIKRegression::clone () const
{
  fthrow(Exception, "GPHIKRegression: clone() not yet implemented" );

  return NULL;
}
  
void GPHIKRegression::predictUncertainty( const NICE::SparseVector * example, double & uncertainty ) const
{  
  if (gphyper == NULL)
     fthrow(Exception, "Regression object not trained yet -- aborting!" );  
  
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
      fthrow(Exception, "GPHIKRegression - your settings disabled the variance approximation needed for uncertainty prediction.");
    }
  }
}

void GPHIKRegression::predictUncertainty( const NICE::Vector * example, double & uncertainty ) const
{  
  if (gphyper == NULL)
     fthrow(Exception, "Regression object not trained yet -- aborting!" );  
  
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
      fthrow(Exception, "GPHIKRegression - your settings disabled the variance approximation needed for uncertainty prediction.");
    }
  }
}

///////////////////// INTERFACE PERSISTENT /////////////////////
// interface specific methods for store and restore
///////////////////// INTERFACE PERSISTENT ///////////////////// 

void GPHIKRegression::restore ( std::istream & is, int format )
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
      std::cerr << " restore GPHIKRegression" << std::endl;
    
    std::string tmp;
    is >> tmp; //class name 
    
    if ( ! this->isStartTag( tmp, "GPHIKRegression" ) )
    {
      std::cerr << " WARNING - attempt to restore GPHIKRegression, but start flag " << tmp << " does not match! Aborting... " << std::endl;
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
      
      if ( this->isEndTag( tmp, "GPHIKRegression" ) )
      {
        b_endOfBlock = true;
        continue;
      }      
      
      tmp = this->removeStartTag ( tmp );
      
      if ( b_restoreVerbose )
        std::cerr << " currently restore section " << tmp << " in GPHIKRegression" << std::endl;
      
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
      std::cerr << "WARNING -- unexpected GPHIKRegression object -- " << tmp << " -- for restoration... aborting" << std::endl;
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
    std::cerr << "GPHIKRegression::restore -- InStream not initialized - restoring not possible!" << std::endl;
    throw;
  }
}

void GPHIKRegression::store ( std::ostream & os, int format ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Regression object not trained yet -- aborting!" );  
  
  if (os.good())
  {
    // show starting point
    os << this->createStartTag( "GPHIKRegression" ) << std::endl;    
    
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
    os << this->createEndTag( "GPHIKRegression" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void GPHIKRegression::clear ()
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

void GPHIKRegression::addExample( const NICE::SparseVector * example, 
			     const double & label, 
			     const bool & performOptimizationAfterIncrement
			   )
{
  
  if ( this->gphyper == NULL )
  {
    //call train method instead
    std::cerr << "Regression object not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
     
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

void GPHIKRegression::addMultipleExamples( const std::vector< const NICE::SparseVector * > & newExamples,
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
    std::cerr << "Regression object not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
    
    this->train ( newExamples, newLabels );    
  }
  else
  {
    this->gphyper->addMultipleExamples( newExamples, newLabels, performOptimizationAfterIncrement );     
  }
}