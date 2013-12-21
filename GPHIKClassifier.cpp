/** 
* @file GPHIKClassifier.cpp
* @brief Main interface for our GP HIK classifier (similar to the feature pool classifier interface in vislearning) (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 02/01/2012

*/
#include <iostream>

#include "core/basics/numerictools.h"
#include <core/basics/Timer.h>

#include "GPHIKClassifier.h"
#include "gp-hik-core/parameterizedFunctions/PFAbsExp.h"
#include "gp-hik-core/parameterizedFunctions/PFExp.h"
#include "gp-hik-core/parameterizedFunctions/PFMKL.h"

using namespace std;
using namespace NICE;


GPHIKClassifier::GPHIKClassifier( const Config *conf, const string & confSection ) 
{
  //default settings, may be overwritten lateron
  gphyper = NULL;
  pf = NULL;
  confCopy = NULL;
  //just a default value
  uncertaintyPredictionForClassification = false;
  
  if ( conf == NULL )
  {
     fthrow(Exception, "GPHIKClassifier: the config is NULL -- use a default config or the restore-function instaed!");
  }
  else
    this->init(conf, confSection);
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

void GPHIKClassifier::init(const Config *conf, const string & confSection)
{
  std::cerr << " init  method " << std::endl;
  double parameterUpperBound = conf->gD(confSection, "parameter_upper_bound", 5.0 );
  double parameterLowerBound = conf->gD(confSection, "parameter_lower_bound", 1.0 );  

  this->noise = conf->gD(confSection, "noise", 0.01);

  string transform = conf->gS(confSection, "transform", "absexp" );
  
  if (pf == NULL)
  {
    std::cerr << " pf is currently NULL  " << std::endl;
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
  else{
    //we already know the pf from the restore-function
    std::cerr << " pf is already loaded" << std::endl;
  }
  this->confSection = confSection;
  this->verbose = conf->gB(confSection, "verbose", false);
  this->debug = conf->gB(confSection, "debug", false);
  this->uncertaintyPredictionForClassification = conf->gB( confSection, "uncertaintyPredictionForClassification", false );
  
  if (confCopy != conf)
  {  
    std::cerr << " copy config" << std::endl;
    this->confCopy = new Config ( *conf );
    //we do not want to read until end of file for restoring    
    confCopy->setIoUntilEndOfFile(false);    
  }
   
  //how do we approximate the predictive variance for classification uncertainty?
  string s_varianceApproximation = conf->gS(confSection, "varianceApproximation", "approximate_fine"); //default: fine approximative uncertainty prediction
  if ( (s_varianceApproximation.compare("approximate_rough") == 0) || ((s_varianceApproximation.compare("1") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_ROUGH;
  }
  else if ( (s_varianceApproximation.compare("approximate_fine") == 0) || ((s_varianceApproximation.compare("2") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_FINE;
  }
  else if ( (s_varianceApproximation.compare("exact") == 0)  || ((s_varianceApproximation.compare("3") == 0)) )
  {
    this->varianceApproximation = EXACT;
  }
  else
  {
    this->varianceApproximation = NONE;
  } 
  std::cerr << "varianceApproximationStrategy: " << s_varianceApproximation  << std::endl;
}

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores )
{
  double tmpUncertainty;
  this->classify( example, result, scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const NICE::Vector * example,  int & result, SparseVector & scores )
{
  double tmpUncertainty;
  this->classify( example, result, scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores, double & uncertainty )
{
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );
  
  scores.clear();
  
  int classno = gphyper->classify ( *example, scores );

  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
  
  result = scores.maxElement();
   
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

void GPHIKClassifier::classify ( const NICE::Vector * example,  int & result, SparseVector & scores, double & uncertainty )
{
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  scores.clear();
  
  int classno = gphyper->classify ( *example, scores );

  if ( scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << example->size() );
  }
  
  result = scores.maxElement();
  
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
void GPHIKClassifier::train ( const std::vector< NICE::SparseVector *> & examples, const NICE::Vector & labels )
{
  if (verbose)
  {
    std::cerr << "GPHIKClassifier::train" << std::endl;
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
  gphyper->optimize ( labels );
  if (verbose)
    std::cerr << "optimization done" << std::endl;
  
  if ( ( varianceApproximation != NONE ) )
  {
    std::cerr << "now prepare for the uncertainty prediction" << std::endl;
    
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
void GPHIKClassifier::train ( const std::vector< SparseVector *> & examples, std::map<int, NICE::Vector> & binLabels )
{ 
  if (verbose)
    std::cerr << "GPHIKClassifier::train" << std::endl;

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
    std::cerr << "now prepare for the uncertainty prediction" << std::endl;
    
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

void GPHIKClassifier::clear ()
{
  if ( gphyper != NULL )
    delete gphyper;
  gphyper = NULL;
}

GPHIKClassifier *GPHIKClassifier::clone () const
{
  fthrow(Exception, "GPHIKClassifier: clone() not yet implemented" );

  return NULL;
}
  
void GPHIKClassifier::predictUncertainty( const NICE::SparseVector * example, double & uncertainty )
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

void GPHIKClassifier::predictUncertainty( const NICE::Vector * example, double & uncertainty )
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

//---------------------------------------------------------------------
//                           protected methods
//---------------------------------------------------------------------
void GPHIKClassifier::restore ( std::istream & is, int format )
{
  if (is.good())
  {
    std::cerr << "restore GPHIKClassifier" << std::endl;
    is.precision (numeric_limits<double>::digits10 + 1);
    
    string tmp;
    is >> tmp;    
    is >> confSection;
    
    if (pf != NULL)
    {
      delete pf;
    }
    string transform;
    is >> transform;
    if ( transform == "absexp" )
    {
      this->pf = new PFAbsExp ();
    } else if ( transform == "exp" ) {
      this->pf = new PFExp ();
    } else {
      fthrow(Exception, "Transformation type is unknown " << transform);
    }    
    pf->restore(is, format);
    
    std::cerr << "pf restored" << std::endl;
            
    //load every options we determined explicitely
    confCopy->clear();
    //we do not want to read until the end of the file
    confCopy->setIoUntilEndOfFile( false );
    confCopy->restore(is, format);
    
    std::cerr << "conf restored" << std::endl;

    //load every settings as well as default options
    this->init(confCopy, confSection); 
    
    std::cerr << "GPHIK initialized" << std::endl;
  
    //first read things from the config
    if ( gphyper == NULL )
      gphyper = new NICE::FMKGPHyperparameterOptimization();
    
    gphyper->initialize ( confCopy, pf, NULL, confSection );
    
    std::cerr << "gphyper initialized" << std::endl;
    
    //then, load everything that we stored explicitely,
    // including precomputed matrices, LUTs, eigenvalues, ... and all that stuff
    gphyper->restore(is, format);    
    
    std::cerr << "gphyper restored" << std::endl;
  }
  else
  {
    std::cerr << "GPHIKClassifier::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void GPHIKClassifier::store ( std::ostream & os, int format ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  if (os.good())
  {
    os.precision (numeric_limits<double>::digits10 + 1);
    
    os << "confSection: "<<  confSection << std::endl;
    
    os << pf->sayYourName() << std::endl;
    pf->store(os, format);
    
    //we do not want to read until end of file for restoring    
    confCopy->setIoUntilEndOfFile(false);
    confCopy->store(os,format);  
    
    //store the underlying data
    //will be done in gphyper->store(of,format)
    //store the optimized parameter values and all that stuff
    gphyper->store(os, format); 
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

std::set<int> GPHIKClassifier::getKnownClassNumbers ( ) const
{
  if (gphyper == NULL)
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  return gphyper->getKnownClassNumbers();
}