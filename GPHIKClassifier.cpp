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
     fthrow(Exception, "GPHIKClassifier: the config is NULL -- use a default config and the restore-function instaed!");
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
  double parameterLowerBound = conf->gD(confSection, "parameter_lower_bound", 1.0 );
  double parameterUpperBound = conf->gD(confSection, "parameter_upper_bound", 5.0 );

  if (gphyper == NULL)
    this->gphyper = new FMKGPHyperparameterOptimization;
  this->noise = conf->gD(confSection, "noise", 0.01);

  string transform = conf->gS(confSection, "transform", "absexp" );
  
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
  else{
    //we already know the pf from the restore-function
  }
  this->confSection = confSection;
  this->verbose = conf->gB(confSection, "verbose", false);
  this->debug = conf->gB(confSection, "debug", false);
  this->uncertaintyPredictionForClassification = conf->gB( confSection, "uncertaintyPredictionForClassification", false );
  
  if (confCopy != conf)
  {  
    this->confCopy = new Config ( *conf );
    //we do not want to read until end of file for restoring    
    confCopy->setIoUntilEndOfFile(false);    
  }
   
  //how do we approximate the predictive variance for classification uncertainty?
  string varianceApproximationString = conf->gS(confSection, "varianceApproximation", "approximate_fine"); //default: fine approximative uncertainty prediction
  if ( (varianceApproximationString.compare("approximate_rough") == 0) || ((varianceApproximationString.compare("1") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_ROUGH;
  }
  else if ( (varianceApproximationString.compare("approximate_fine") == 0) || ((varianceApproximationString.compare("2") == 0)) )
  {
    this->varianceApproximation = APPROXIMATE_FINE;
  }
  else if ( (varianceApproximationString.compare("exact") == 0)  || ((varianceApproximationString.compare("3") == 0)) )
  {
    this->varianceApproximation = EXACT;
  }
  else
  {
    this->varianceApproximation = NONE;
  } 
}

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores )
{
  double tmpUncertainty;
  this->classify( example, result, scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const SparseVector * example,  int & result, SparseVector & scores, double & uncertainty )
{
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
      NICE::Vector uncertainties;
      this->predictUncertainty( example, uncertainties );
      uncertainty = uncertainties.Max();
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
    std::cerr << "GPHIKClassifier::train" << std::endl;

  Timer t;
  t.start();
  FastMinKernel *fmk = new FastMinKernel ( examples, noise, this->debug );
  t.stop();
  if (verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  
  
  gphyper = new FMKGPHyperparameterOptimization ( confCopy, pf, fmk, confSection ); 

  if (verbose)
    cerr << "Learning ..." << endl;
  // go go go
  gphyper->optimize ( labels );
  if (verbose)
    std::cerr << "optimization done, now prepare for the uncertainty prediction" << std::endl;
  
  if ( (varianceApproximation == APPROXIMATE_ROUGH) )
  {
    //prepare for variance computation (approximative)
    gphyper->prepareVarianceApproximation();
  }
  //for exact variance computation, we do not have to prepare anything

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
  
  gphyper = new FMKGPHyperparameterOptimization ( confCopy, pf, fmk, confSection ); 

  if (verbose)
    cerr << "Learning ..." << endl;
  // go go go
  gphyper->optimize ( binLabels );
  if (verbose)
    std::cerr << "optimization done, now prepare for the uncertainty prediction" << std::endl;
  
  if ( (varianceApproximation == APPROXIMATE_ROUGH) )
  {
    //prepare for variance computation (approximative)
    gphyper->prepareVarianceApproximation();
  }
  //for exact variance computation, we do not have to prepare anything

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
  
void GPHIKClassifier::predictUncertainty( const NICE::SparseVector * example, NICE::Vector & uncertainties )
{  
  //we directly store the predictive variances in the vector, that contains the classification uncertainties lateron to save storage
  switch (varianceApproximation)    
  {
    case APPROXIMATE_ROUGH:
    {
      gphyper->computePredictiveVarianceApproximateRough( *example, uncertainties );
      break;
    }
    case APPROXIMATE_FINE:
    {
      gphyper->computePredictiveVarianceApproximateFine( *example, uncertainties );
      break;
    }    
    case EXACT:
    {
      gphyper->computePredictiveVarianceExact( *example, uncertainties );
      break;
    }
    default:
    {
//       std::cerr << "No Uncertainty Prediction at all" << std::endl;
      fthrow(Exception, "GPHIKClassifier - your settings disabled the variance approximation needed for uncertainty prediction.");
//       uncertainties.resize( 1 );
//       uncertainties.set( numeric_limits<double>::max() );
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
            
    //load every options we determined explicitely
    confCopy->clear();
    //we do not want to read until the end of the file
    confCopy->setIoUntilEndOfFile( false );
    confCopy->restore(is, format);

    //load every settings as well as default options
    this->init(confCopy, confSection); 
  
    //first read things from the config
    gphyper->initialize ( confCopy, pf );
    
    //then, load everything that we stored explicitely,
    // including precomputed matrices, LUTs, eigenvalues, ... and all that stuff
    gphyper->restore(is, format);      
  }
  else
  {
    std::cerr << "GPHIKClassifier::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void GPHIKClassifier::store ( std::ostream & os, int format ) const
{
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

void GPHIKClassifier::addExample( const NICE::SparseVector * example, const double & label, const bool & performOptimizationAfterIncrement)
{
  gphyper->addExample( *example, label, performOptimizationAfterIncrement );
}

void GPHIKClassifier::addMultipleExamples( const std::vector< const NICE::SparseVector *> & newExamples, const NICE::Vector & newLabels, const bool & performOptimizationAfterIncrement)
{
  //are new examples available? If not, nothing has to be done
  if ( newExamples.size() < 1)
    return;
  
  gphyper->addMultipleExamples( newExamples, newLabels, performOptimizationAfterIncrement );
}