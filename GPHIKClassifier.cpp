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

GPHIKClassifier::GPHIKClassifier( const Config *_conf, 
                                  const string & _confSection 
                                )
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
  
  this->confSection = _confSection;  
  
  // if no config file was given, we either restore the classifier from an external file, or run ::init with 
  // an emtpy config (using default values thereby) when calling the train-method
  if ( _conf != NULL )
  {
    this->initFromConfig( _conf, _confSection );
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
  if ( this->gphyper != NULL )
    delete this->gphyper;
}

void GPHIKClassifier::initFromConfig(const Config *_conf, 
                                     const string & _confSection
                                    )
{ 
  this->d_noise     = _conf->gD( _confSection, "noise", 0.01);

  this->confSection = _confSection;
  this->b_verbose   = _conf->gB( _confSection, "verbose", false);
  this->b_debug     = _conf->gB( _confSection, "debug", false);
  this->uncertaintyPredictionForClassification 
                    = _conf->gB( _confSection, "uncertaintyPredictionForClassification", false );
  

   
  //how do we approximate the predictive variance for classification uncertainty?
  string s_varianceApproximation = _conf->gS(_confSection, "varianceApproximation", "approximate_fine"); //default: fine approximative uncertainty prediction
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
    this->gphyper->setNrOfEigenvaluesToConsiderForVarApprox ( std::max( _conf->gI(_confSection, "nrOfEigenvaluesToConsiderForVarApprox", 1 ), 1) );
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
  
  if ( this->b_verbose )
    std::cerr << "varianceApproximationStrategy: " << s_varianceApproximation  << std::endl;
  
  //NOTE init all member pointer variables here as well
  this->gphyper->initFromConfig ( _conf, _confSection /*possibly delete the handing of confSection*/);
}

///////////////////// ///////////////////// /////////////////////
//                         GET / SET
///////////////////// ///////////////////// ///////////////////// 

std::set<uint> GPHIKClassifier::getKnownClassNumbers ( ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  return gphyper->getKnownClassNumbers();
}


///////////////////// ///////////////////// /////////////////////
//                      CLASSIFIER STUFF
///////////////////// ///////////////////// /////////////////////

void GPHIKClassifier::classify ( const SparseVector * _example,  
                                 uint & _result, 
                                 SparseVector & _scores 
                               ) const
{
  double tmpUncertainty;
  this->classify( _example, _result, _scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const NICE::Vector * _example,  
                                 uint & _result, 
                                 SparseVector & _scores 
                               ) const
{
  double tmpUncertainty;
  this->classify( _example, _result, _scores, tmpUncertainty );
}

void GPHIKClassifier::classify ( const SparseVector * _example,  
                                 uint & _result, 
                                 SparseVector & _scores, 
                                 double & _uncertainty 
                               ) const
{
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );
    
  _scores.clear(); 
  
  if ( this->b_debug )
  {
    std::cerr << "GPHIKClassifier::classify (sparse)" << std::endl;
    _example->store( std::cerr );  
  }
 
  _result = gphyper->classify ( *_example, _scores );

  if ( this->b_debug )
  {  
    _scores.store ( std::cerr ); 
    std::cerr << "_result: " << _result << std::endl;
  }

  if ( _scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << _example->size() );
  }
  
  if ( this->uncertaintyPredictionForClassification )
  {
    if ( this->b_debug )
    {
      std::cerr << "GPHIKClassifier::classify -- uncertaintyPredictionForClassification is true"  << std::endl;
    }
    
    if ( this->varianceApproximation != NONE)
    {
      this->predictUncertainty( _example, _uncertainty );
    }  
    else
    {
//       //do nothing
      _uncertainty = std::numeric_limits<double>::max();
    }
  }
  else
  {
    if ( this->b_debug )
    {
      std::cerr << "GPHIKClassifier::classify -- uncertaintyPredictionForClassification is false"  << std::endl;
    }    
    
    //do nothing
    _uncertainty = std::numeric_limits<double>::max();
  }    
}

void GPHIKClassifier::classify ( const NICE::Vector * _example,  
                                 uint & _result, 
                                 SparseVector & _scores, 
                                 double & _uncertainty 
                               ) const
{
  
  if ( ! this->b_isTrained )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  _scores.clear();
  
  if ( this->b_debug )
  {  
    std::cerr << "GPHIKClassifier::classify (non-sparse)" << std::endl;
    std::cerr << *_example << std::endl;
  }
    
  _result = this->gphyper->classify ( *_example, _scores );
  
  if ( this->b_debug )
  {  
    std::cerr << "GPHIKClassifier::classify (non-sparse) -- classification done " << std::endl;
  }
 

  if ( _scores.size() == 0 ) {
    fthrow(Exception, "Zero scores, something is likely to be wrong here: svec.size() = " << _example->size() );
  }
    
  if ( this->uncertaintyPredictionForClassification )
  {
    if ( this->varianceApproximation != NONE)
    {
      this->predictUncertainty( _example, _uncertainty );
    }  
    else
    {
      //do nothing
      _uncertainty = std::numeric_limits<double>::max();
    }
  }
  else
  {
    //do nothing
    _uncertainty = std::numeric_limits<double>::max();
  }  
}

/** training process */
void GPHIKClassifier::train ( const std::vector< const NICE::SparseVector *> & _examples, 
                              const NICE::Vector & _labels 
                            )
{
  
  //FIXME add check whether the classifier has been trained already. if so, discard all previous results.
    
  // security-check: examples and labels have to be of same size
  if ( _examples.size() != _labels.size() ) 
  {
    fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );  
  }  
  
  if (b_verbose)
  {
    std::cerr << "GPHIKClassifier::train" << std::endl;
  }
  
  Timer t;
  t.start();
  
  FastMinKernel *fmk = new FastMinKernel ( _examples, d_noise, this->b_debug );

  this->gphyper->setFastMinKernel ( fmk ); 
  
  t.stop();
  if (b_verbose)
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  
 

  if (b_verbose)
    std::cerr << "Learning ..." << endl;

  // go go go
  this->gphyper->optimize ( _labels );
  if (b_verbose)
    std::cerr << "optimization done" << std::endl;
  
  if ( ( this->varianceApproximation != NONE ) )
  {    
    switch ( this->varianceApproximation )    
    {
      case APPROXIMATE_ROUGH:
      {
        this->gphyper->prepareVarianceApproximationRough();
        break;
      }
      case APPROXIMATE_FINE:
      {
        this->gphyper->prepareVarianceApproximationFine();
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
  if (b_verbose)
    std::cerr << "Learning finished" << std::endl;
}

/** training process */
void GPHIKClassifier::train ( const std::vector< const NICE::SparseVector *> & _examples, 
                              std::map<uint, NICE::Vector> & _binLabels 
                            )
{ 
  // security-check: examples and labels have to be of same size
  for ( std::map< uint, NICE::Vector >::const_iterator binLabIt = _binLabels.begin();
        binLabIt != _binLabels.end();
        binLabIt++ 
      )
  {
    if ( _examples.size() != binLabIt->second.size() ) 
    {
      fthrow(Exception, "Given examples do not match label vector in size -- aborting!" );  
    }
  }
  
  if ( this->b_verbose )
    std::cerr << "GPHIKClassifier::train" << std::endl;
 
  Timer t;
  t.start();
  
  FastMinKernel *fmk = new FastMinKernel ( _examples, d_noise, this->b_debug );
  this->gphyper->setFastMinKernel ( fmk );  
  
  t.stop();
  if ( this->b_verbose )
    std::cerr << "Time used for setting up the fmk object: " << t.getLast() << std::endl;  



  if ( this->b_verbose )
    std::cerr << "Learning ..." << std::endl;
  
  // go go go
  this->gphyper->optimize ( _binLabels );
  
  if ( this->b_verbose )
    std::cerr << "optimization done, now prepare for the uncertainty prediction" << std::endl;
  
  if ( ( this->varianceApproximation != NONE ) )
  {    
    switch ( this->varianceApproximation )    
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
  if ( this->b_verbose )
    std::cerr << "Learning finished" << std::endl;
}

GPHIKClassifier *GPHIKClassifier::clone () const
{
  fthrow(Exception, "GPHIKClassifier: clone() not yet implemented" );

  return NULL;
}
  
void GPHIKClassifier::predictUncertainty( const NICE::SparseVector * _example, 
                                          double & _uncertainty 
                                        ) const
{  
  if ( this->gphyper == NULL )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  //we directly store the predictive variances in the vector, that contains the classification uncertainties lateron to save storage
  switch ( this->varianceApproximation )    
  {
    case APPROXIMATE_ROUGH:
    {
      this->gphyper->computePredictiveVarianceApproximateRough( *_example, _uncertainty );
      break;
    }
    case APPROXIMATE_FINE:
    {
      this->gphyper->computePredictiveVarianceApproximateFine( *_example, _uncertainty );
      break;
    }    
    case EXACT:
    {
      this->gphyper->computePredictiveVarianceExact( *_example, _uncertainty );
      break;
    }
    default:
    {
      fthrow(Exception, "GPHIKClassifier - your settings disabled the variance approximation needed for uncertainty prediction.");
    }
  }
}

void GPHIKClassifier::predictUncertainty( const NICE::Vector * _example, 
                                          double & _uncertainty 
                                        ) const
{  
  if ( this->gphyper == NULL )
     fthrow(Exception, "Classifier not trained yet -- aborting!" );  
  
  //we directly store the predictive variances in the vector, that contains the classification uncertainties lateron to save storage
  switch ( this->varianceApproximation )    
  {
    case APPROXIMATE_ROUGH:
    {
      this->gphyper->computePredictiveVarianceApproximateRough( *_example, _uncertainty );
      break;
    }
    case APPROXIMATE_FINE:
    {
      this->gphyper->computePredictiveVarianceApproximateFine( *_example, _uncertainty );
      break;
    }    
    case EXACT:
    {
      this->gphyper->computePredictiveVarianceExact( *_example, _uncertainty );
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

void GPHIKClassifier::restore ( std::istream & _is, 
                                int _format 
                              )
{
  //delete everything we knew so far...
  this->clear();
  
  bool b_restoreVerbose ( false );
#ifdef B_RESTOREVERBOSE
  b_restoreVerbose = true;
#endif  
  
  if ( _is.good() )
  {
    if ( b_restoreVerbose ) 
      std::cerr << " restore GPHIKClassifier" << std::endl;
    
    std::string tmp;
    _is >> tmp; //class name 
    
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
    
    _is.precision (numeric_limits<double>::digits10 + 1);
    
    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      _is >> tmp; // start of block 
      
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
        _is >> confSection;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("gphyper") == 0 )
      {
        if ( this->gphyper == NULL )
          this->gphyper = new NICE::FMKGPHyperparameterOptimization();
        
        //then, load everything that we stored explicitely,
        // including precomputed matrices, LUTs, eigenvalues, ... and all that stuff
        this->gphyper->restore( _is, _format );  
          
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }   
      else if ( tmp.compare("b_isTrained") == 0 )
      {
        _is >> b_isTrained;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("d_noise") == 0 )
      {
        _is >> d_noise;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("b_verbose") == 0 )
      {
        _is >> b_verbose;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("b_debug") == 0 )
      {
        _is >> b_debug;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }      
      else if ( tmp.compare("uncertaintyPredictionForClassification") == 0 )
      {
        _is >> uncertaintyPredictionForClassification;        
        _is >> tmp; // end of block 
        tmp = this->removeEndTag ( tmp );
      }
      else if ( tmp.compare("varianceApproximation") == 0 )
      {
        unsigned int ui_varianceApproximation;
        _is >> ui_varianceApproximation;        
        varianceApproximation = static_cast<VarianceApproximation> ( ui_varianceApproximation );
        _is >> tmp; // end of block 
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

void GPHIKClassifier::store ( std::ostream & _os, 
                              int _format 
                            ) const
{ 
  if ( _os.good() )
  {
    // show starting point
    _os << this->createStartTag( "GPHIKClassifier" ) << std::endl;    
    
    _os.precision (numeric_limits<double>::digits10 + 1);
    
    _os << this->createStartTag( "confSection" ) << std::endl;
    _os << confSection << std::endl;
    _os << this->createEndTag( "confSection" ) << std::endl; 
   
    _os << this->createStartTag( "gphyper" ) << std::endl;
    //store the underlying data
    //will be done in gphyper->store(of,format)
    //store the optimized parameter values and all that stuff
    this->gphyper->store( _os, _format );
    _os << this->createEndTag( "gphyper" ) << std::endl; 
    
    
    /////////////////////////////////////////////////////////
    // store variables which we previously set via config    
    /////////////////////////////////////////////////////////
    _os << this->createStartTag( "b_isTrained" ) << std::endl;
    _os << b_isTrained << std::endl;
    _os << this->createEndTag( "b_isTrained" ) << std::endl; 
    
    _os << this->createStartTag( "d_noise" ) << std::endl;
    _os << d_noise << std::endl;
    _os << this->createEndTag( "d_noise" ) << std::endl;
    
    
    _os << this->createStartTag( "b_verbose" ) << std::endl;
    _os << b_verbose << std::endl;
    _os << this->createEndTag( "b_verbose" ) << std::endl; 
    
    _os << this->createStartTag( "b_debug" ) << std::endl;
    _os << b_debug << std::endl;
    _os << this->createEndTag( "b_debug" ) << std::endl; 
    
    _os << this->createStartTag( "uncertaintyPredictionForClassification" ) << std::endl;
    _os << uncertaintyPredictionForClassification << std::endl;
    _os << this->createEndTag( "uncertaintyPredictionForClassification" ) << std::endl;
    
    _os << this->createStartTag( "varianceApproximation" ) << std::endl;
    _os << varianceApproximation << std::endl;
    _os << this->createEndTag( "varianceApproximation" ) << std::endl;     
  
    
    
    // done
    _os << this->createEndTag( "GPHIKClassifier" ) << std::endl;    
  }
  else
  {
    std::cerr << "OutStream not initialized - storing not possible!" << std::endl;
  }
}

void GPHIKClassifier::clear ()
{
  if ( this->gphyper != NULL )
  {
    delete this->gphyper;
    this->gphyper = NULL;
  }
}

///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////
// interface specific methods for incremental extensions
///////////////////// INTERFACE ONLINE LEARNABLE /////////////////////

void GPHIKClassifier::addExample( const NICE::SparseVector * _example, 
                                  const double & _label, 
                                  const bool & _performOptimizationAfterIncrement
                                )
{
  
  if ( ! this->b_isTrained )
  {
    //call train method instead
    std::cerr << "Classifier not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
     
    std::vector< const NICE::SparseVector *> examplesVec;
    examplesVec.push_back ( _example );
    
    NICE::Vector labelsVec ( 1 , _label );
    
    this->train ( examplesVec, labelsVec );
  }
  else
  {
    this->gphyper->addExample( _example, _label, _performOptimizationAfterIncrement );  
  }
}

void GPHIKClassifier::addMultipleExamples( const std::vector< const NICE::SparseVector * > & _newExamples,
                                           const NICE::Vector & _newLabels,
                                           const bool & _performOptimizationAfterIncrement
                                         )
{
  //are new examples available? If not, nothing has to be done
  if ( _newExamples.size() < 1)
    return;

  if ( ! this->b_isTrained )
  {
    //call train method instead
    std::cerr << "Classifier not initially trained yet -- run initial training instead of incremental extension!"  << std::endl;
    
    this->train ( _newExamples, _newLabels );    
  }
  else
  {
    this->gphyper->addMultipleExamples( _newExamples, _newLabels, _performOptimizationAfterIncrement );     
  }
}
