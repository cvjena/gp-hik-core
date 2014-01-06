/** 
* @file ParameterizedFunction.cpp
* @brief Simple parameterized multi-dimensional function (Implementation)
* @author Erik Rodner, Alexander Freytag
* @date 01/04/2012
*/

// STL includes
#include <iostream>

// NICE-core includes
#include "ParameterizedFunction.h"

using namespace NICE;
using namespace std;


ParameterizedFunction::ParameterizedFunction( uint dimension )
{
  m_parameters.resize(dimension);
}
      
void ParameterizedFunction::applyFunctionToDataMatrix ( std::vector< std::vector< double > > & dataMatrix ) const
{
  // REMARK: might be inefficient due to virtual calls
  int iCnt(0);
  for ( vector< vector<double> >::iterator i = dataMatrix.begin() ; i != dataMatrix.end(); i++, iCnt++ )
  {
    uint index = 0;
    for ( vector<double>::iterator j = i->begin(); j != i->end(); j++, index++ )
    {
      *j = f ( iCnt, *j );
    }
  }
}

void ParameterizedFunction::restore ( std::istream & is, int format )
{
  if (is.good())
  {
    is.precision (std::numeric_limits<double>::digits10 + 1); 
    
    std::string tmp;    

    bool b_endOfBlock ( false ) ;
    
    while ( !b_endOfBlock )
    {
      is >> tmp; // start of block 
      
      if ( this->isEndTag( tmp, "ParameterizedFunction" ) )
      {
        b_endOfBlock = true;
        continue;
      }
                  
      
      tmp = this->removeStartTag ( tmp );
      
      if ( tmp.compare("m_parameters") == 0 )
      {
          is >> m_parameters;
      }
      else
      {
	std::cerr << "WARNING -- unexpected ParameterizedFunction object -- " << tmp << " -- for restoration... aborting" << std::endl;
	throw;	
      }
      
      is >> tmp; // end of block 
      tmp = this->removeEndTag ( tmp );      
    }
   }
  else
  {
    std::cerr << "ParameterizedFunction::restore -- InStream not initialized - restoring not possible!" << std::endl;
  }
}

void ParameterizedFunction::store ( std::ostream & os, int format ) const
{
  if (os.good())
  {
    // show starting point
    os << this->createStartTag( "ParameterizedFunction" ) << std::endl;
    
    os.precision (std::numeric_limits<double>::digits10 + 1);
    
    os << this->createStartTag( "m_parameters" ) << std::endl;
    os << m_parameters << std::endl;
    os << this->createEndTag( "m_parameters" ) << std::endl;   
    
    
    // done
    os << this->createEndTag( "ParameterizedFunction" ) << std::endl;   
  }
};