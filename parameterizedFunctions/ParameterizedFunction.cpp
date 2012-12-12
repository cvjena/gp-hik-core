/** 
* @file ParameterizedFunction.cpp
* @brief Simple parameterized multi-dimensional function (Implementation)
* @author Erik Rodner
* @date 01/04/2012

*/
#include <iostream>

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
    is.precision (numeric_limits<double>::digits10 + 1);
    
    string tmp;
    is >> tmp;
    is >> m_parameters;
  }
}

void ParameterizedFunction::store ( std::ostream & os, int format ) const
{
  if (os.good())
  {
    os.precision (numeric_limits<double>::digits10 + 1); 
    os << "m_parameters: " << std::endl << m_parameters << std::endl;
  }
};