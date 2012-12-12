/** 
* @file LogDetApprox.h
* @brief LogDet Approximations (Interface - abstract)
* @author Alexander Freytag
* @date 05-01-2012 (dd-mm-yyyy)
*/
#ifndef LOGDETAPPROXINCLUDE
#define LOGDETAPPROXINCLUDE

#include "core/vector/MatrixT.h"

namespace NICE {

 /** 
 * @class LogDetApprox
 * @brief LogDet Approximations (abstract interface)
 * @author Alexander Freytag
 */
 
	class LogDetApprox 
	{

		protected:
			
		public:
			LogDetApprox(){};
			~LogDetApprox(){};
			
			virtual double getLogDetApproximation(const NICE::Matrix & A)=0;
	};
} //namespace

#endif