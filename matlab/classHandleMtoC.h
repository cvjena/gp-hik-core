/** 
* @file classHandleMtoC.h
* @brief Generic class to pass C++ objects to matlab (Interface and inline implementations)
* @author Alexander Freytag
* @date 19-12-2013 (dd-mm-yyyy)

*/
#ifndef _NICE_CLASSHANDLEMTOCINCLUDE
#define _NICE_CLASSHANDLEMTOCINCLUDE

#include "mex.h"
#include <stdint.h>
#include <iostream>
#include <string>
#include <cstring>
#include <typeinfo>

#define CLASS_HANDLE_SIGNATURE 0xFF00F0A3

  /** 
  * @class FMKGPHyperparameterOptimization
  * @brief Generic class to pass C++ objects to matlab
  * @author Alexander Freytag
  */
template<class objectClass> class ClassHandle
{
  private:
      //!
      uint32_t i_mySignature;
      
      //! typeid.name of object class we refere to
      std::string s_myName;
      
      //! the actual pointer to our C++ object
      objectClass* p_myPtr;  
  
  public:
    
    /**
    * @brief standard constructor
    *
    * @param ptr pointer to the c++ object
    */    
      ClassHandle ( objectClass* p_ptr ) : p_myPtr(p_ptr), s_myName( typeid(objectClass).name() )
      {
        i_mySignature = CLASS_HANDLE_SIGNATURE;
      }
      
    /**
    * @brief standard destructor
    */        
      ~ClassHandle()
      {
          // reset internal variables
          i_mySignature = 0;
          
          // clearn up data
          delete p_myPtr;
      }
      
    /**
    * @brief check whether the class handle was initialized properly, i.e., we point to an actual object
    */       
      bool isValid()
      { 
        return ( (i_mySignature == CLASS_HANDLE_SIGNATURE) && !strcmp( s_myName.c_str(), typeid(objectClass).name() )   );
      }
      
    /**
    * @brief get the pointer to the actual object
    */       
      objectClass * getPtrToObject()
      { 
        return p_myPtr;
      }


};


////////////////////////////////////////////
//           conversion methods           //
////////////////////////////////////////////

/**
* @brief convert handle to C++ object into matlab usable data
*/ 
template<class objectClass> inline mxArray *convertPtr2Mat(objectClass *ptr)
{
    // prevent user from clearing the mex file! Otherwise, storage leaks might be caused
    mexLock();
    
    // allocate memory
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    
    // convert handle do matlab usable data
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new ClassHandle<objectClass>(ptr));
    
    return out;
}

/**
* @brief convert matlab usable data referring to an object into handle to C++ object
*/ 
template<class objectClass> inline ClassHandle<objectClass> *convertMat2HandlePtr(const mxArray *in)
{
    // check that the given pointer actually points to a real object
    if ( ( mxGetNumberOfElements(in) != 1 )     ||
         ( mxGetClassID(in) != mxUINT64_CLASS ) ||
           mxIsComplex(in)
       )
        mexErrMsgTxt("Input must be a real uint64 scalar.");
        
    ClassHandle<objectClass> *ptr = reinterpret_cast<ClassHandle<objectClass> *>(*((uint64_t *)mxGetData(in)));
    
    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    
    return ptr;
}

/**
* @brief convert matlab usable data referring to an object into direct pointer to the underlying C++ object
*/ 
template<class objectClass> inline objectClass *convertMat2Ptr(const mxArray *in)
{
    return convertMat2HandlePtr<objectClass>(in)->getPtrToObject();
}

/**
* @brief convert matlab usable data referring to an object into direct pointer to the underlying C++ object
*/ 
template<class objectClass> inline void destroyObject(const mxArray *in)
{
    // clean up
    delete convertMat2HandlePtr<objectClass>(in);
    
    // storage is freed, so users can savely clear the mex file again at any time...
    mexUnlock();
}

#endif // _NICE_CLASSHANDLEMTOCINCLUDE
