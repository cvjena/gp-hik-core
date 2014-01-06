/** 
* @file FeatureMatrixT.h
* @brief A feature matrix, storing (sparse) features sorted per dimension (Interface)
* @author Alexander Freytag
* @date 07-12-2011 (dd-mm-yyyy)
*/
#ifndef FEATUREMATRIXINCLUDE
#define FEATUREMATRIXINCLUDE

// STL includes
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <limits>

// NICE-core includes
#include <core/basics/Exception.h>
#include <core/basics/Persistent.h>
// 
#include <core/vector/MatrixT.h>
#include <core/vector/SparseVectorT.h>
// 
#ifdef NICE_USELIB_MATIO
  #include <core/matlabAccess/MatFileIO.h> 
#endif
  
// gp-hik-core includes
#include "SortedVectorSparse.h"
#include "gp-hik-core/parameterizedFunctions/ParameterizedFunction.h"


namespace NICE {

  /** 
 * @class FeatureMatrixT
 * @brief A feature matrix, storing (sparse) features sorted per dimension
 * @author Alexander Freytag
 */  
  
template<class T> class FeatureMatrixT : NICE::Persistent
{

  protected:
    int n;
    int d;
    std::vector<NICE::SortedVectorSparse<T> > features;
    
    //! verbose flag for output after calling the restore-function
    bool verbose;
    //! debug flag for output during debugging
    bool debug;


  public:
    
  //! STL-like typedef for type of elements
  typedef T value_type;

  //! STL-like typedef for const element reference
  typedef const T& const_reference;

  //! STL-like typedef for iterator
  typedef T* iterator;

  //! STL-like typedef for const iterator
  typedef const T* const_iterator;

  //! STL-like typedef for element reference
  typedef T& reference;
  
    //------------------------------------------------------
    // several constructors and destructors
    //------------------------------------------------------
  
    /** 
    * @brief Default constructor
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    FeatureMatrixT();
    
    /** 
    * @brief Recommended constructor
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy) 
    */
    FeatureMatrixT(const std::vector<std::vector<T> > & _features, const int & _dim = -1);
    
#ifdef NICE_USELIB_MATIO
    /** 
    * @brief Constructor reading data from matlab-files
    * @author Alexander Freytag
    * @date 10-01-2012 (dd-mm-yyyy)
    */
    FeatureMatrixT(const sparse_t & _features, const int & _dim = -1);//, const int & nrFeatures);
#endif

    /** just another constructor for sparse features */
    FeatureMatrixT(const std::vector< const NICE::SparseVector * > & X, const bool dimensionsOverExamples = false, const int & _dim = -1);
    
#ifdef NICE_USELIB_MATIO
    /**
    * @brief Constructor reading data from matlab-files and providing the possibility to
    * restrict the number of examples to a certain subset
    *
    * @param _features sparse data matrix (sett MatFileIO)
    * @param examples set of example indices
    */
    FeatureMatrixT(const sparse_t & _features, const std::map<int, int> & examples , const int & _dim = -1);
#endif

    /** 
    * @brief Default destructor
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    ~FeatureMatrixT();
    
    //------------------------------------------------------
    // several get and set methods including access operators
    //------------------------------------------------------
    
    /** 
    * @brief Get number of examples
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
      int get_n() const;
    /** 
    * @brief Get number of dimensions
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
      int get_d() const;
      
    /** 
    * @brief Sets the given dimension and re-sizes internal data structure. WARNING: this will completely remove your current data!
    * @author Alexander Freytag
    * @date 06-12-2011 (dd-mm-yyyy)
    */
      void set_d(const int & _d);
      
    /** set verbose flag used for restore-functionality*/
    void setVerbose( const bool & _verbose);
    bool getVerbose( ) const;     
    
    /** set debug flag used for debug output*/
    void setDebug( const bool & _debug);
    bool getDebug( ) const;        
      
      
    /** 
    * @brief  Compare F with this
    * @author Alexander Freytag
    * @date 05-01-2012 (dd-mm-yyyy)
    * @pre Dimensions of \c F and \c this must be equal
    * @param F data to compare with
    * @return true if \c F and \c this are equal
    */
    inline bool operator==(const FeatureMatrixT<T> & F) const;
    
    /**
    * @brief Compare \c F with \c this.
    * @author Alexander Freytag
    * @date 05-01-2012 (dd-mm-yyyy)
    * @pre Size of \c F and \c this must be equal
    * @param F data to compare with
    * @return true if \c F and \c this are not equal
    */
    inline bool operator!= (const FeatureMatrixT<T> & F) const;

    /**
    * @brief Copy data from \c F to \c this.
    * @author Alexander Freytag
    * @date 05-01-2012 (dd-mm-yyyy)
    * @param v New data
    * @return \c *this
    */
    inline FeatureMatrixT<T>& operator=(const FeatureMatrixT<T> & F);
      
    /** 
    * @brief Matrix-like operator for element access, performs validity check
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    inline T operator()(const int row, const int col) const;
    
    /** 
    * @brief Element access without validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    inline T getUnsafe(const int row, const int col) const;

    /** 
    * @brief Element access of original values without validity check
    * @author Erik Rodner
    */
    inline T getOriginal(const int row, const int col) const;

    /** 
    * @brief Sets a specified element to the given value, performs validity check
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    inline void set (const int row, const int col, const T & newElement, bool setTransformedValue = false);
    
    /** 
    * @brief Sets a specified element to the given value, without validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    inline void setUnsafe (const int row, const int col, const T & newElement, bool setTransformedValue = false);
    
    /** 
    * @brief Access to all element entries of a specified dimension, including validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void getDimension(const int & dim, NICE::SortedVectorSparse<T> & dimension) const;
    
    /** 
    * @brief Access to all element entries of a specified dimension, without validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void getDimensionUnsafe(const int & dim, NICE::SortedVectorSparse<T> & dimension) const;
    
    /** 
    * @brief Finds the first element in a given dimension, which equals elem (orig feature value, not the transformed one)
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void findFirstInDimension(const int & dim, const T & elem, int & position) const;
    
    /** 
    * @brief Finds the last element in a given dimension, which equals elem (orig feature value, not the transformed one)
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)1
    */
    void findLastInDimension(const int & dim, const T & elem, int & position) const;
    
    /** 
    * @brief Finds the first element in a given dimension, which is larger as elem (orig feature value, not the transformed one)
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void findFirstLargerInDimension(const int & dim, const T & elem, int & position) const;
    
    /** 
    * @brief Finds the last element in a given dimension, which is smaller as elem (orig feature value, not the transformed one)
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void findLastSmallerInDimension(const int & dim, const T & elem, int & position) const;
    
    //------------------------------------------------------
    // high level methods
    //------------------------------------------------------
    
    /**
    * @brief apply a parameterized function to the feature matrix
    * @author Alexander Freytag
    * @date 04-05-2012 (dd-mm-yyyy)
    *
    * @param pf the parameterized function (optional), if not given, nothing will be done
    */    
    void applyFunctionToFeatureMatrix ( const NICE::ParameterizedFunction *pf = NULL );
    
    /** 
    * @brief Computes the ratio of sparsity across the matrix
    * @author Alexander Freytag
    * @date 11-01-2012 (dd-mm-yyyy)
    */
    double computeSparsityRatio();

    /** 
    * @brief add a new feature and insert its elements in the already ordered structure
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    void add_feature(const std::vector<T> & feature, const NICE::ParameterizedFunction *pf = NULL);
    /** 
    * @brief add a new feature and insert its elements in the already ordered structure, will be casted to type T
    * @author Alexander Freytag
    * @date 25-04-2012 (dd-mm-yyyy)
    */    
    void add_feature(const NICE::SparseVector & feature, const NICE::ParameterizedFunction *pf = NULL);

    /** 
    * @brief add several new features and insert their elements in the already ordered structure
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    void add_features(const std::vector<std::vector<T> > & _features );
    
    /** 
    * @brief set the stored features to new values - which means deleting the old things and inserting the new ones. Return resulting permutation according to each dimension
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    void set_features(const std::vector<std::vector<T> > & _features, std::vector<std::vector<int> > & permutations, const int & _dim = -1);
    void set_features(const std::vector<std::vector<T> > & _features, std::vector<std::map<int,int> > & permutations, const int & _dim = -1);
    void set_features(const std::vector<std::vector<T> > & _features, const int & _dim = -1);
    void set_features(const std::vector< const NICE::SparseVector * > & _features, const bool dimensionsOverExamples = false, const int & _dim = -1);
    
    /**
    * @brief get a permutation vector for each dimension
    *
    * @param resulting permutation matrix
    */
    void getPermutations( std::vector<std::vector<int> > & permutations) const;
    void getPermutations( std::vector<std::map<int,int> > & permutations) const;
      
    /** 
    * @brief Prints the whole Matrix (outer loop over dimension, inner loop over features)
    * @author Alexander Freytag
    * @date 07-12-2011 (dd-mm-yyyy)
    */
    void print(std::ostream & os) const;
    
    /** 
    * @brief Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    * @author Alexander Freytag
    * @date 12-01-2012 (dd-mm-yyyy)
    */
    void computeNonSparseMatrix(NICE::MatrixT<T> & matrix, bool transpose = false) const;
    
    /** 
    * @brief Computes the whole non-sparse matrix. WARNING: this may result in a really memory-consuming data-structure!
    * @author Alexander Freytag
    * @date 12-01-2012 (dd-mm-yyyy)
    */
    void computeNonSparseMatrix(std::vector<std::vector<T> > & matrix, bool transpose = false) const;
    
    /** 
    * @brief Swaps to specified elements, performing a validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void swap(const int & row1, const int & col1, const int & row2, const int & col2);
    
    /** 
    * @brief Swaps to specified elements, without performing a validity check
    * @author Alexander Freytag
    * @date 08-12-2011 (dd-mm-yyyy)
    */
    void swapUnsafe(const int & row1, const int & col1, const int & row2, const int & col2);

    /**
    * @brief direct access to elements
    *
    * @param dim feature index
    *
    * @return sorted feature values
    */
    const SortedVectorSparse<T> & getFeatureValues ( int dim ) const { return features[dim]; };
 
    /**
    * @brief direct read/write access to elements
    *
    * @param dim feature index
    *
    * @return sorted feature values
    */
    SortedVectorSparse<T> & getFeatureValues ( int dim ) { return features[dim]; };
   
    
    /**
    * @brief compute the diagonal elements of the HIK kernel matrix induced by the features
    *
    * @param diagonalElements resulting vector
    */
    void hikDiagonalElements( Vector & diagonalElements ) const;

    /**
    * @brief Compute the trace of the HIK kernel matrix induced by the features
    *
    * @return value of the trace
    */
    double hikTrace() const;
    
    /**
    * @brief Return the number of nonzero elements in a specified dimension, that are currently stored in the feature matrix
    *
    * @return number of nonzero elements on the specified dimension
    */ 
    int getNumberOfNonZeroElementsPerDimension(const int & dim) const;
   
    /**
    * @brief Return the number of zero elements in a specified dimension, that are currently stored in the feature matrix
    *
    * @return number of nonzero elements on the specified dimension
    */ 
    int getNumberOfZeroElementsPerDimension(const int & dim) const;
    
    /** Persistent interface */
    virtual void restore ( std::istream & is, int format = 0 );
    virtual void store ( std::ostream & os, int format = 0 ) const;
    virtual void clear ( );

};

  //! default definition for a FeatureMatrix
  typedef FeatureMatrixT<double> FeatureMatrix;
  typedef FeatureMatrixT<bool> BoolFeatureMatrix;
  typedef FeatureMatrixT<char> CharFeatureMatrix;
  typedef FeatureMatrixT<int> IntFeatureMatrix;
  typedef FeatureMatrixT<float> FloatFeatureMatrix;


} // namespace

#ifdef __GNUC__
#include "gp-hik-core/FeatureMatrixT.tcc"
#endif

#endif
