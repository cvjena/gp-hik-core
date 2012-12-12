/** 
* @file tools.h
* @brief Some very basic methods (e.g. generation of random numbers) (Interface and Implementation)
* @author Alexander Freytag
* @date 12/06/2011
*/
#ifndef FASTHIK_TOOLSINCLUDE
#define FASTHIK_TOOLSINCLUDE

#include "core/vector/MatrixT.h"
#include "core/vector/VectorT.h"
#include <cstdlib>


#include <vector>
#include <algorithm>
#include "core/vector/MatrixT.h"
#include <iostream>
#include <fstream>

using namespace std;

/** 
* @brief float extension of rand
* @author Alexander Freytag
* @date 12/06/2011
*/
inline float frand() { return (float)rand() / (float)(RAND_MAX); }; //returns from 0 to 1
/** 
* @brief double extension of rand
* @author Alexander Freytag
* @date 12/06/2011
*/
inline double drand() { return (double)rand() / (double)(RAND_MAX); }; //returns from 0 to 1

/** 
* @brief generates a random matrix of size n x d with values between zero and one
* @author Alexander Freytag
* @date 12/06/2011
*/
inline void generateRandomFeatures(const int & n, const int & d , NICE::Matrix & features)
{
	features.resize(n,d);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < d; j++)
		{
			features(i,j) = drand();
		}
	}
}

/** 
* @brief generates a std::vector of NICE::Vector ( size n x d ) with values between zero and one
* @author Alexander Freytag
* @date 12/06/2011
*/
inline void generateRandomFeatures(const int & n, const int & d , std::vector<NICE::Vector> & features)
{
	features.clear();
	for (int i = 0; i < n; i++)
	{
		NICE::Vector feature(d);
		for (int j = 0; j < d; j++)
		{
			feature[i] = drand();
		}
		features.push_back(feature);
		
	}
}

/** 
* @brief generates a std::vector of NICE::Vector ( size n x d ) with values between zero and one
* @author Alexander Freytag
* @date 12/06/2011
*/
inline void generateRandomFeatures(const int & n, const int & d , std::vector<std::vector<double> > & features)
{
	features.clear();
	for (int i = 0; i < n; i++)
	{
		std::vector<double> feature;
		feature.clear();
		for (int j = 0; j < d; j++)
		{
			feature.push_back(drand());
		}
		features.push_back(feature);
	}
}

/** 
* @brief generates a std::vector of std::vector with size n and values between zero and one
* @author Alexander Freytag
* @date 12/06/2011
*/
inline void generateRandomFeatures(const int & n, std::vector<double> & features)
{
	features.clear();
	for (int i = 0; i < n; i++)
	{
		features.push_back(drand());
	}
}

/** 
* @brief generates a std::vector of std::vector with size n and values between zero and one
* @author Alexander Freytag
* @date 12/06/2011
*/
inline void generateRandomFeatures(const int & n, NICE::Vector & features)
{
	features.resize(n);
	for (int i = 0; i < n; i++)
	{
		features[i] = drand48();
	}
}

/** 
* @brief generates a NICE::Matrix with random entries between zero and range
* @author Alexander Freytag
* @date 05-01-2012 (dd-mm-yyyy)
*/
inline NICE::Matrix generateRandomMatrix(const int & rows, const int & cols, const bool & symmetric = false, const int & range=1)
{
	NICE::Matrix M(rows,cols);
	if (symmetric)
	{
		//TODO check, wether is is more efficient to run first over j and then over i
		for (int i = 0; i < rows; i++)
			for (int j = i; j < cols; j++)
			{
				M(i,j) = drand()*range;
				M(j,i) = M(i,j);
			}
	}
	else
	{
		//TODO check, wether is is more efficient to run first over j and then over i
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				M(i,j) = drand()*range;
	}
	return M;
}

/** 
* @brief generates a NICE::Matrix with random entries between zero and range
* @author Alexander Freytag
* @date 05-01-2012 (dd-mm-yyyy)
*/
inline void generateRandomMatrix(NICE::Matrix M, const int & rows, const int & cols, const bool & symmetric = false, const int & range=1)
{
	M.resize(rows,cols);
	if (symmetric)
	{
		//TODO check, wether is is more efficient to run first over j and then over i
		for (int i = 0; i < rows; i++)
			for (int j = i; j < cols; j++)
			{
				M(i,j) = drand()*range;
				M(j,i) = M(i,j);
			}
	}
	else
	{
		//TODO check, wether is is more efficient to run first over j and then over i
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				M(i,j) = drand()*range;
	}
}

/** 
* @brief computes arbitrary Lp-Norm of a double-vector. Standard is L2-norm 
* @author Alexander Freytag
* @date 12/08/2011
*/
inline double vectorNorm(const std::vector<double> & a, const int & p=2)
{
	double norm(0.0);
	for (int i = 0; i < (int) a.size(); i++)
	{
		norm += pow(fabs(a[i]),p);
	}
	norm = pow(norm, 1.0/p);
	
	return norm;
}

/** 
* @brief Transposes a vector of vectors, assuming all inner vectors having the same length. Allocates space as much as already needed for the current data
* @author Alexander Freytag
* @date 12/08/2011
*/
template<class ElementType>
inline void transposeVectorOfVectors(std::vector<std::vector<ElementType> > & features)
{
	//unsave! did not check wether all dimensions are equally filled
	int d (features.size());
	int n ( (features[0]).size() );
	
	std::vector<std::vector<ElementType> > old_features(features);
	
	int tmp(n);
	n = d;
	d = tmp;
	features.resize(d);
	
	for (int dim = 0; dim < d; dim++)
	{
		features[dim].resize(n);
		for (int feat = 0; feat < n; feat++)
		{
			(features[dim])[feat] =  (old_features[feat])[dim];
		}
	}
}

/** 
* @brief Prints the whole Matrix (outer loop over dimension, inner loop over features)
* @author Alexander Freytag
* @date 12/07/2011
*/
inline void printMatrix(NICE::Matrix K)
{
   for (int row = 0; row < (int)K.rows(); row++)
   {
     for (int col = 0; col < (int)K.cols(); col++)
     {
       std::cerr << K(row,col) << " ";
     }
     std::cerr << std::endl;
  }
};

/** 
* @brief Prints the whole Matrix (outer loop over dimension, inner loop over features)
* @author Alexander Freytag
* @date 12/07/2011
*/
template <typename T>
inline void printMatrix(const std::vector< std::vector<T> > & K)
{
   for (int row = 0; row < (int)K.size(); row++)
   {
     for (int col = 0; col < (int)K[row].size(); col++)
     {
       std::cerr << K[row][col] << " ";
     }
     std::cerr << std::endl;
  }
};

template <class T>
inline void read_values(const std::string & file_of_values, std::vector<T> & values)
{
  std::cerr << "read from file " << file_of_values << std::endl;
  values.clear();
  ifstream iss(file_of_values.c_str());
  if ( !iss.good() )
  {
    std::cerr << "read_images::Unable to read the data!" << std::endl;
    return;
  }
  
  while ( ! iss.eof())
  {
    string str_float;
    T val;
    iss >> val; //str_float;
//     str_float >> val;
    values.push_back(val);
    std::cerr << val << " ";
  }
  std::cerr << std::endl;
  iss.close();
}

template <class T>
inline void write_values(const std::string & destination, const std::vector<T> & values)
{
  ofstream oss(destination.c_str());
  if ( !oss.good() )
  {
    std::cerr << "read_images::Unable to write the data!" << std::endl;
    return;
  }
  
  for (uint i = 0; i < values.size(); i++)
//   for (typename std::vector<T>::const_iterator it = values.begin; it != values.end(); it++)
  {
    oss << values[i] << std::endl;
  }
  
  oss.close();
}

template <class T>
inline void calculating_mean(const std::vector<T> numbers, T & mean)
{
  mean = 0.0;
  if (numbers.size() == 0) 
  {
    #ifndef NO_PROMPTS
    cerr << "calculating_mean: No numbers given." << endl;
    #endif
    return;
  }
  for (typename std::vector<T>::const_iterator it = numbers.begin(); it != numbers.end(); it++)
    mean += *it;
  
  if (numbers.size() > 0)
    mean /= (T) ((double) numbers.size());
  else
    mean = (T) 0;
}

template <class T>
inline void calculating_mean(const std::vector<T> numbers, std::vector<T> & means, const int & stepSize)
{
  means.clear();
  double mean(0.0);
  if (numbers.size() == 0) 
  {
    #ifndef NO_PROMPTS
    cerr << "calculating_mean: No numbers given." << endl;
    #endif
    return;
  }
  
  int cnt(0);
  for (typename std::vector<T>::const_iterator it = numbers.begin(); it != numbers.end(); it++, cnt++)
  {
    mean += *it;
    if ( cnt == (stepSize-1))
    {
      mean /= (T) ((double) stepSize);
      means.push_back(mean);
      mean = 0.0;
      cnt = -1;
    }
  }

}

inline double calculating_mean(const std::vector<double> numbers)
{
  double mean(0.0);
  if (numbers.size() == 0) 
  {
    #ifndef NO_PROMPTS
    cerr << "calculating_mean: No numbers given." << endl;
    #endif
    return mean;
  }
  for (std::vector<double>::const_iterator it = numbers.begin(); it != numbers.end(); it++)
    mean += *it;
  
  mean /= ((double) numbers.size());
  
  return mean;
}

inline void calculateMeanPerDimension(const std::vector<std::vector<double> > & numbers, std::vector<double> & meanValues)
{
  if (numbers.size() == 0)
    return;
  
  meanValues.resize(numbers[0].size());
  for (uint dim = 0; dim < numbers[0].size(); dim++)
  {
    meanValues[dim] = 0.0;
  }
  
  for (uint i = 0; i < numbers.size(); i++)
  {
    for (uint dim = 0; dim < numbers[i].size(); dim++)
    {
      meanValues[dim] += numbers[i][dim];
    }
  }
  
  for (uint dim = 0; dim < numbers[0].size(); dim++)
  {
    meanValues[dim] /= (double) numbers.size();
  }
}


inline void calculating_std_dev(const std::vector<double> numbers, const double & mean, double & std_dev)
{
  std_dev = 0.0;
  if (numbers.size() == 0) 
  {
    #ifndef NO_PROMPTS
    cerr << "calculating_mean: No numbers given." << endl;
    #endif
    return;
  }
  
  for (std::vector<double>::const_iterator it = numbers.begin(); it != numbers.end(); it++)
        std_dev += pow((*it) - mean,2);
  
  std_dev /= ((double) numbers.size());
  std_dev = sqrt(std_dev);
}

inline double calculating_std_dev(const std::vector<double> numbers, const double & mean)
{
  double std_dev (0.0);
  if (numbers.size() == 0) 
  {
    #ifndef NO_PROMPTS
    cerr << "calculating_mean: No numbers given." << endl;
    #endif
    return std_dev;
  }
  
  for (std::vector<double>::const_iterator it = numbers.begin(); it != numbers.end(); it++)
        std_dev += pow((*it) - mean,2);
  
  std_dev /= ((double) numbers.size());
  std_dev = sqrt(std_dev);
  
  return std_dev;
}

#endif
