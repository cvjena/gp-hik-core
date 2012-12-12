#ifdef NICE_USELIB_CPPUNIT

#include <string>
#include <exception>

#include <core/matlabAccess/MatFileIO.h>

#include <gp-hik-core/tools.h>

#include "TestFeatureMatrixT.h"

const bool verbose = false;
const bool verboseStartEnd = true;
const uint n = 15;
const uint d = 3;
const double sparse_prob = 0.8;

using namespace NICE;
using namespace std;

CPPUNIT_TEST_SUITE_REGISTRATION( TestFeatureMatrixT );

void TestFeatureMatrixT::setUp() {
}

void TestFeatureMatrixT::tearDown() {
}

void TestFeatureMatrixT::testSetup()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestFeatureMatrixT::testSetup ===================== " << std::endl;
  
	  std::vector< std::vector<double> > dataMatrix;

  generateRandomFeatures ( d, n, dataMatrix );

  int nrZeros(0);
  for ( uint i = 0 ; i < d; i++ )
  {
    for ( uint k = 0; k < n; k++ )
      if ( drand48() < sparse_prob ) 
		{
			dataMatrix[i][k] = 0.0;
			nrZeros++;
		}
  }

  if ( verbose ) {
    cerr << "data matrix: " << endl;
    printMatrix ( dataMatrix );
    cerr << endl;
  }
  
  transposeVectorOfVectors(dataMatrix);
  NICE::FeatureMatrixT<double> fm(dataMatrix);
  
  if ( (n*d)>0)
  {
  if (verbose)
    std::cerr << "fm.computeSparsityRatio(): " << fm.computeSparsityRatio() << " (double)nrZeros/(double)(n*d): " << (double)nrZeros/(double)(n*d) << std::endl;
  CPPUNIT_ASSERT_DOUBLES_EQUAL(fm.computeSparsityRatio(), (double)nrZeros/(double)(n*d), 1e-8);
  }
  
  transposeVectorOfVectors(dataMatrix);
  std::vector<std::vector<int> > permutations;
  if (verbose)
    std::cerr << "now try to set_features" << std::endl;
  fm.set_features(dataMatrix, permutations);
  if ( (n*d)>0)
  {
  if (verbose)
    std::cerr << "fm.computeSparsityRatio(): " << fm.computeSparsityRatio() << " (double)nrZeros/(double)(n*d): " << (double)nrZeros/(double)(n*d) << std::endl;
  CPPUNIT_ASSERT_DOUBLES_EQUAL(fm.computeSparsityRatio(), (double)nrZeros/(double)(n*d), 1e-8);
  }
  
  NICE::MatrixT<double> matNICE;
  fm.computeNonSparseMatrix(matNICE);
  
  if (verbose)
  {
    std::cerr << "converted NICE-Matrix" << std::endl;
    std::cerr << matNICE << std::endl;
  }
  
  std::vector<std::vector<double> > matSTD;
  fm.computeNonSparseMatrix(matSTD);
  
  if (verbose)
  {
    std::cerr << "converted std-Matrix" << std::endl;
    printMatrix(matSTD);
  }
  
  if (verboseStartEnd)
    std::cerr << "================== TestFeatureMatrixT::testSetup done ===================== " << std::endl;  
}


void TestFeatureMatrixT::testMatlabIO()
{
  
  if (verboseStartEnd)
    std::cerr << "================== TestFeatureMatrixT::testMatlabIO ===================== " << std::endl;
  
  NICE::MatFileIO matfileIOA = MatFileIO("./sparse3x3matrixA.mat",MAT_ACC_RDONLY);
  sparse_t sparseA;
  matfileIOA.getSparseVariableViaName(sparseA,"A");
  NICE::FeatureMatrixT<double> fmA(sparseA);//, 3);
  if ( verbose )
  {
    fmA.print(std::cerr);
  }
  if (verbose)
    std::cerr << "fmA.get_n(): " << fmA.get_n() << " fmA.get_d(): " << fmA.get_d() << std::endl;

  NICE::MatFileIO matfileIOM = MatFileIO("./sparse20x30matrixM.mat",MAT_ACC_RDONLY);
  sparse_t sparseM;
  matfileIOM.getSparseVariableViaName(sparseM,"M");
  NICE::FeatureMatrixT<double> fmM(sparseM);//, 20);
  if ( verbose )
  {
    fmM.print(std::cerr);
  }
  if (verbose)
    std::cerr << "fmM.get_n(): " << fmM.get_n() << " fmM.get_d(): " << fmM.get_d() << std::endl;

  NICE::MatrixT<double> matNICE;
  fmM.computeNonSparseMatrix(matNICE, true);
  
  if (verbose)
  {
    std::cerr << "converted NICE-Matrix" << std::endl;
    std::cerr << matNICE << std::endl;
  }
  
  std::vector<std::vector<double> > matSTD;
  fmM.computeNonSparseMatrix(matSTD, true);
  
  if (verbose)
  {
    std::cerr << "converted std-Matrix" << std::endl;
    printMatrix(matSTD);
  }

//   std::string filename = "/home/dbv/bilder/imagenet/devkit-1.0/demo/demo.train.mat";
//   std::string dataMatrixMatlab = "training_instance_matrix";
//   
//   	//tic
// 	time_t  readSparseMatlabMatrix_start = clock();
//   
//   std::cerr << "try to read " << filename << std::endl;
//   MatFileIO matfileIO = MatFileIO(filename,MAT_ACC_RDONLY);
//   std::cerr << "matfileIO successfully done"<< std::endl;
//   
//   sparse_t sparse;
//   matfileIO.getSparseVariableViaName(sparse,dataMatrixMatlab);
// 
// 	//toc
// 	float time_readSparseMatlabMatrix = (float) (clock() - readSparseMatlabMatrix_start);
// 	std::cerr << "Time for reading  the sparse Matlab Matrix: " << time_readSparseMatlabMatrix/CLOCKS_PER_SEC << " s" << std::endl;
//   
//   std::cerr << "sparse-struct read, now try to give it to our FeatureMatrixT" << std::endl;
//   
//   	//tic
// 	time_t  readSparseIntoFM_start = clock();
// 
// 	NICE::FeatureMatrixT<double> fm(sparse);
// 	
//   	//toc
// 	float time_readSparseIntoFM = (float) (clock() - readSparseIntoFM_start);
// 	std::cerr << "Time for parsing the sparse Matrix into our FeatureMatrixT-struct: " << time_readSparseIntoFM/CLOCKS_PER_SEC << " s" << std::endl;
// 
//   
// 	std::cerr << "fm.get_n(): " << fm.get_n() << " fm.get_d(): " << fm.get_d() << std::endl;
// 	std::cerr << "fm.computeSparsityRatio() of Imagenet: " << fm.computeSparsityRatio() << std::endl;

  if (verboseStartEnd)
    std::cerr << "================== TestFeatureMatrixT::testMatlabIO done===================== " << std::endl;
  
}


#endif
