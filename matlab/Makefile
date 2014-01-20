NICEFLAGS1=$(shell pkg-config libgp-hik-core --cflags --libs)
NICEFLAGS=$(subst -fopenmp,,$(NICEFLAGS1))

default:
	/home/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
	/home/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

classification:
	/home/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

regression:        
	/home/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
