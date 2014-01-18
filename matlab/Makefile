NICEFLAGS1=$(shell pkg-config libgp-hik-core --cflags --libs)
NICEFLAGS=$(subst -fopenmp,,$(NICEFLAGS1))

default:
	/home/alex/bin/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
	/home/alex/bin/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

classification:
	/home/alex/bin/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKClassifierMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp

regression:        
	/home/alex/bin/matlab/7.14/bin/mex ${NICEFLAGS} -largeArrayDims GPHIKRegressionMex.cpp ConverterMatlabToNICE.cpp ConverterNICEToMatlab.cpp
