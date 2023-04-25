CFLAGS=-O2

all:	histogram_V2 histogram_atomic histogram_V1 histogram_V1_ILP histogram_graph

histogram_V2:	histogram_V2.cu pgm.o
	nvcc ${CFLAGS}  histogram_V2.cu pgm.o -o histogram_V2

histogram_V1:	histogram_V1.cu pgm.o
	nvcc ${CFLAGS} histogram_V1.cu pgm.o -o histogram_V1

histogram_V1_ILP:	histogram_V1_ILP.cu pgm.o
	nvcc ${CFLAGS} histogram_V1_ILP.cu pgm.o -o histogram_V1_ILP

histogram_atomic:	histogram_atomic.cu pgm.o
	nvcc ${CFLAGS} histogram_atomic.cu pgm.o -o histogram_atomic

histogram_graph:	histogram_graph.cu pgm.o
	nvcc ${CFLAGS} histogram_graph.cu pgm.o -o histogram_graph

pgm.o:	../common/pgm.cpp 
	g++ ${CFLAGS} -c ../common/pgm.cpp -o ./pgm.o
