all:	histogram

histogram:	histogram.cu pgm.o
	nvcc  histogram.cu pgm.o -o histogram

pgm.o:	../common/pgm.cpp
	g++ -c ../common/pgm.cpp -o ./pgm.o
