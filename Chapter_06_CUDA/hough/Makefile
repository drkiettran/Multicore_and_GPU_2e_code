all:	hough

hough:	hough.cu pgm.o
	nvcc -arch=compute_75 hough.cu pgm.o -o hough

pgm.o:	../common/pgm.cpp
	g++ -c ../common/pgm.cpp -o ./pgm.o
