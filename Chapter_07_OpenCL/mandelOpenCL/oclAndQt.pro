SOURCES+=main.cpp
HEADERS+=../cl_utility.h
CONFIG+=qt 
LIBS+= -lOpenCL -lpthread
TARGET=mandelOCL
QMAKE_CXX=g++
QMAKE_CC=gcc
QMAKE_LINK=g++
