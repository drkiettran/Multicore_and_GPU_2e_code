SOURCES += mpiAndThreads.cpp
QMAKE_CXXFLAGS += -std=c++17
LIBS += -pthread
TARGET = mpiAndThreads
QMAKE_CXX=mpic++
QMAKE_CC=mpicc
QMAKE_LINK=mpic++
