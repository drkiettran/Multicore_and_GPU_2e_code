SOURCES += prodCons.cpp
SOURCES += semaphore.cpp
HEADERS += semaphore.h
TARGET = prodCons
QMAKE_CXXFLAGS += -fopenmp -std=c++17
QMAKE_LFLAGS += -fopenmp -std=c++17
