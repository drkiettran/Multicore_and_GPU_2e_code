CONFIG += qt
SOURCES += terminProdConsMess.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = terminProdConsMess
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
