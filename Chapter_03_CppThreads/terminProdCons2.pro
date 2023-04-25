CONFIG += qt
SOURCES += terminProdCons2.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = terminProdCons2
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
