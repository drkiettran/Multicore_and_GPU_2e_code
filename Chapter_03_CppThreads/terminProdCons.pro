CONFIG += qt
SOURCES += terminProdCons.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = terminProdCons
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
