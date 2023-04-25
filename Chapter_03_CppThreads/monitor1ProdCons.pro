CONFIG += qt
SOURCES += monitor1ProdCons.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = monitor1ProdCons
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
