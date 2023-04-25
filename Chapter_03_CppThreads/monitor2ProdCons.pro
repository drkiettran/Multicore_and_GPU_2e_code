CONFIG += qt
SOURCES += monitor2ProdCons.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = monitor2ProdCons
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
