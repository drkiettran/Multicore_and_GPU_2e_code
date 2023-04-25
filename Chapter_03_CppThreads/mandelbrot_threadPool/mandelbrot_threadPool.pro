TARGET = mandelbrot_threadPool
CONFIG += qt
HEADERS += customThreadPool.h
SOURCES += main.cpp
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17 -latomic
