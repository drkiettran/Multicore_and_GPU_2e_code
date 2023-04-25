CONFIG += qt
SOURCES += fairBarber.cpp semaphore.cpp
HEADERS += semaphore.h
TARGET = fairBarber
LIBS += -pthread
QMAKE_CXXFLAGS += -std=c++17
