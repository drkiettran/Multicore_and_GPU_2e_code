CONFIG += qt
SOURCES += qtconcurProdCons.cpp
SOURCES += semaphore.cpp
HEADERS += semaphore.h
TARGET = qtconcurProdCons
INCLUDEPATH += /usr/include/x86_64-linux-gnu/qt5/QtConcurrent
LIBS += -lQt5Concurrent -pthread
QMAKE_CXXFLAGS += -std=c++17
