CONFIG += qt
SOURCES += outliers.cpp
TARGET = outliers
INCLUDEPATH += /usr/include/x86_64-linux-gnu/qt5/QtConcurrent
LIBS += -lQt5Concurrent
QMAKE_CXXFLAGS += -std=c++17  
