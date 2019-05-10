#-------------------------------------------------
#
# Project created by QtCreator 2014-01-17T08:51:35
#
#-------------------------------------------------

QT       += core

#QT       -= gui

TARGET = SAENO
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    cpp/Vec3D.cpp \
    cpp/Mat3D.cpp \
    cpp/DRec.cpp \
    cpp/Chameleon.cpp \
    cpp/configHelper.cpp \
    cpp/buildEpsilon.cpp \
    cpp/buildBeams.cpp \
    cpp/tensorHelper.cpp \
    cpp/sparseHelper.cpp \
    cpp/multigridHelper.cpp \
    cpp/FiniteBodyForces.cpp \
    cpp/imageHelper.cpp \
    cpp/downhillSimplexHelper.cpp \
    cpp/VirtualBeads.cpp \
    cpp/stack3DHelper.cpp

QMAKE_CXXFLAGS += -std=c++11
QMAKE_LFLAGS += -Wl,--large-address-aware
QMAKE_CXXFLAGS_DEBUG -= -O
QMAKE_CXXFLAGS_DEBUG -= -O1
QMAKE_CXXFLAGS_DEBUG -= -O2
QMAKE_CXXFLAGS_DEBUG -= -O3
QMAKE_CXXFLAGS_RELEASE += -O3

