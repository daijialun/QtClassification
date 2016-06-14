#-------------------------------------------------
#
# Project created by QtCreator 2016-06-11T17:07:47
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Classification
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH += /usr/local/include/  \
        /deep/tmp/caffe_mvco/include/ \
        /deep/tmp/caffe_mvco/build/src

LIBS += -L/deep/tmp/caffe_mvco/build/lib

LIBS += -lcaffe

LIBS += /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_imgcodecs.so \
        /usr/local/lib/libopencv_imgproc.so

LIBS += -pthread -lcaffe  -lglog -lgflags -lprotobuf -lboost_system \
             -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lopencv_core \
            -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++


#INCLUDEPATH += /usr/local/cuda/include
#LIBS += -L/usr/local/cuda/lib64
#LIBS += -lcudart -lcublas -lcurand

QMAKE_CXXFLAGS += -DCPU_ONLY

QMAKE_CXXFLAGS += -Wall -Wno-sign-compare

