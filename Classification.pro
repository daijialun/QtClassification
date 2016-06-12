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

INCLUDEPATH += /usr/local/include/

LIBS += /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_imgcodecs.so \
        /usr/local/lib/libopencv_imgproc.so
