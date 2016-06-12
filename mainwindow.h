#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDialog>
#include <QString>
#include <QFileDialog>
#include <QPixmap>
#include <QPainter>
#include <QImage>
#include <QDebug>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QDialog *dialog;

public slots:
    void ShowDialog();
    void ShowImage();
};

#endif // MAINWINDOW_H
