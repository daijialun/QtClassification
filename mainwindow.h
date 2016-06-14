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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "caffe/caffe.hpp"
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <utility>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static bool PairCompare(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs);

private:
    Ui::MainWindow *ui;
    QDialog *dialog;

public slots:
    void ShowDialog();
    void ShowImage();
    void ChangeModelIndex();
};

#endif // MAINWINDOW_H
