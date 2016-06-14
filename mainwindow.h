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

enum deepModel {
        ORIGIN,
        LOCAL,
        GLOBAL,
};

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
     static bool PairCompare(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs);
    caffe::shared_ptr< caffe::Net<float> > net_;
    cv::Mat mImage;
    QString model;
    caffe::Blob<float>* blobImage;
    caffe::Blob<float>* blobPrediction;
    int height_;
    int width_;
    int channels_;
    std::vector<std::string> labels_;
    void ShowTopImage(std::vector<std::string> labels);

public slots:
    void ShowDialog();
    void Prediction();
    void ChangeModelIndex();
    void SelectModel();
};

#endif // MAINWINDOW_H
