#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    connect(ui->btnUpload, SIGNAL(clicked(bool)), this, SLOT(ShowDialog()) );
    connect(ui->btnShow, SIGNAL(clicked(bool)), this, SLOT(ShowImage()) );
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::ShowDialog()  {
    dialog = new QDialog;
    QString fileName = QFileDialog::getOpenFileName(dialog, "Open Image", "/home/white_ghoul/", "Images (*.jpg *.png)");
    ui->lineImagePath->setText(fileName);

    QPixmap pix;
    pix.load(fileName);
    if(pix.height()>300 || pix.width()>450)  {
            QPixmap imgScaled;
            imgScaled = pix.scaled(450, 300, Qt::KeepAspectRatio);
            ui->labelImage->setPixmap(imgScaled);
    }
    else
            ui->labelImage->setPixmap(pix);
}

void MainWindow::ShowImage()  {
        using namespace cv;
        QString pathImage = ui->lineImagePath->text();
        std::string path = pathImage.toStdString();
        cv::Mat img = cv::imread(path);
        if( img.empty() )  { qDebug() << "Error Image"; }
        else { qDebug() << "Yes"; }
        //cv::namedWindow("TEST");
        //imshow("test",img);
        //cv::waitKey();
}
