#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->comboBoxModel->addItem("Origin");
    ui->comboBoxModel->addItem("Local");
    ui->comboBoxModel->addItem("Global");
    ui->comboBoxModel->setCurrentIndex(-1);

    connect(ui->btnUpload, SIGNAL(clicked(bool)), this, SLOT(ShowDialog()) );
    connect(ui->btnShow, SIGNAL(clicked(bool)), this, SLOT(ShowImage()) );
    connect(ui->comboBoxModel, SIGNAL(currentIndexChanged(QString)), this, SLOT(ChangeModelIndex()) );
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
        using namespace caffe;
        QString pathImage = ui->lineImagePath->text();
        std::string path = pathImage.toStdString();
        cv::Mat matImage = cv::imread(path, 0);
        if( matImage.empty() )  { qDebug() << "Error Image"; }
        else { qDebug() << "Yes"; }
        //cv::namedWindow("TEST");
        //cv::imshow("TEST",img);

        // ********** New Network ************* //
        shared_ptr< Net<float> > net_;
        net_.reset( new Net<float>("deploy.prototxt", TEST));
        net_->CopyTrainedLayersFrom("alexnet.caffemodel");

        CHECK_EQ(net_->num_inputs(), 1) << "Network shoud have exactly one input.";
        CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";


        // ******** Network Input Blob ********* //
        Blob<float> *input_layer = net_->input_blobs()[0];
        qDebug() << QString("Normal Input Shape:  ")
                        << QString::number( input_layer->num() ) << " "
                        << QString::number(input_layer->channels() ) << " "
                       << QString::number(input_layer->height() ) << " "
                       << QString::number(input_layer->width() );
        CHECK( input_layer->channels() == 1 )
                << "Input layer only have 1 channel.";


        // ******* Paramters ************ //
        int nums_ =  input_layer->num();
        int channels_ = input_layer->channels();
        int height_ = input_layer->height();
        int width_ = input_layer->width();
        cv::Size sizeGeometry(height_, width_);


        // ******** Mean Value ********** //
        cv::Mat matMean(height_, width_, CV_32FC1, cv::Scalar(191,0,0,0));


        // ******** Load Labels ********** //
        std::ifstream label_file("label.txt");
        CHECK(label_file) << "Unable to open labels file " << label_file;
        string line;
        std::vector<string> labels_;
        while( std::getline(label_file, line) )  {
                    labels_.push_back(line);
        }


        // ******** Network Output ********//
        Blob<float> *output_layer = net_->output_blobs()[0];
        CHECK_EQ( output_layer->channels(), labels_.size()) <<
                 "Number of labels is different from the output layer dimension.";
        qDebug()  << QString("Normal Output Shape:  ")
                        << QString::number( output_layer->num() ) << " "
                        << QString::number(output_layer->channels() ) << " "
                       << QString::number(output_layer->height() ) << " "
                       << QString::number(output_layer->width() );


        // ******* Reshape Network for Only One Image ***** //
        input_layer->Reshape(1, channels_, height_, width_);
        net_->Reshape();

        /*qDebug() << QString::number( input_layer->num() ) << " "
                        << QString::number(input_layer->channels() ) << " "
                       << QString::number(input_layer->height() ) << " "
                       << QString::number(input_layer->width() );*/


        // ******** Transform Image ************ //
        cv::Mat matResized;
        if( matImage.size() != sizeGeometry )
            cv::resize(matImage, matResized, sizeGeometry);
        else
            matResized = matImage;
        cv::Mat matConvert;
        matResized.convertTo(matConvert, CV_32FC1);
        cv::Mat matNormalized;
        cv::subtract(matConvert, matMean, matNormalized);
        //cv::imshow("Normalized",matNormalized);
        //cv::waitKey();


        // ******* Input Image to Blob ************ //
        Blob<float>* blobImage = net_->input_blobs()[0];
        qDebug() <<  QString("Image Input Shape:  ")
                        << QString::number( blobImage->num() ) << " "
                        << QString::number(blobImage->channels() ) << " "
                       << QString::number(blobImage->height() ) << " "
                       << QString::number(blobImage->width() );
        float* blobData = blobImage->mutable_cpu_data();
        uchar* mData = matNormalized.data;
        for(int i=0; i<height_*width_; i++)  {
                *blobData = static_cast<float>(*mData);
                blobData++;
                mData++;
        }

        // *********** Forward ************* //
        net_->Forward();

        Blob<float>* blobOut = net_->output_blobs()[0];
        qDebug()    <<  QString("Image Output Shape:  ")
                        << QString::number( blobOut->num() ) << " "
                        << QString::number(blobOut->channels() ) << " "
                       << QString::number(blobOut->height() ) << " "
                       << QString::number(blobOut->width() );

        // *********** Sort Scores ********** //
        const float* begin = blobOut->cpu_data();
        const float* end =  begin + blobOut->channels();
        std::vector<float> vec_scores(begin, end);

        std::vector<std::pair<int, float> > vec_pairs;              // first => sequence ; second => scores
        for(int i=0; i<vec_scores.size(); i++)  {
                vec_pairs.push_back( std::make_pair(i, vec_scores[i]) );
        }
        /*for(int i=0; i<vec_scores.size(); i++)  {
                qDebug() << QString::number(vec_pairs[i].first) << " "
                                << QString::number(vec_pairs[i].second);
        }*/

        std::partial_sort(vec_pairs.begin(), vec_pairs.begin()+5, vec_pairs.end(), PairCompare);

        std::vector<int> vec_seq;           // seq => shunxu
        for(int i=0; i<5; i++)  {
                vec_seq.push_back(vec_pairs[i].first);
        }
        for(int i=0; i<5; i++)  {
                qDebug() << QString::number(i) << " "
                                << QString::fromStdString(labels_[vec_seq[i]])
                                << QString::number(vec_pairs[i].second);
        }
        ui->labelClass1->setText(QString::fromStdString(labels_[vec_seq[0]]));
        ui->labelClass2->setText(QString::fromStdString(labels_[vec_seq[1]]));
        ui->labelClass3->setText(QString::fromStdString(labels_[vec_seq[2]]));
        ui->labelClass4->setText(QString::fromStdString(labels_[vec_seq[3]]));
        ui->labelClass5->setText(QString::fromStdString(labels_[vec_seq[4]]));

        ui->labelScore1->setText(QString::number(vec_pairs[0].second));
        ui->labelScore2->setText(QString::number(vec_pairs[1].second));
        ui->labelScore3->setText(QString::number(vec_pairs[2].second));
        ui->labelScore4->setText(QString::number(vec_pairs[3].second));
        ui->labelScore5->setText(QString::number(vec_pairs[4].second));


}

bool MainWindow::PairCompare(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
    return lhs.second > rhs.second;
}
