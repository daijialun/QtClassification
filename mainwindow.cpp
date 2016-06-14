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
    connect(ui->btnShow, SIGNAL(clicked(bool)), this, SLOT(Prediction()) );
    connect(ui->comboBoxModel, SIGNAL(currentIndexChanged(QString)), this, SLOT(ChangeModelIndex()) );
    connect(ui->btnSelect, SIGNAL(clicked(bool)), this, SLOT(SelectModel()) );
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
    QString pathImage = ui->lineImagePath->text();
    std::string path = pathImage.toStdString();
    mImage = cv::imread(path, 0);
}

void MainWindow::Prediction()  {
        QString path = ui->lineImagePath->text();
        std::string imgPath = path.toStdString();
        mImage = cv::imread(imgPath, 0);
        if( mImage.empty() )  { qDebug() << "Error image input"; }

        // ******** Image Preprocess ******** //
        cv::Mat mResized;
        if( mImage.size() != cv::Size(height_, width_) )
                 cv::resize(mImage, mResized, cv::Size(height_, width_));
        else
                 mResized = mImage;
        cv::Mat mConvert;
        mResized.convertTo(mConvert, CV_32FC1);
        cv::Mat mNormalized;
        if( model == "Origin")  {
                 cv::Mat mMean(height_, width_, CV_32FC1, cv::Scalar(191,0,0,0)); //  Mean Value
                 cv::subtract(mConvert, mMean, mNormalized);
        }
        else  {
                mNormalized = mImage;
        }


        // ******** Transform Image to Blob ******** //
        blobImage = net_->input_blobs()[0];
        qDebug() << QString("Image Blob Input Shape:  ")
                        << QString::number( blobImage->num() ) << " "
                        << QString::number(blobImage->channels() ) << " "
                       << QString::number(blobImage->height() ) << " "
                       << QString::number(blobImage->width() );

        CHECK_EQ( blobImage->num(), 1) << "Network just test one image once.";
        CHECK_EQ(blobImage->channels(), 1) << "Network must be trained by one channel.";
        CHECK_EQ( blobImage->height(), 227) << "Network must input size: 227x227";
        CHECK_EQ( blobImage->width(), 227) << "Network must input size: 227x227";

        float* blobData = blobImage->mutable_cpu_data();
        uchar* mData = mNormalized.data;
        for(int i=0; i<height_*width_; i++)  {
                *blobData = static_cast<float>(*mData);
                blobData++;
                mData++;
        }

        // *********** Forward ************* //
        net_->Forward();

        blobPrediction = net_->output_blobs()[0];
        qDebug()    <<  QString("Prediction Output Shape:  ")
                        << QString::number( blobPrediction->num() ) << " "
                        << QString::number(blobPrediction->channels() ) << " "
                       << QString::number(blobPrediction->height() ) << " "
                       << QString::number(blobPrediction->width() );

        // *********** Sort Scores ********** //
        const float* begin = blobPrediction->cpu_data();
        const float* end =  begin + blobPrediction->channels();
        std::vector<float> vec_scores(begin, end);

        std::vector<std::pair<int, float> > vec_pairs;              // first => sequence ; second => scores
        for(int i=0; i<vec_scores.size(); i++)  {
                vec_pairs.push_back( std::make_pair(i, vec_scores[i]) );
        }
        for(int i=0; i<vec_scores.size(); i++)  {
                qDebug() << QString::number(vec_pairs[i].first) << " "
                                << QString::number(vec_pairs[i].second);
        }

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

void MainWindow::ChangeModelIndex()  {
        model = ui->comboBoxModel->currentText();
        qDebug() << model;

}

void MainWindow::SelectModel()  {
        if( model == "Origin")  {
                net_.reset( new caffe::Net<float>("deploy.prototxt", caffe::TEST));
                net_->CopyTrainedLayersFrom("alexnet.caffemodel");
        }
        else if( model == "Local" )  {
            net_.reset( new caffe::Net<float>("deploy.prototxt", caffe::TEST));
            net_->CopyTrainedLayersFrom("alexnet.caffemodel");
        }
        else if( model == "Global" )  {
            net_.reset( new caffe::Net<float>("deploy.prototxt", caffe::TEST));
            net_->CopyTrainedLayersFrom("alexnet.caffemodel");
        }

        // ******** Check Input Number and Output Number ******** //
        CHECK_EQ(net_->num_inputs(), 1) << "Network shoud have exactly one input.";
        CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

        caffe::Blob<float> *input_layer = net_->input_blobs()[0];
        channels_ = input_layer->channels();
        height_ = input_layer->height();
        width_ = input_layer->width();
        CHECK_EQ(channels_, 1) << "Network should only have one channel.";    // Check Model Channels


        // ******** Load Label.txt ******** //
        std::ifstream label_file("label.txt");
        CHECK(label_file) << "Unable to open labels file " << label_file;
        std::string line;
        while( std::getline(label_file, line) )  {
                    labels_.push_back(line);
        }


        // ******** Network Output ******** //
        caffe::Blob<float> *output_layer = net_->output_blobs()[0];
        CHECK_EQ( output_layer->channels(), labels_.size() ) << "Prediciton number must be equal to the class number.";     // Check labels equation


        // ******* Reshape Network for Only One Image ***** //
        input_layer->Reshape(1, channels_, height_, width_);
        net_->Reshape();
}
