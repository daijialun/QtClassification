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
    if( pix.height()>ui->labelImage->height() || pix.width()>ui->labelImage->width() )  {
            QPixmap imgScaled;
            imgScaled = pix.scaled(ui->labelImage->height(), ui->labelImage->width(), Qt::KeepAspectRatio);
            ui->labelImage->setPixmap(imgScaled);
    }
    else
            ui->labelImage->setPixmap(pix);
    QString pathImage = ui->lineImagePath->text();
    std::string path = pathImage.toStdString();
    mImage = cv::imread(path, 0);
}

void MainWindow::Prediction()  {
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

        for(int i=0; i<mNormalized.rows; i++)  {
                float* fData = mNormalized.ptr<float>(i);
                for(int j=0; j<mNormalized.cols; j++)  {
                        *blobData = fData[j];
                        blobData++;
                }
        }
        /*mNormalized.convertTo(mNormalized, CV_8UC1);
        for(int i=0; i<height_*width_; i++)  {
                *blobData = static_cast<float>(*mNormalized.data);
                blobData++;
                mNormalized.data++;
        }
        /*uchar* mData = mNormalized.data;
        for(int i=0; i<height_*width_; i++)  {
                *blobData = static_cast<float>(*mData);
                blobData++;
                mData++;
        }*/

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

        //std::vector<int> vec_seq;           // seq => shunxu list
        //for(int i=0; i<5; i++)  {
       //         vec_seq.push_back(vec_pairs[i].first);
        //}
        std::vector<std::string> labels_sorted;
        for(int i=0; i<labels_.size(); i++)  {
                labels_sorted.push_back( labels_[vec_pairs[i].first] );
        }
        for(int i=0; i<5; i++)  {
                qDebug() << QString::number(i) << " "
                                << QString::fromStdString(labels_[vec_pairs[i].first])
                                << QString::number(vec_pairs[i].second);
        }
        ui->labelClass1->setText(QString::fromStdString(labels_[vec_pairs[0].first]));
        ui->labelClass2->setText(QString::fromStdString(labels_[vec_pairs[1].first]));
        ui->labelClass3->setText(QString::fromStdString(labels_[vec_pairs[2].first]));
        ui->labelClass4->setText(QString::fromStdString(labels_[vec_pairs[3].first]));

        ui->labelScore1->setText(QString::number(vec_pairs[0].second));
        ui->labelScore2->setText(QString::number(vec_pairs[1].second));
        ui->labelScore3->setText(QString::number(vec_pairs[2].second));
        ui->labelScore4->setText(QString::number(vec_pairs[3].second));

        ShowTopImage(labels_sorted);

}

bool MainWindow::PairCompare(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
    return lhs.second > rhs.second;
}

void MainWindow::ShowTopImage(std::vector<std::string> labels)
{
        QString label1 = ":/images/" + QString::fromStdString(labels[0]) + ".png";
        QString label2 = ":/images/" + QString::fromStdString(labels[1]) + ".png";
        QString label3 = ":/images/" + QString::fromStdString(labels[2]) + ".png";
        QString label4 = ":/images/" + QString::fromStdString(labels[3]) + ".png";

        QPixmap qpixTop1, qpixTop2, qpixTop3, qpixTop4;
        qpixTop1.load(label1);
        qpixTop2.load(label2);
        qpixTop3.load(label3);
        qpixTop4.load(label4);


        ui->labelTop1->setPixmap(qpixTop1.scaled(ui->labelTop1->height(), ui->labelTop1->width(), Qt::KeepAspectRatio) );
       // ui->labelTop1->setPixmap(qpixTop1);
        //ui->labelTop1->resize(  qpixTop1.width(), qpixTop1.height() );
        ui->labelTop2->setPixmap(qpixTop2.scaled(ui->labelTop2->height(), ui->labelTop2->width(), Qt::KeepAspectRatio) );
       // ui->labelTop2->setPixmap(qpixTop2);
        //ui->labelTop2->resize(  qpixTop2.width(), qpixTop2.height() );
       ui->labelTop3->setPixmap(qpixTop3.scaled(ui->labelTop3->height(), ui->labelTop3->width(), Qt::KeepAspectRatio) );
        //ui->labelTop3->setPixmap(qpixTop3);
        //ui->labelTop3->resize(  qpixTop3.width(), qpixTop3.height() );
       ui->labelTop4->setPixmap(qpixTop4.scaled(ui->labelTop4->height(), ui->labelTop4->width(), Qt::KeepAspectRatio) );
       // ui->labelTop4->setPixmap(qpixTop4);
        //ui->labelTop4->resize(  qpixTop4.width(), qpixTop4.height() );

}

void MainWindow::ChangeModelIndex()  {
        model = ui->comboBoxModel->currentText();
        qDebug() << model;

}

void MainWindow::SelectModel()  {
        if( model == "Origin")  {
                net_.reset( new caffe::Net<float>("origin.prototxt", caffe::TEST));
                net_->CopyTrainedLayersFrom("origin.caffemodel");
        }
        else if( model == "Local" )  {
            net_.reset( new caffe::Net<float>("local.prototxt", caffe::TEST));
            net_->CopyTrainedLayersFrom("local.caffemodel");
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
        labels_.clear();
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
