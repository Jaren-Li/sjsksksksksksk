/*
 * @Author: your name
 * @Date: 2022-04-12 21:38:25
 * @LastEditTime: 2022-05-01 14:15:08
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /RM_Standard2022-master/main.cpp
 */
#include <iostream>
#include <thread>
#include "rmcv/rmcv.h"
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

// TODO: everything use float, except PnP(double) (needs more investigation)

using namespace std;
using namespace cv;
using namespace ov;
using namespace dnn;

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

int classnum = 4;

float weightAngles[10];
unsigned char weightAngleNumber = 0;

void imagePreprocessing(Mat img, float* &result){
    Mat RGBImg, ResizeImg;
    cvtColor(img, RGBImg, COLOR_BGR2RGB);
    cv::resize(RGBImg, ResizeImg, Size(320, 320));
    // mean_rgb = [0.485, 0.456, 0.406]
    // std_rgb  = [0.229, 0.224, 0.225]

    int channels = ResizeImg.channels(), height = ResizeImg.rows, width = ResizeImg.cols;

    result = (float*)malloc(channels * height * width * sizeof(float));
    memset(result, 0, channels * height * width * sizeof(float));

    // Convert HWC to CHW and Normalize
    float mean_rgb[3] = {0.485, 0.456, 0.406};
    float std_rgb[3]  = {0.229, 0.224, 0.225};
    uint8_t* ptMat = ResizeImg.ptr<uint8_t>(0);
    int area = height * width;
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int srcIdx = c * area + h * width + w;
                int divider = srcIdx / 3;  // 0, 1, 2
                for (int i = 0; i < 3; ++i)
                {
                    result[divider + i * area] = static_cast<float>((ptMat[srcIdx] * 1.0f/255.0f - mean_rgb[i]) * 1.0f/std_rgb[i] );
                }
            }
        }
    }
}

void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

std::vector<BoxInfo> decode_infer(const ov::Tensor& box_tensor, 
                    const ov::Tensor& score_tensor,
                    float threshold,
                    float scale_x,
                    float scale_y) {
    vector<BoxInfo> results;
    BoxInfo box;
    ov::Shape boxShape =  box_tensor.get_shape();
    ov::Shape scoreShape =  score_tensor.get_shape();
    const float *out = box_tensor.data<const float>();
    const float *out1 = score_tensor.data<const float>();
    for(size_t i = 0; i < classnum; i++){
        for(size_t j = scoreShape[2]*i ; j < scoreShape[2]*(i+1); j++){
                if(out1[j] > threshold){
                    box.x1 = out[(j-scoreShape[2]*i)*4];
                    box.y1 = out[(j-scoreShape[2]*i)*4 + 1];
                    box.x2 = out[(j-scoreShape[2]*i)*4 + 2];
                    box.y2 = out[(j-scoreShape[2]*i)*4 + 3];
                    box.score = out1[j];
                    box.label = i;
                    results.push_back(box);
                }
        }
    }
    nms(results, 0.5);
    for(BoxInfo &box : results){
        box.x1 = box.x1 * scale_x;
        box.x2 = box.x2 * scale_x;
        box.y1 = box.y1 * scale_y;
        box.y2 = box.y2 * scale_y;
    }
    return results;
}

static const char *class_names[] = {"0", "2", "3", "4", "5", "6"};

float getWeightAngel(int speed, float dis,  float &weightAngel, float &flightTime)
{
    // float targetAngle;
    // float t1 = (-30 + sqrt(30 * 30 - 2 * 18.89027  * dis)) / (-18.89027 );
    // float t2 = (-30 - sqrt(30 * 30 - 2 * 18.89027  * dis)) / (-18.89027 );
    // // t1<t2?a = (-1.1 + 0.5 * 9.8 * t1 * t1)/(30.0 * t1):a = (-1.1 + 0.5 * 9.8 * t2 * t2)/(30.0 * t2);
    // t1 < t2 ? flightTime = t1 : flightTime = t2;
    // // cerr << "t1:" <<t1 << endl;
    // // cerr << "t2:" << t2 << endl;
    // // cerr << "a:" << a << endl;
    // targetAngle = (-1.1 + 0.5 * 9.8 * flightTime * flightTime)/(30.0 * flightTime);
    // weightAngel =  (targetAngle-asin(-1.1/(dis)))*180/3.14;

    // int speed = 12;
    // cout << speed << endl;
    float Speed;
    if(speed == 15){
        Speed = (float)speed-4.5;
    }else if(speed == 18){
        Speed = (float)speed-7;
    }else if(speed == 30){
        Speed = (float)speed;
    }
    
    float targetAngle;
    float kSmall = 0.000067165;
    float kBig = 0.000429838;
    float mSmall = 0.0032;
    float mBig = 0.043;
    float height = 1.2;
    // float height = -0.3;
    float kv2m = kSmall * Speed * Speed / mSmall;
    float t1 = (Speed + sqrt(Speed * Speed - 2 * kv2m  * dis)) / kv2m;
    float t2 = (Speed - sqrt(Speed * Speed - 2 * kv2m  * dis)) / kv2m;
    
    float kv2mss = kSmall * speed * speed / mSmall;
    float t11 = (speed + sqrt(speed * speed - 2 * kv2mss  * dis)) / kv2mss;
    float t22 = (speed - sqrt(speed * speed - 2 * kv2mss  * dis)) / kv2mss;
    // cout << t1 << endl;
    // cout << t2 << endl;
    if(t1 > 0 && t2 > 0){
        t1 < t2 ? flightTime = t1 : flightTime = t2;
    }  
    else {
        if(t1 > 0){
            flightTime = t1;
        }
        else if(t2 > 0)
        {
            flightTime = t2;
        }
        else cout << "solve error" << endl; 
    }
    targetAngle = (height + 0.5 * 9.82 * flightTime * flightTime)/(Speed * flightTime);
    weightAngel =  (targetAngle-asin(height/(dis)))*180/3.14;
    // t11>t22?flightTime = t22:flightTime = t11;
}


int main(int argc, char *argv[]) {
    rm::SerialPort serialPort;

    rm::Request request{1, 0, 18, 0};
    rm::Response r;

    float vp,vy;
    bool speedSymbol =true;
    bool addspeed = true;

    //weight add
    int speed = 30;           //射速
    int coefficient = 1.3536; //k*v*v/m    k为空气系数     小弹丸：1.3536
    // float height = -0.2;      //以m为单位,枪口水平为xy，可以为负值
    float weightAngel=0.0, flightTime=0.0;

    //创建卡尔曼滤波器firstKF
    const int stateNum=4;                                      //状态值4×1向量(p,y,△p,△y)
    const int measureNum=4;                                    //测量值4×1向量(p,y,△p,△y)
    const int controlNum = 2;
    KalmanFilter KF(stateNum, measureNum, controlNum);
    setIdentity(KF.measurementMatrix);                                             //测量矩阵H
    setIdentity(KF.controlMatrix);    
    // setIdentity(KF.processNoiseCov, Scalar::all(0.07.5e-10));                            //系统噪声方差矩阵Q
    setIdentity(KF.processNoiseCov, Scalar::all(2.5e-9));                        //系统噪声方差矩阵Q 5e-9/1e-10
    setIdentity(KF.measurementNoiseCov, Scalar::all(0.05));   //0.0286                     //测量噪声方差矩阵R 0.05
    setIdentity(KF.errorCovPost, Scalar::all(0.05));                                  //后验错误估计协方差矩阵P
    randn(KF.statePost, Scalar::all(0), Scalar::all(1));                         //初始状态值x(0)
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                                          //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义
    Mat controlment = Mat::zeros(controlNum, 1,CV_32F);
    Mat prediction;
    KF.transitionMatrix = (Mat_<float>(4, 4) <<   
                                1,0,8   ,0,   
                                0,1,0       ,8,   
                                0,0,1       ,0,   
                                0,0,0       ,1 );//元素导入矩阵，按行;  
                            KF.controlMatrix = (Mat_<float>(4, 2) <<   
                                    32         , 0,   
                                    0                ,32,   
                                    8                ,0,   
                                    0                ,8);//元素导入矩阵，按行;  
    bool firstKF = true;
    float padd, yadd; 

    thread serialPortThread([&]() {
        bool status = serialPort.Initialize();
        bool receiveData;
        int failure = 0;
        while (status) {
            receiveData = serialPort.Receive(request);
            // cout << receiveData << endl;
            if(receiveData){
                // cout << request.pitch.data << "," << request.yaw.data << endl;
            //     if(firstKF == false){
            //             // if(isnan(request.vppitch.f) || isnan(request.yaw.f)) cerr << "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP" << endl;
            //         controlment.at<float>(0) = 0.0;
            //         controlment.at<float>(1) = 0.0;
            //         measurement.at<float>(0) = (float)request.pitch.data;
            //         measurement.at<float>(1) = (float)request.yaw.data;
            //         measurement.at<float>(2) = 0.0;
            //         measurement.at<float>(3) = 0.0;
            //         setIdentity(KF.errorCovPost, Scalar::all(1)); 
            //         firstKF = true;
            //     }
            //     else{
            //         // cout<<         float(r.pitch.data) << endl;
            //         // cout << float(r.yaw.data) << endl;
            //         prediction = KF.predict();
            //         padd = prediction.at<float>(0) - measurement.at<float>(0);
            //         yadd = p.data += flyPAdd;
                    // r.yaw.data += flyYAdd;rediction.at<float>(1) - measurement.at<float>(1);
            //         // cerr << "------------------------------" << endl;
            //         // cerr << prediction << endl;
            //         // cout << padd << "," << yadd << endl;
            //         // cerr << "_______________________________" << endl;
            //         // cerr << measurement << endl;
            //         // cerr << "------------------------------" << endl;
            //         if(fabs(float(r.pitch.data)) > 100 || fabs(float(r.yaw.data)) > 100){
            //             controlment.at<float>(0) =  (((float)request.pitch.data - (float)measurement.at<float>(0))/20.0 - measurement.at<float>(2))/20.0;
            //             controlment.at<float>(1) = (((float)request.yaw.data - (float)measurement.at<float>(1))/20.0 - measurement.at<float>(3))/20.0;

            //             measurement.at<float>(2) = ((float)request.pitch.data - (float)measurement.at<float>(0))/20.0;
            //             measurement.at<float>(3) = ((float)request.yaw.data - (float)measurement.at<float>(1))/20.0;
                                    
            //             measurement.at<float>(0) = (float)request.pitch.data ;
            //             measurement.at<float>(1) = (float)request.yaw.data ;
            //             KF.correct(measurement);
            //         }
            //         else{
            //             controlment.at<float>(0) =  (((float)request.pitch.data + float(r.pitch.data) - (float)measurement.at<float>(0))/20.0 - measurement.at<float>(2))/20.0;
            //             controlment.at<float>(1) = (((float)request.yaw.data + float(r.yaw.data)- (float)measurement.at<float>(1))/20.0 - measurement.at<float>(3))/20.0;

            //             measurement.at<float>(2) = ((float)request.pitch.data + float(r.pitch.data) - (float)measurement.at<float>(0))/20.0;
            //             measurement.at<float>(3) = ((float)request.yaw.data + float(r.yaw.data) - (float)measurement.at<float>(1))/20.0;
                                    
            //             measurement.at<float>(0) = (float)request.pitch.data + float(r.pitch.data);
            //             measurement.at<float>(1) = (float)request.yaw.data + float(r.yaw.data);
            //             KF.correct(measurement);
            //         }
            //     }
            }
            if (!receiveData && failure < 256) {
                // std::cout << "Serial port receive failed." << std::endl;
                // failure++;
            } else {
                // status = false;
            }
        }
        std::cout << "Serial port closed." << std::endl;
    });
    time_t begin, end;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1284.67014818928, 0, 645.232541159854, 0, 1280.50458091643, 502.868323257852, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.057022633866224, 0.106303229938183, 0, 0, 0);
    Mat lastROI;
    Point2f lastBox;
    thread detectionThread([&]() {
        rm::DahengCamera camera;
        // bool status = camera.dahengCameraInit((char *) "KE0210030296", true, (int) (1.0 / 210.0 * 1000000), 1.5);
        bool status = camera.dahengCameraInit((char *) "KE0210010004", true, 5000, 1.0);
        string model_path = "/home/yq/桌面/rm2022/RM_Standard2022-master/bb2345processed.onnx";
        string model_xml = "/home/yq/桌面/rm2022/picodet5_12b.xml";
        string model_bin = "/home/yq/桌面/rm2022/picodet5_12b.bin";
        Core core;
        ov::Shape input_shape = {1, 3, 270, 320};
        std::shared_ptr<ov::Model> model = core.read_model(model_xml);
        CompiledModel compiled_model = core.compile_model(model, "CPU");
        InferRequest infer_request = compiled_model.create_infer_request();
        auto input_port = compiled_model.input();   
        Mat imageRoi;
        while (status) {
            float* imgPtr;
            cv::Mat frame = camera.getFrame();
            cv::Mat blob;
        if(!frame.empty()){


            // cv::flip(frame,frame,-1);
            
            blob = blobFromImage(frame, 1.0/255.0, Size(320,270),true, false, CV_32F);
            float* input_data = (float*)blob.data;
            Tensor input_tensor(input_port.get_element_type(), input_shape, input_data);
            infer_request.set_input_tensor(input_tensor);   
            infer_request.infer();
            const ov::Tensor& box_tensor = infer_request.get_output_tensor(0);
            const ov::Tensor& score_tensor = infer_request.get_output_tensor(1);
            float scale_x = frame.size().width / 320.0;
            float scale_y = frame.size().height / 320.0;
            std::vector<BoxInfo> boxs;

            boxs = decode_infer(box_tensor, score_tensor, 0.5, scale_x, scale_y);
            // cout << boxs.size() << endl;
            for(BoxInfo &box : boxs){
                cv::rectangle(frame, cv::Rect(cv::Point(box.x1, box.y1),cv::Point(box.x2, box.y2)),
                        Scalar(0, 255, 0));
                char text[256];
                sprintf(text, "%s %.1f%%", class_names[box.label], box.score * 100);
                cv::putText(frame, text, cv::Point(box.x1 - 20, box.y1 -20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
            }
            auto ownCamp = static_cast<rm::CampType>(request.ownCamp);
            std::vector<rm::LightBar> lightBars;
            std::vector<rm::Armour> armours;
            cv::Mat binary;
            for(BoxInfo &box : boxs) {
                if(box.x1 < box.x2) {
                    box.x1 -= 0.25 * fabs(box.x1 - box.x2);
                    box.x2 += 0.25 * fabs(box.x1 - box.x2);
                }else{
                    box.x1 += 0.25 * fabs(box.x1 - box.x2);
                    box.x2 -= 0.25 * fabs(box.x1 - box.x2);
                }

                
                if(box.x1 <= 0 || box.y1 <= 0 || box.x2 <= 0 || box.y2 <= 0) continue;
                if(box.x1 > frame.cols || box.y1 > frame.rows || box.x2 > frame.cols || box.y2 > frame.rows) continue;
                // cout << box.x1 << "," << box.y1 << "    " << box.x2 << "," << box.y2 << endl;
                cv::rectangle(frame, cv::Rect(cv::Point2f(box.x1, box.y1),cv::Point2f(box.x2, box.y2)),
                        Scalar(255, 0, 255));
                // double t = (double)getTickCount();
                cv::Mat imageROI = frame(Rect(cv::Point2i(box.x1, box.y1),cv::Point2i(box.x2, box.y2)));
                lastROI = imageROI;
                lastBox = cv::Point2f(box.x1, box.y1);
                rm::ExtractColor(imageROI, binary, ownCamp, true, 40, {5, 5});
                rm::FindLightBars(binary, lightBars, 1.5, 10, 25, 80, 1500, frame, cv::Point2f(box.x1, box.y1), true);
                // cout << lightBars.size() << endl;
                rm::FindArmour(lightBars, armours, 90, 90, 0.0, 0.9, 0.5, ownCamp,
                               cv::Size2f {(float) frame.cols, (float) frame.rows});
            //                                        t = ((double)getTickCount() - t)/getTickFrequency();
            // cout << "执行时间(秒): " << t << endl;
            }
            if(boxs.size() == 0 && !lastROI.empty()){
                rm::ExtractColor(lastROI, binary, ownCamp, true, 40, {5, 5});
                rm::FindLightBars(binary, lightBars, 1.5, 10, 25, 80, 1500, frame, lastBox, true);
                rm::FindArmour(lightBars, armours, 90, 90, 0.0, 0.9, 0.5, ownCamp,
                               cv::Size2f {(float) frame.cols, (float) frame.rows});
            }
// cout << padd << "," << yadd << endl;
            // cout << armours.size() << endl;
                if(!armours.empty()) { 
                    cv::Mat tvecs, rvecs;
                    rm::SolvePNP(armours[0].vertices, cameraMatrix, distCoeffs, {7.5, 7.5}, tvecs, rvecs);
                    double distance = rm::SolveDistance(tvecs);
                    r.pitch.data = atan2(tvecs.at<double>(1), tvecs.at<double>(2))*180.0/CV_PI;
                    r.yaw.data = atan2(tvecs.at<double>(0), tvecs.at<double>(2))*180.0/CV_PI;
                    getWeightAngel(request.speed, distance * 0.01, weightAngel, flightTime);
                    weightAngles[weightAngleNumber] = weightAngel;
                    weightAngleNumber += 1;
                    if(weightAngleNumber == 10) weightAngleNumber = 0;

                    flightTime = flightTime * 1000;
                    // cout << distance << endl;
                    
                    
                    // cout << KF.statePost.at<float>(2) << " " << KF.statePost.at<float>(3) << endl;
                    //KF
                    if(firstKF == true){
                        // if(isnan(request.pitch.f) || isnan(request.yaw.f)) cerr << "PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP" << endl;
                    controlment.at<float>(0) = 0.0;
                    controlment.at<float>(1) = 0.0;
                    measurement.at<float>(0) = (float)request.pitch.data;
                    measurement.at<float>(1) = (float)request.yaw.data;
                    measurement.at<float>(2) = 0.0;
                    measurement.at<float>(3) = 0.0;
                    firstKF = false;
                    }
                    else{
                        prediction = KF.predict();
                        // padd = prediction.at<float>(0) - measurement.at<float>(0);
                        // yadd = prediction.at<float>(1) - measurement.at<float>(1);
                        // cerr << "------------------------------" << endl;
                        // cerr << "mea" <<measurement << endl;
                        // cout << padd << "," << yadd << endl;
                        // cout << measurement.at<float>(2) << "，" << measurement.at<float>(3) << endl;
                        // cerr << "_______________________________" << endl;
                        // cerr << "mea" << measurement << endl;flyPAdd;
                    // r.yaw.data += flyYAdd;quest.pitch.data + r.pitch.data;
                            // measurement.at<float>(1) = (float)request.yaw.data + r.yaw.data;
                            controlment.at<float>(0) =  (((float)request.pitch.data + float(r.pitch.data) - (float)measurement.at<float>(0))/8.0 - measurement.at<float>(2))/8.0;
                        controlment.at<float>(1) = (((float)request.yaw.data + float(r.yaw.data)- (float)measurement.at<float>(1))/8.0 - measurement.at<float>(3))/8.0;

                        measurement.at<float>(2) = ((float)request.pitch.data + float(r.pitch.data) - (float)measurement.at<float>(0))/8.0;
                        measurement.at<float>(3) = ((float)request.yaw.data + float(r.yaw.data) - (float)measurement.at<float>(1))/8.0;
                                    
                        measurement.at<float>(0) = (float)request.pitch.data + float(r.pitch.data);
                        measurement.at<float>(1) = (float)request.yaw.data + float(r.yaw.data);
                            KF.correct(measurement);
                    }
                    // cout << KF.statePost << endl;
                        // else{htBars(lightBars, frame, -1);
                // rm::debug::DrawArmours(armours, frame, -1);
                // cv::imshow("frame", frame);
                        //     measurement.at<float>(0) = (float)request.pitch.data - float(r.pitch.data);
                        //     measurement.at<float>(1) = (float)request.yaw.data - float(r.yaw.data);
                            // cout << "-----------------" << endl;
                            // KF.correct(measurement);
                        // }
                    // }
                    // padd = KFsta.at<float>(0) - measurement.at<float>(0);
                    //     yadd = prediction.at<float>(1) - measurement.at<float>(1);
                    // if(fabs(KF.statePost.at<float>(3)) > 0.003){
                    //     bool abcd;
                    //     KF.statePost.at<float>(3)>vy?abcd = true:abcd = false;
                    //     if(abcd != speedSymbol){
                    //         // firstKF = true;
                            // cout << "______________________" << endl;
                    //         // setIdentity(KF.errorCovPost, Scalar::all(2.5e-7));
                    //     }
                    //     speedSymbol = abcd;
                    // }
                    // cout << KF.statePost << endl;
                    vp = KF.statePost.at<float>(2);
                    vy = KF.statePost.at<float>(3);
                    cout << vp * 1000 << "," << vy * 1000 << endl;
                    // cout << KF.errorCovPre << endl;
                    padd = KF.statePost.at<float>(0) - measurement.at<float>(0);
                    yadd = KF.statePost.at<float>(1) - measurement.at<float>(1);
                    // cout << KF.statePost.at<float>(2) << "," << KF.statePost.at<float>(3) << endl;
                    // cout << flightTime << endl;
                    float flyPAdd = KF.statePost.at<float>(2) * (flightTime) ;
                    float flyYAdd = KF.statePost.at<float>(3) * (flightTime) ;
                    // cout << "ddddddd" <<flyPAdd << "," << flyYAdd << endl;
                    // cout << weightAngel << endl;

                    float evalAngle = 0.0;
                    for(int i = 0; i < 10; i++){
                        evalAngle += weightAngles[i];
                    }
                    evalAngle /= 10.0;
                    // cout << evalAngle << endl;
                    //if photo is not filp,should data-=weightAngel
                    // if(!isnan(weightAngel)) r.pitch.data += weightAngel;//photo filp
                    if(fabs(vp * 1000) > 0.5)
                    {
                        r.pitch.data += padd;
                        r.pitch.data += flyPAdd;
                    }
                    if(fabs(vy * 1000) > 0.5)
                    {
                        r.yaw.data += yadd;
                        r.yaw.data += flyYAdd;
                    }
                    r.pitch.data -= evalAngle;
                    
                    // if(fabs(vy * 1000) > 1.0) r.yaw.data += yadd;
                    
                    // if(fabs(vy * 1000) > 1.0) r.yaw.data += flyYAdd;
                    // cout << padd << "," << yadd << endl;
                    // cout << distance << endl;
                    // if(fabs(flyPAdd) > 1.0) r.pitch.data += flyPAdd;
                    // if(fabs(flyYAdd) > 1.0) r.yaw.data += flyYAdd;
                    // cout << distance << endl;
                    r.rank = floor(distance/10);
                    // cout << r.pitch.data << endl;
                    // if(fabs(r.pitch.data) > 5) r.pitch.data /= 1.4;
                    // if(fabs(r.yaw.data) > 5) r.yaw.data /= 1.4;
                    // r.pitch.data -= 2.0;
                    r.yaw.data = -r.yaw.data;
                    // r.pitch.data = -r.pitch.data;
                    std::cout << "data" <<r.pitch.data << "  " << r.yaw.data << std::endl;
                    // if(yadd > 0.1) cout << "yes" << endl;
                    serialPort.Send(r);


                }else{

                    cout << "target lost_______________________"<< weightAngleNumber << endl;
                    firstKF = true;
                }

                
                // else{
                //     controlment.at<float>(0) = 0.0;
                //     controlment.at<float>(framevPost, Scalar::all(1));                                  
                //     randn(KF.statePost, Scalar::all(0), Scalar::all(1));
                // }
                
                // rm::debug::DrawLightBars(lightBars, frame, -1);
                // rm::debug::DrawArmours(armours, frame, -1);
                // cv::imshow("frame", frame);
                // if(!binary.empty()) cv::imshow("binary", binary);
                // cv::waitKey(1);

                delete imgPtr; 
            } else {
                status = false;
            }
        }
                       

    });

    serialPortThread.join();
    detectionThread.join();
    return 0;
}
