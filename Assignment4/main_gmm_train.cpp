#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

Mat source;
bool leftDown = false, leftup = false;
Point cor1, cor2;
//create a corresponding rectangular
Rect rect;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		//cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		leftDown = true;
		cor1.x = x;
		cor1.y = y;
		cout << "Corner 1: " << cor1 << endl;

	}
	if (event == EVENT_LBUTTONUP)
	{
		leftup = true;
		cor2.x = x;
		cor2.y = y;
		cout << "Corner 2: " << cor2 << endl;
	}
	if (leftDown == true && leftup == false) //when the left button is down
	{
		Point pt;
		pt.x = x;
		pt.y = y;
		Mat temp_img = source.clone();
		rectangle(temp_img, cor1, pt, Scalar(0, 0, 255)); //drawing a rectangle continuously
		imshow("original", temp_img);

	}
	if (leftDown == true && leftup == true) //when the selection is done
	{
		rect.width = abs(cor1.x - cor2.x);
		rect.height = abs(cor1.y - cor2.y);
		rect.x = min(cor1.x, cor2.x);
		rect.y = min(cor1.y, cor2.y);
		cout << "rectangular captured." << endl;
		cout << "x: " << rect.x << endl;
		cout << "y: " << rect.y << endl;
		cout << "height: " << rect.height << endl;
		cout << "width: " << rect.width << endl;
		cout << "press any key to continue or choose again" << endl;
		leftDown = false;
		leftup = false;
	}
}

void main(void)
{
	const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109.png";
	const string imageGroundTruthPath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109_groundtruth.png";
	int num_clusters = 2;
	Mat gtMask;
	//load the image
	source = imread(imagePath);
	gtMask = imread(imageGroundTruthPath);
	//
	namedWindow("original", 1);
	//set the callback function for any mouse event, Prompt the user to draw a rectangle over the object
	setMouseCallback("original", CallBackFunc, NULL);
	imshow("original", source);
	waitKey();
	destroyWindow("original");
	/*cout << "x: " << rect.x << endl;
	cout << "y: " << rect.y << endl;
	cout << "height: " << rect.height << endl;
	cout << "width: " << rect.width << endl;*/

	//
	//ouput images
	cv::Mat meanImg(source.rows, source.cols, CV_32FC3);
	cv::Mat fgImg(source.rows, source.cols, CV_8UC3);
	cv::Mat bgImg(source.rows, source.cols, CV_8UC3);

	//convert the input image to float
	cv::Mat floatSource;
	source.convertTo(floatSource, CV_32F);

	//now convert the float image to column vector
	cv::Mat samples(source.rows * source.cols, 3, CV_32FC1);
	int idx = 0;
	for (int y = 0; y < source.rows; y++) {
		cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
		for (int x = 0; x < source.cols; x++) {
			samples.at<cv::Vec3f >(idx++, 0) = row[x];
		}
	}
	//we need just 2 clusters
	//cv::EM Params params(2);
	cout << "Starting EM training" << endl;
	EM em(2);
	em.train(samples);
	cout << "Finished training EM" << endl;

	vector<Mat> segmented;
	for (int i = 0; i < 2; i++)
		segmented.push_back(Mat::zeros(source.rows, source.cols, CV_8UC3));

	//now classify each of the source pixels
	idx = 0;
	for (int y = 0; y < source.rows; y++) {
		for (int x = 0; x < source.cols; x++) {

			//classify
			if ((x > rect.x) && (x<rect.x + rect.width) && (y>rect.y) && (y < rect.y + rect.height)){
				int result = em.predict(samples.row(idx++))[1];
				segmented[result].at<Vec3b>(y, x)[0] = source.at<Vec3b>(y, x)[0];
				segmented[result].at<Vec3b>(y, x)[1] = source.at<Vec3b>(y, x)[1];
				segmented[result].at<Vec3b>(y, x)[2] = source.at<Vec3b>(y, x)[2];
			}
			else{
				int result = em.predict(samples.row(idx++))[1];
				segmented[result].at<Vec3b>(y, x)[0] = 0;
				segmented[result].at<Vec3b>(y, x)[1] = 0;
				segmented[result].at<Vec3b>(y, x)[2] = 0;
			}
		}
	}
	cv::imshow("for", segmented[0]);
	cv::waitKey(0);
	cv::imshow("Background", segmented[1]);
	cv::waitKey(0);
}
// following part also works
void main(void)
{
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109.png";
	const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/b4nature_animals_land009.png";
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/cheeky_penguin.png";
	const string imageGroundTruthPath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109_groundtruth.png";
	int num_clusters = 2;
	Mat gtMask;
	//load the image
	source = imread(imagePath);
	gtMask = imread(imageGroundTruthPath);
	//
	namedWindow("original", 1);
	//set the callback function for any mouse event, Prompt the user to draw a rectangle over the object
	//	setMouseCallback("original", CallBackFunc, NULL);
	imshow("original", source);
	waitKey();
	destroyWindow("original");
	/*cout << "x: " << rect.x << endl;
	cout << "y: " << rect.y << endl;
	cout << "height: " << rect.height << endl;
	cout << "width: " << rect.width << endl;*/

	//convert the input image to float
	cv::Mat floatSource;
	Mat gray_img;
	cvtColor(source, gray_img, CV_RGB2GRAY);
	gray_img.convertTo(floatSource, CV_32F);

	//now convert the float image to column vector
	cv::Mat samples(source.rows * source.cols, 1, CV_32FC1);
	for (int y = 0; y < floatSource.rows; y++)
		for (int x = 0; x < floatSource.cols; x++)
			samples.at<float>(y + x*floatSource.rows, 0) = floatSource.at<float>(y, x);

	//we need just 2 clusters
	//cv::EM Params params(2);
	Mat logLikelihoods;
	Mat labels;
	Mat probs;
	cout << "Starting EM training" << endl;
	EM em(2);
	em.train(samples, logLikelihoods, labels, probs);
	cout << "Finished training EM" << endl;
	//
	vector<Mat> segmented;
	for (int i = 0; i < 2; i++)
		segmented.push_back(Mat::zeros(source.rows, source.cols, CV_8UC1));

	//now classify each of the source pixels
	for (int y = 0; y < source.rows; y++) {
		for (int x = 0; x < source.cols; x++) {
			int result = labels.at<int>(y + x*source.rows, 0);
			segmented[result].at<uchar>(y, x) = gray_img.at<uchar>(y, x);
		}
	}
	cv::imshow("foreground", segmented[0]);
	imwrite("b4nature_animals_land009_gmm_gray_foreground.jpg", segmented[0]);
	cv::waitKey(0);
	cv::imshow("Background", segmented[1]);
	imwrite("b4nature_animals_land009_gmm_gray_background.jpg", segmented[1]);
	cv::waitKey(0);
}