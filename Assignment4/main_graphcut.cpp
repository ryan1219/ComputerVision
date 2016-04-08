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

Mat img;
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
		Mat temp_img = img.clone();
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
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/b4nature_animals_land009.png";
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/cheeky_penguin.png";
	const string imageGroundTruthPath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109_groundtruth.png";
	int num_clusters = 2;
	Mat gtMask;
	//load the image
	img = imread(imagePath);
	gtMask = imread(imageGroundTruthPath);
	//
	namedWindow("original", 1);
	//set the callback function for any mouse event, Prompt the user to draw a rectangle over the object
	setMouseCallback("original", CallBackFunc, NULL);
	imshow("original", img);
	waitKey();
	destroyWindow("original");
	/*cout << "x: " << rect.x << endl;
	cout << "y: " << rect.y << endl;
	cout << "height: " << rect.height << endl;
	cout << "width: " << rect.width << endl;*/
	
	// run the grapCut function
	//Mat img_lab;
	Mat mask;
	Mat bgdModel, fgdModel;
	
	//convert to lab color space
	//cvtColor(img, img, CV_BGR2Lab);
	//imshow("img_lab", img); 
	//waitKey(0);
	//

	// GrabCut segmentation
	cv::grabCut(img,    // input image
		mask,   // segmentation result
		rect,// rectangle containing foreground
		bgdModel, fgdModel, // models
		1,        // number of iterations
		cv::GC_INIT_WITH_RECT); // use rectangle

	cout << "grabCut finish" << endl;

	Mat mask_foreground;
	Mat mask_background;
	Mat mask_background_a;
	Mat mask_background_b;

	// Get the pixels marked as likely foreground
	cv::compare(mask, cv::GC_PR_FGD, mask_foreground, cv::CMP_EQ);
	// Get the pixels marked as background
	cv::compare(mask, cv::GC_BGD, mask_background_a, cv::CMP_EQ);
	// Get the pixels marked as likely background
	cv::compare(mask, cv::GC_PR_BGD, mask_background_b, cv::CMP_EQ);
	mask_background = mask_background_a + mask_background_b;
	// Generate output image
	cv::Mat foreground(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat background(img.size(),CV_8UC3,cv::Scalar(255,255,255));
	img.copyTo(foreground, mask_foreground); // bg pixels not copied
	img.copyTo(background, mask_background);
	// draw rectangle on original image
	//cv::rectangle(image, rectangle, cv::Scalar(255, 255, 255), 1);

	imshow("img", img);

	imshow("100_0109_graphcut_foreground.jpg", foreground);
	imwrite("100_0109_graphcut_foreground.jpg", foreground);
	//Mat background = img - foreground;
	imshow("100_0109_graphcut_background.jpg", background);
	imwrite("100_0109_graphcut_background.jpg", background);
	waitKey(0);
}