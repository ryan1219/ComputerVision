#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
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
}

void main(void)
{
	const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109.png";
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/b4nature_animals_land009.png";
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/cheeky_penguin.png";
	const string imageGroundTruthPath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109_groundtruth.png";
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

	// Create the feature matrix, as an 𝐻𝑊 by 𝐷 Mat object
		Mat feature(img.rows * img.cols, 3, CV_32F);
		for (int y = 0; y < img.rows; y++){
			for (int x = 0; x < img.cols; x++){
				for (int z = 0; z < 3; z++){
					feature.at<float>(y + x*img.rows, z) = img.at<Vec3b>(y, x)[z];
				}
			}
		}
	// Create the feature matrix, as an 𝐻𝑊 by 1 Mat object
/*	Mat feature(img.rows * img.cols, 1, CV_32F);
	Mat gray_img;
	cvtColor(img, gray_img, CV_RGB2GRAY);

	imshow("gray image", gray_img);
	waitKey(0);

	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			feature.at<float>(y + x*img.rows, 0) = gray_img.at<uchar>(y, x);
*/
	// Create the initial label matrix, as an 𝐻𝑊 by 1 Mat object
	Mat label(img.rows * img.cols, 1, CV_32S);
	//Use the input rectangle and assign the initial labels of the points inside and outside
	//of the rectangle to 1 and 0, respectively
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if ((x > rect.x) && (x<rect.x + rect.width) && (y>rect.y) && (y < rect.y + rect.height))
			{
				label.at<int>(y + x*img.rows, 0) = 1;
			}
			else{
				label.at<int>(y + x*img.rows, 0) = 0;
			}
		}
	}
	// Run the kmeans function
	int clusterCount = 2;
	int attempts = 1;
	Mat centers;
	kmeans(feature, clusterCount, label, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	//
	//imshow("lable", label);
	//waitKey(0);

	//display result, if feature image is grayscale
/*	Mat new_image(gray_img.size(), gray_img.type());
	for (int y = 0; y < gray_img.rows; y++)
		for (int x = 0; x < gray_img.cols; x++)
		{
			if ((x > rect.x) && (x<rect.x + rect.width) && (y>rect.y) && (y < rect.y + rect.height))
			{
				int cluster_idx = label.at<int>(y + x*img.rows, 0);
				new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
				//new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
				//new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
			}
			else{
				new_image.at<uchar>(y, x) = 0;
				//new_image.at<Vec3b>(y, x)[1] = 0;
				//new_image.at<Vec3b>(y, x)[2] = 0;
			}
		}
*/

	//display results, if feature image is rgb
		Mat new_image(img.size(), img.type());
		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				if ((x > rect.x) && (x<rect.x + rect.width) && (y>rect.y) && (y < rect.y + rect.height))
				{
					int cluster_idx = label.at<int>(y + x*img.rows, 0);
					new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
					new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
					new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
				}
				else{
					new_image.at<Vec3b>(y, x)[0] = 0;
					new_image.at<Vec3b>(y, x)[1] = 0;
					new_image.at<Vec3b>(y, x)[2] = 0;
				}
			}
		}
	imshow("clustered image", new_image);
	imwrite("cheeky_penguin_kmean_grayscale.jpg", new_image);
	waitKey(0);
}