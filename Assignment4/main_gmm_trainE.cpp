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
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/100_0109.png";
	//const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/b4nature_animals_land009.png";
	const string imagePath = "C:/Users/Administrator/Desktop/ecse415/a4/cheeky_penguin.png";
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

	//Create an object from the OpenCV EM class 
	const int cov_mat_type = cv::EM::COV_MAT_GENERIC;
	TermCriteria term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 300, 0.1);
	EM gmm(num_clusters);

	// Create the feature matrix, as an 𝐻𝑊 by 𝐷 Mat object
	Mat feature(img.rows * img.cols, 3, CV_64FC1);
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			for (int z = 0; z < 3; z++)
				feature.at<double>(y + x*img.rows, z) = img.at<Vec3b>(y, x)[z];

	//Create initial means, as an 𝑛𝑢𝑚C𝑙𝑢𝑠𝑡𝑒𝑟𝑠 by 𝐷 Mat object of CV_64F type
	Mat initialMeans(num_clusters, 3, CV_64FC1);
	double fR = 0.0, fG = 0.0, fB = 0.0, bR = 0.0, bG = 0.0, bB = 0.0;
	int forgroundCount = 0, backgroundCount = 0;
	for (int r = 0; r< img.rows; r++){
		for (int c = 0; c < img.cols; c++){
			for (int z = 0; z < 3; z++){
				//R
				if (z == 0){
					if ((c > rect.x) && (c<rect.x + rect.width) && (r>rect.y) && (r < rect.y + rect.height)){
						fR = fR + img.at<Vec3b>(r, c)[z];
						forgroundCount++;
					}
					else{
						bR = bR + img.at<Vec3b>(r, c)[z];
						backgroundCount++;
					}
				}
				//G
				else if (z == 1){
					if ((c > rect.x) && (c<rect.x + rect.width) && (r>rect.y) && (r < rect.y + rect.height)){
						fG = fG + img.at<Vec3b>(r, c)[z];

					}
					else{
						bG = bG + img.at<Vec3b>(r, c)[z];
					}
				}
				//B
				else if (z == 2){
					if ((c > rect.x) && (c<rect.x + rect.width) && (r>rect.y) && (r < rect.y + rect.height)){
						fB = fB + img.at<Vec3b>(r, c)[z];
					}
					else{
						bB = bB + img.at<Vec3b>(r, c)[z];
					}
				}
			}
		}
	}
	initialMeans.at<double>(0, 0) = bR / (double)backgroundCount;
	initialMeans.at<double>(0, 1) = bG / (double)backgroundCount;
	initialMeans.at<double>(0, 2) = bB / (double)backgroundCount;
	initialMeans.at<double>(1, 0) = fR / (double)forgroundCount;
	initialMeans.at<double>(1, 1) = fG / (double)forgroundCount;
	initialMeans.at<double>(1, 2) = fB / (double)forgroundCount;


	//Create initial covariance matrices as a vector<Mat> object
	vector<Mat> initialCovariance;
	// calculate background covariance
	Vec3f sumOfPixels = Vec3f(0, 0, 0);
	float sumRR = 0, sumRG = 0, sumRB = 0, sumGG = 0, sumGB = 0, sumBB = 0;
	Mat covarianceMat = Mat::zeros(3, 3, CV_64FC1);
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c) {
			if ((c > rect.x) && (c<rect.x + rect.width) && (r>rect.y) && (r < rect.y + rect.height))
			{
				const Vec3b &currentPixel = img.at<Vec3b>(Point(c, r));
				sumOfPixels += Vec3b(currentPixel[0], currentPixel[1], currentPixel[2]);
				sumRR += currentPixel[0] * currentPixel[0];
				sumRG += currentPixel[0] * currentPixel[1];
				sumRB += currentPixel[0] * currentPixel[2];
				sumGG += currentPixel[1] * currentPixel[1];
				sumGB += currentPixel[1] * currentPixel[2];
				sumBB += currentPixel[2] * currentPixel[2];
			}
		}
	}
	int nPixels = img.rows * img.cols;
	assert(nPixels > 0);
	Vec3f avgOfPixels = sumOfPixels / nPixels;
	covarianceMat.at<double>(0, 0) = sumRR / nPixels - avgOfPixels[0] * avgOfPixels[0];
	covarianceMat.at<double>(0, 1) = sumRG / nPixels - avgOfPixels[0] * avgOfPixels[1];
	covarianceMat.at<double>(0, 2) = sumRB / nPixels - avgOfPixels[0] * avgOfPixels[2];

	covarianceMat.at<double>(1, 1) = sumGG / nPixels - avgOfPixels[1] * avgOfPixels[1];
	covarianceMat.at<double>(1, 2) = sumGB / nPixels - avgOfPixels[1] * avgOfPixels[2];
	covarianceMat.at<double>(2, 2) = sumBB / nPixels - avgOfPixels[2] * avgOfPixels[2];

	covarianceMat.at<double>(1, 0) = covarianceMat.at<double>(0, 1);
	covarianceMat.at<double>(2, 0) = covarianceMat.at<double>(0, 2);
	covarianceMat.at<double>(2, 1) = covarianceMat.at<double>(1, 2);
	cout << "covariance of image: " << covarianceMat << endl;
	initialCovariance.push_back(covarianceMat);

	// calculate foreground covariance
	//assert(img.type() == img_copy.type());
	Mat img_copy(img, rect);
	//imshow("img_copy", img_copy);
	//waitKey();
	//img.reshape(1, rect.height*rect.width);
	Mat img_copy_f = img;
	img_copy_f = img_copy_f.reshape(1, img_copy_f.rows*img_copy_f.cols);
	Mat covar, mean;
	calcCovarMatrix(img_copy_f, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	covar /= (img_copy_f.rows * img_copy_f.cols);
	cout << "covariance through opencv: " << covar << std::endl;
	initialCovariance.push_back(covar);

	//Create initial weight matrix of mixture component as a 1 by 𝑛𝑢𝑚𝐶𝑙𝑢𝑠𝑡𝑒𝑟𝑠 floating-point
	//Mat object
	Mat weightMatrix(1, num_clusters, CV_64F);
	int forgroundPixels = 0;
	int backgroundPixels = 0;
	for (int r = 0; r < img.rows; ++r) {
		for (int c = 0; c < img.cols; ++c)
		{
			if ((c > rect.x) && (c<rect.x + rect.width) && (r>rect.y) && (r < rect.y + rect.height))
			{
				forgroundPixels++;
			}
			else{
				backgroundPixels++;
			}
		}
	}
	weightMatrix.at<double>(0, 1) = (double)forgroundPixels / (double)nPixels;
	weightMatrix.at<double>(0, 0) = (double)backgroundPixels / (double)nPixels;

	//Use the trainE method of the EM object to estimate the GMM parameters and the pixel labels.
	Mat logLikelihoods;
	Mat labels;
	Mat probs;
	gmm.trainE(feature, initialMeans, initialCovariance, weightMatrix, logLikelihoods, labels, probs);

	//	cout << "lables" << labels << endl;
	//	cout << "probs" << probs << endl;
	//	waitKey(0);
	//Display result
	vector<Mat> segmented;
	for (int i = 0; i < num_clusters; i++)
		segmented.push_back(Mat::zeros(img.rows, img.cols, CV_8UC3));

	//img.convertTo(float_img, CV_32F);
	int index = 0;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int result = labels.at<int>(y + x*img.rows, 0);
			//			if (probs.at<double>(index, 0)>probs.at<double>(index, 1)){
			//				result = 0;
			//			}
			//			else{
			//				result = 1;
			//			}
			segmented[result].at<Vec3b>(y, x)[0] = img.at<Vec3b>(y, x)[0];
			segmented[result].at<Vec3b>(y, x)[1] = img.at<Vec3b>(y, x)[1];
			segmented[result].at<Vec3b>(y, x)[2] = img.at<Vec3b>(y, x)[2];
		}
	}
	
	imshow("background", segmented[0]);
	imwrite("cheeky_penguin_gmm_rgb_background.jpg", segmented[0]);
	//waitKey(0);

	imshow("foreground", segmented[1]);
	imwrite("cheeky_penguin_gmm_rgb_foreground.jpg", segmented[1]);
	waitKey(0);
}