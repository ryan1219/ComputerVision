#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2\stitching\detail\blenders.hpp>
#include <stdlib.h>

using namespace cv;
using namespace std;

// Functions prototypes
void Homography(const vector<Mat> &Images, vector<Mat> &transforms);
void FindOutputLimits(const vector<Mat> &Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax);
void warpMasks(const vector<Mat> &Images, vector<Mat> &masks_warped, const vector<Mat> &transforms, const Mat &panorama);
void warpImages(const vector<Mat> &Images, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama);
void BlendImages(const vector<Mat> &Images, Mat &pano_feather, Mat &pano_multiband, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama);
int main()
{
	// Initialize OpenCV nonfree module
	initModule_nonfree();			

	// Set the dir/name of each image 
	const int NUM_IMAGES = 6;
	const string IMG_NAMES[] = { "C:/Users/Administrator/Desktop/a3/0.jpg",
		"C:/Users/Administrator/Desktop/a3/1.jpg",
		"C:/Users/Administrator/Desktop/a3/2.jpg",
		"C:/Users/Administrator/Desktop/a3/3.jpg",
		"C:/Users/Administrator/Desktop/a3/4.jpg",
		"C:/Users/Administrator/Desktop/a3/5.jpg",
	};
	/*{ "C:/Users/Administrator/Desktop/a3/0.jpg",
		"C:/Users/Administrator/Desktop/a3/5.jpg",
		"C:/Users/Administrator/Desktop/a3/4.jpg",
		"C:/Users/Administrator/Desktop/a3/3.jpg",
		"C:/Users/Administrator/Desktop/a3/2.jpg",
		"C:/Users/Administrator/Desktop/a3/1.jpg",
	};
	
	/*{ "C:/Users/Administrator/Desktop/field/FieldB09.JPG",
		"C:/Users/Administrator/Desktop/field/FieldB10.JPG",
		"C:/Users/Administrator/Desktop/field/FieldB11.JPG",
		"C:/Users/Administrator/Desktop/field/FieldC11.JPG",
		"C:/Users/Administrator/Desktop/field/FieldC10.JPG",
		"C:/Users/Administrator/Desktop/field/FieldC09.JPG", };
		*/
	
		
	// Load the images

	vector<Mat> Images;
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		Images.push_back(imread(IMG_NAMES[i]));
	}
	////display images
	//for (int i = 0; i < NUM_IMAGES; i++)
	//{
	//	imshow("images", Images[i]);
	//	waitKey(0);
	//}
	
	// 1. Initialize all the transforms to the 3x3 identity matrix
	vector<Mat> transforms;
	//cout << "transforms vector size: "<<transforms.size() << endl;
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		transforms.push_back(Mat::eye(3, 3, CV_64F));
		// << "transforms matrix initial " << transforms[i] << endl;
	}
	cout << "transforms vector size: "<<transforms.size() << endl;

	//// 2. Calculate the transformation matrices
	Homography(Images, transforms);
	//print out the transforms
	for (int k = 0; k < transforms.size(); k++)
	{
		cout << "transforms"<<"= " << transforms[k] << endl;
	}
	//// 3. Compute the min and max limits of the transformations
	int xMin, xMax, yMin, yMax;
	FindOutputLimits(Images, transforms, xMin, xMax, yMin, yMax);

	//// 4. Compute the size of the panorama
	////
	//// 5. Initialize the panorama image		
	Mat panorama = Mat(yMax - yMin + 1, xMax - xMin + 1, CV_64F);

	cout << "size of panorama: " << panorama.size() << endl;
	//// 6. Initialize warped mask images
	vector<Mat> masks_warped;
	masks_warped.resize(NUM_IMAGES);

	//cout << "size of masks_warped: " << masks_warped.size() << endl;
	//// 7. Warp image masks
	warpMasks(Images, masks_warped, transforms, panorama);
	
	////display masks_warped
	//imshow("masks_warped", masks_warped[0]);
	//imwrite("C:/Users/Administrator/Desktop/ masks_warped0.jpg", masks_warped[0]);
	//waitKey(0);

	//imshow("masks_warped", masks_warped[1]);
	//imwrite("C:/Users/Administrator/Desktop/ masks_warped1.jpg", masks_warped[1]);
	//waitKey(0);
	/*for (int i = 0; i < masks_warped.size(); i++)
	{
		cout << "masks_warped"<<i<<"= " << masks_warped[i] << endl;
	}*/

	//// 8. Warp the images
	warpImages(Images, masks_warped, transforms, panorama);
	//Display the panorama image and wait for a user keypress
	imshow("panorama", panorama);
	imwrite("C:/Users/Administrator/Desktop/panorama_result.jpg", panorama);
	waitKey(0);

	////// 9. Initialize the blended panorama images	
	//Mat  pano_feather = Mat(yMax - yMin + 1, xMax - xMin + 1, CV_64F);
	//Mat  pano_multiband = Mat(yMax - yMin + 1, xMax - xMin + 1, CV_64F);
	//
	////// 10. Blend
	//BlendImages(Images, pano_feather, pano_multiband, masks_warped, transforms, panorama);
	//
	//imshow("pano_feather", pano_feather);
	//imwrite("C:/Users/Administrator/Desktop/pano_feather.jpg", pano_feather);
	//waitKey(0);
	//
	//imshow("pano_multiband", pano_multiband);
	//imwrite("C:/Users/Administrator/Desktop/pano_multiband.jpg", pano_multiband);
	//waitKey(0);

	system("pause");
	return 0;
}

void Homography(const vector<Mat> &Images, vector<Mat> &transforms)
{
	//for each image 𝐼[𝑛], starting from the second image(𝑛 = 1)
	for (int i = 1; i < Images.size(); i++)
	{		
		//a.Detect and extract SIFT key points and descriptors for 𝐼[𝑛] and 𝐼[𝑛 − 1]
		//create a sift feature detector object
		Ptr<FeatureDetector> FeatureDetector = FeatureDetector::create("SIFT");
		//Create a SIFT descriptor extractor object
		Ptr<DescriptorExtractor> DescriptorExtractor = DescriptorExtractor::create("SIFT");
		// KePoints 
		vector<KeyPoint> kp_n;
		vector<KeyPoint> kp_n_1;
		//Detect SIFT key points
		FeatureDetector->detect(Images[i - 1], kp_n_1);
		FeatureDetector->detect(Images[i], kp_n);
		// To store the SIFT descriptor
		Mat descriptor_n;
		Mat descriptor_n_1;
		//compute sift descriptor
		DescriptorExtractor->compute(Images[i - 1], kp_n_1, descriptor_n_1);
		DescriptorExtractor->compute(Images[i], kp_n, descriptor_n);

		//b. Match the descriptors of the two images using the OpenCV BFMatcher class
		BFMatcher matcher(NORM_L1, false);
		vector<DMatch > matches;
		matcher.match(descriptor_n_1, descriptor_n, matches);

		//c. Use the OpenCV drawMatches() function to draw the matched key points 
		cv::Mat img_matches;
		drawMatches(Images[i - 1], kp_n_1, Images[i], kp_n, matches, img_matches, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
					
		////display matched images
		//imshow("image match", img_matches);
		//string address = "C:/Users/Administrator/Desktop/matchedKp"+to_string(i)+".jpg";
		//imwrite(address, img_matches);
		//waitKey(0);

		//d.Store the pixel coordinates of the matched key points into two vector<Point2d> objects
		vector<Point2d> image_n_matched_kp;
		vector<Point2d> image_n_1_matched_kp;
		for (int j = 0; j < matches.size(); j++)
		{
			image_n_matched_kp.push_back(kp_n[matches[j].trainIdx].pt);
			image_n_1_matched_kp.push_back(kp_n_1[matches[j].queryIdx].pt);
		}

		//e. Estimate the transformation between 𝐼[𝑛] and 𝐼[𝑛 − 1]
		Mat h = findHomography(image_n_matched_kp, image_n_1_matched_kp, CV_RANSAC);
		//cout << "h= "<< h << endl;
		//cout << "transforms2= " << transforms[i] << endl;

		//f. Compute the transformation between 𝐼[𝑛] and 𝐼[0]	
		transforms[i] = transforms[i - 1] * h;
		
		//cout <<"T = " << transforms[i]<<endl;
		//string s = "C:/Users/Administrator/Desktop/descriptors" + to_string(i) + ".txt";
		//cv::FileStorage fsWrite(s, FileStorage::WRITE);
		//fsWrite << "descriptors" << transforms[i];
		//fsWrite.release();
	}
}

void FindOutputLimits(const vector<Mat> &Images, vector<Mat> &transforms, int &xMin, int &xMax, int &yMin, int &yMax)
{
	vector<Mat> all_projected_corners;
	//1) For each transformation matrix 𝑇[𝑛], find the new coordinates of the image corners
	for (int i = 1; i < transforms.size(); i++)
	{
		Mat c1 = (Mat_<double>(3, 1) << 0, 0, 1);
		//cout << "c1" << c1 << endl;
		Mat c2 = (Mat_<double>(3, 1) << 0, Images[i].rows-1, 1);
		//cout << "c2" << c2 << endl;
		Mat c3 = (Mat_<double>(3, 1) << Images[i].cols-1, 0, 1);
		Mat c4 = (Mat_<double>(3, 1) << Images[i].cols - 1, Images[i].rows - 1, 1);
		Mat tc1 = transforms[i] * c1;
		//cout << "tc1" << tc1 << endl;
		Mat tc2 = transforms[i] * c2;
		Mat tc3 = transforms[i] * c3;
		Mat tc4 = transforms[i] * c4;
		all_projected_corners.push_back(tc1);
		all_projected_corners.push_back(tc2);
		all_projected_corners.push_back(tc3);
		all_projected_corners.push_back(tc4);
	}

	for (int i = 0; i < all_projected_corners.size(); i++)
	{
		cout << "corners = " << all_projected_corners[i] << endl;
	}
	cout << "nubmer of corner: " << all_projected_corners.size() << endl;

	//Find the min and the max coordinates of all projected corners of all images 
	xMin = (int)all_projected_corners[0].at<double>(0, 0);
	xMax = (int)all_projected_corners[0].at<double>(0, 0);
	yMin = (int)all_projected_corners[0].at<double>(1, 0);
	yMax = (int)all_projected_corners[0].at<double>(1, 0);
	for (int i = 0; i < all_projected_corners.size(); i++)
	{		 
		//cout << "all_projected_corners[i].at<double>(1, 1)" << all_projected_corners[i].at<double>(0, 0) << endl;
		if ((int)all_projected_corners[i].at<double>(0, 0) < xMin)
		{
			xMin = (int)all_projected_corners[i].at<double>(0, 0);
		}
		else if (all_projected_corners[i].at<double>(0, 0) > (double)xMax)
		{
			xMax = (int)all_projected_corners[i].at<double>(0, 0);
		}

		if (all_projected_corners[i].at<double>(1, 0) < (double)yMin)
		{
			yMin = (int)all_projected_corners[i].at<double>(1, 0);
		}
		else if (all_projected_corners[i].at<double>(1, 0)>(double)yMax)
		{
			yMax = (int)all_projected_corners[i].at<double>(1, 0);
		}
	}
	cout << "xMin: " << xMin << endl;
	cout << "xMax: " << xMax << endl;
	cout << "yMin: " << yMin << endl;
	cout << "yMax: " << yMax << endl;
	//3)
	//Create the translation matrix as a 3×3 Mat object and initialize it to the identity
	//	matrix with the type of CV_64F
	Mat translation_matrix = Mat::eye(3, 3, CV_64F);
	translation_matrix.at<double>(0, 2) = -(double)xMin;
	translation_matrix.at<double>(1, 2) = -(double)yMin;

	//display translation matrix
	cout << "translation matrix= " << translation_matrix << endl;
	//Apply the translation matrix to all the transformation matrices by multiplying from left
	for (int i = 1; i < transforms.size(); i++)
	{
		transforms[i] =  translation_matrix*transforms[i];
	}
}

void warpMasks(const vector<Mat> &Images, vector<Mat> &masks_warped, const vector<Mat> &transforms, const Mat &panorama)
{
	vector<Mat> masks;
	masks.resize(Images.size());
	//Create image masks as a vector<Mat> masks object
	for (int i = 0; i < Images.size(); i++)
	{
		Mat m(Images[i].rows, Images[i].cols, CV_8U, Scalar(255));
		masks[i] = m;
	}
	//Calculate warped mask images
	for (int i = 0; i < Images.size(); i++)
	{
		warpPerspective(masks[i], masks_warped[i], transforms[i], panorama.size());
		//cout << "first masks size" << masks_warped[i].size() << endl;
		//cout << "first masks warped= " << masks_warped[i] << endl;
		//system("pause");
	}

}

void warpImages(const vector<Mat> &Images, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama)
{
	//Images[0].copyTo(panorama, masks_warped[0]);
	for (int i = 0; i < Images.size(); i++)
	{
		Mat warp_image;
		//Warp the image using 𝑇[𝑛] and the OpenCV warpPerspective function
		warpPerspective(Images[i], warp_image, transforms[i], panorama.size(), BORDER_CONSTANT, 1);
				
		//Copy the non - zero pixels of the warped image to the panorama image using 𝑚𝑎𝑠𝑘𝑠_𝑤𝑎𝑟𝑝𝑒𝑑[𝑛]
		warp_image.copyTo(panorama, masks_warped[i]);
		
		/*imshow("warped_image", warp_image);
		string address = "C:/Users/Administrator/Desktop/warped_image"+to_string(i)+".jpg";
		imwrite(address, warp_image);
		waitKey(0);*/
	}
}

void BlendImages(const vector<Mat> &Images, Mat &pano_feather, Mat &pano_multiband, const vector<Mat> &masks_warped, const vector<Mat> &transforms, Mat &panorama)
{
	//2) Create an OpenCV’s detail::FeatherBlender and detail::MultiBandBlender objects
	detail::FeatherBlender FeatherBlender;
	detail::MultiBandBlender MultiBandBlender;
	//3)prepare the blenders
	FeatherBlender.prepare(Rect(0, 0, pano_feather.cols, pano_feather.rows));
	MultiBandBlender.prepare(Rect(0, 0, pano_feather.cols, pano_feather.rows));
	//4)feed the images to the blenders
	for (int i = 0; i < Images.size(); i++)
	{
		Mat warp_image;
		//Warp the image using 𝑇[𝑛] and the OpenCV warpPerspective function
		warpPerspective(Images[i], warp_image, transforms[i], panorama.size(), BORDER_CONSTANT, 1);
		//Convert the type of the warped image to CV_16S
		Mat warp_image_converted;
		warp_image.convertTo(warp_image_converted, CV_16S);
		//Feed the warped image to the blender objects
		FeatherBlender.feed(warp_image_converted, masks_warped[i], Point(0, 0));
		MultiBandBlender.feed(warp_image_converted, masks_warped[i], Point(0, 0));
	}
	//5) Blend the images
	Mat a;
	FeatherBlender.blend(pano_feather, a);
	Mat b;
	MultiBandBlender.blend(pano_multiband, b);
	//Convert the type of the pano_feather and pano_multiband to CV_8U
	pano_feather.convertTo(pano_feather, CV_8U);
	pano_multiband.convertTo(pano_multiband, CV_8U);
}