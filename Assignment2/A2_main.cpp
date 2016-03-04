#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace cv;
using namespace std;

/* Helper class declaration and definition */
class Caltech101
{
public:
	Caltech101::Caltech101(string datasetPath, const int numTrainingImages, const int numTestImages)
	{	
		cout << "Loading Caltech 101 dataset" << endl;
		numImagesPerCategory = numTrainingImages + numTestImages;

		// load "Categories.txt"
		ifstream infile(datasetPath + "/" + "Categories.txt");
		cout << "\tChecking Categories.txt" << endl;
		if (!infile.is_open())
		{
			cout << "\t\tError: Cannot find Categories.txt in " << datasetPath << endl;
			return;
		}
		cout << "\t\tOK!" << endl;

		// Parse category names
		cout << "\tParsing category names" << endl;
		string catname;
		while (getline(infile, catname))
		{
			categoryNames.push_back(catname);
		}
		cout << "\t\tdone!" << endl;

		// set num categories
		int numCategories = (int)categoryNames.size();

		// initialize outputs size
		trainingImages = vector<vector<Mat>>(numCategories);
		trainingAnnotations = vector<vector<Rect>>(numCategories);
		testImages = vector<vector<Mat>>(numCategories);
		testAnnotations = vector<vector<Rect>>(numCategories);

		// generate training and testing indices
		randomShuffle();		

		// Load data
		cout << "\tLoading images and annotation files" << endl;
		string imgDir = datasetPath + "/" + "Images";
		string annotationDir = datasetPath + "/" + "Annotations";
		for (int catIdx = 0; catIdx < categoryNames.size(); catIdx++)
		//for (int catIdx = 0; catIdx < 1; catIdx++)
		{
			string imgCatDir = imgDir + "/" + categoryNames[catIdx];
			string annotationCatDir = annotationDir + "/" + categoryNames[catIdx];
			for (int fileIdx = 0; fileIdx < numImagesPerCategory; fileIdx++)
			{
				// use shuffled training and testing indices
				int shuffledFileIdx = indices[fileIdx];
				// generate file names
				stringstream imgFilename, annotationFilename;
				imgFilename << "image_" << setfill('0') << setw(4) << shuffledFileIdx << ".jpg";
				annotationFilename << "annotation_" << setfill('0') << setw(4) << shuffledFileIdx << ".txt";

				// Load image
				string imgAddress = imgCatDir + '/' + imgFilename.str();
				Mat img = imread(imgAddress, CV_LOAD_IMAGE_COLOR);
				// check image data
				if (!img.data)
				{
					cout << "\t\tError loading image in " << imgAddress << endl;
					return;
				}

				// Load annotation
				string annotationAddress = annotationCatDir + '/' + annotationFilename.str();
				ifstream annotationIFstream(annotationAddress);
				// Checking annotation file
				if (!annotationIFstream.is_open())
				{
					cout << "\t\tError: Error loading annotation in " << annotationAddress << endl;
					return;
				}
				int tl_col, tl_row, width, height;
				Rect annotRect;
				while (annotationIFstream >> tl_col >> tl_row >> width >> height)
				{
					annotRect = Rect(tl_col - 1, tl_row - 1, width, height);					
				}

				// Split training and testing data
				if (fileIdx < numTrainingImages)
				{
					// Training data
					trainingImages[catIdx].push_back(img);
					trainingAnnotations[catIdx].push_back(annotRect);
				}
				else
				{
					// Testing data
					testImages[catIdx].push_back(img);
					testAnnotations[catIdx].push_back(annotRect);
				}				
			}			
		}
		cout << "\t\tdone!" << endl;		
		successfullyLoaded = true;
		cout << "Dataset successfully loaded: " << numCategories << " categories, " << numImagesPerCategory  << " images per category" << endl << endl;
	}

	bool isSuccessfullyLoaded()	{  return successfullyLoaded; }

	void dispTrainingImage(int categoryIdx, int imageIdx)
	{		
		Mat image = trainingImages[categoryIdx][imageIdx];
		Rect annotation = trainingAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated training image", image);
		waitKey(0);
		destroyWindow("Annotated training image");
	}
	
	void dispTestImage(int categoryIdx, int imageIdx)
	{
		Mat image = testImages[categoryIdx][imageIdx];
		Rect annotation = testAnnotations[categoryIdx][imageIdx];
		rectangle(image, annotation, Scalar(255, 0, 255), 2);
		imshow("Annotated test image", image);
		waitKey(0);
		destroyWindow("Annotated test image");
	}

	vector<string> categoryNames; 
	vector<vector<Mat>> trainingImages;
	vector<vector<Rect>> trainingAnnotations;
	vector<vector<Mat>> testImages;
	vector<vector<Rect>> testAnnotations;

private:
	bool successfullyLoaded = false;
	int numImagesPerCategory;
	vector<int> indices;
	void randomShuffle()
	{
		// set init values
		for (int i = 1; i <= numImagesPerCategory; i++) indices.push_back(i);

		// permute using built-in random generator
		random_shuffle(indices.begin(), indices.end());		
	}
};

/* Function prototypes */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors);

void main(void)
{
	/* Initialize OpenCV nonfree module */
	initModule_nonfree();

	/* Put the full path of the Caltech 101 folder here */
	const string datasetPath = "C:/Users/Administrator/Desktop/Caltech 101";

	/* Set the number of training and testing images per category */
	const int numTrainingData = 40;
	const int numTestingData = 2;

	/* Set the number of codewords*/
	const int numCodewords = 400; 

	/* Load the dataset by instantiating the helper class */
	Caltech101 Dataset(datasetPath, numTrainingData, numTestingData);

	/* Terminate if dataset is not successfull loaded */
	if (!Dataset.isSuccessfullyLoaded())
	{
		cout << "An error occurred, press Enter to exit" << endl;
		getchar();
		return;
	}	
	
	/* Variable definition */
	Mat codeBook;	
	std::vector<vector<Mat>> imageDescriptors(Dataset.trainingImages.size(), std::vector<Mat>(numTrainingData));

	/* Training */	
	Train(Dataset, codeBook, imageDescriptors, numCodewords);

	/* Testing */	
	Test(Dataset, codeBook, imageDescriptors);
	system("pause");
}

/* Train BoW */
void Train(const Caltech101 &Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
{
	//create a sift feature detector object
	Ptr<FeatureDetector> FeatureDetector = FeatureDetector::create("SIFT");
	//Create a SIFT descriptor extractor object
	Ptr<DescriptorExtractor> DescriptorExtractor = DescriptorExtractor::create("SIFT");
	//Create a Mat object to store all the SIFT descriptors of all training images of all categories
	Mat D;
//
//	//For each training image of each object category
//	for (int i = 0; i < Dataset.trainingImages.size(); i++){
//		for (int j = 0; j < Dataset.trainingImages[i].size(); j++){
//
//			//
//			Mat image_sift;
//			// KePoints 
//			vector<KeyPoint> kp;
//			//Detect SIFT key points in I
//			FeatureDetector->detect(Dataset.trainingImages[i][j], kp);
//			//draw the sift keypoints and visually inspect them
////			drawKeypoints(Dataset.trainingImages[i][j], kp, image_sift);
//			//
////			namedWindow("first_sift", CV_WINDOW_AUTOSIZE);
////			imshow("first_sift", image_sift);		
////			waitKey(0);
//			//
//			//Discard the keypoints outside of the annotation rectangle
//			int size = kp.size();
//			for (int k = 0; k < size; k++){
//				if (Dataset.trainingAnnotations[i][j].x + Dataset.trainingAnnotations[i][j].width < kp[k].pt.x || kp[k].pt.x < Dataset.trainingAnnotations[i][j].x || kp[k].pt.y >Dataset.trainingAnnotations[i][j].y + Dataset.trainingAnnotations[i][j].height || kp[k].pt.y < Dataset.trainingAnnotations[i][j].y)
//				{
//					kp.erase(kp.begin()+k);
//				}
//				if (size != kp.size()){
//					--k;
//					size = kp.size();
//				}
//			}
//
//			/*
//			//draw discard key points
//			drawKeypoints(Dataset.trainingImages[i][j], kp, image_sift);
//			//draw rectangle
//			Rect annotation = Dataset.trainingAnnotations[i][j];
//			rectangle(image_sift, annotation, Scalar(255, 0, 255), 2);
//			//display image
//			namedWindow("discardkp_sift", CV_WINDOW_AUTOSIZE);
//			imshow("discardkp_sift", image_sift);
//			imwrite("C:/Users/Administrator/Desktop/result2.jpg", image_sift);
//			waitKey(0);
//   			*/
//			// To store the SIFT descriptor of current image
//			Mat descriptor;
//			//compute sift descriptor
//			DescriptorExtractor->compute(Dataset.trainingImages[i][j], kp, descriptor);
//			D.push_back(descriptor);
//		}
//	}

	/*
	//save descriptor D into txt file for later use, don't need to calculate descriptor each time
	cv::FileStorage fsWrite("C:/Users/Administrator/Desktop/YourDescriptors.txt", FileStorage::WRITE);
	fsWrite<< "descriptors" << D;
	fsWrite.release();
	*/
	//read discriptor saved in disc
	Mat descriptor_from_disc;
	cv::FileStorage fsRead("C:/Users/Administrator/Desktop/YourDescriptors.txt", FileStorage::READ);
	fsRead["descriptors"] >> descriptor_from_disc;
	fsRead.release();
	/*test code
	//compare two mats
	cv::Mat diff;
	cv::compare(D, descriptor_from_disc, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff);
	cout<< "nz is " << nz;
	*/
	//
	//5) Create a bag of words trainer object
	// Use the BOWKMeansTrainer class
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	BOWKMeansTrainer bowTrainer(numCodewords, tc, 1, KMEANS_PP_CENTERS);
	//	6) Add the descriptors to the bag of words trainer object
	// Use the add method of the bag of words trainer object
	bowTrainer.add(descriptor_from_disc);
	//	7) Compute the codebook
	//	Use the cluster method of the bag of words trainer object
	//  Store the codebook to the input variable codeBook
	codeBook = bowTrainer.cluster();
	//8) Create a Brute Force descriptor matcher object
	Ptr<DescriptorMatcher> DisciptorMatcher = DescriptorMatcher::create("BruteForce");
	//9) Create a bag of words descriptor extractor object
	Ptr<BOWImgDescriptorExtractor> bowDE (new BOWImgDescriptorExtractor(DescriptorExtractor, DisciptorMatcher));
	//10) Set the codebook of the bag of words descriptor extractor object
	bowDE->setVocabulary(codeBook);
	//11)
	for (int i = 0; i < Dataset.trainingImages.size(); i++){
		for (int j = 0; j < Dataset.trainingImages[i].size(); j++){			
			//
			Mat image_sift;
			// KePoints 
			vector<KeyPoint> kp;
			//Detect SIFT key points in I
			FeatureDetector->detect(Dataset.trainingImages[i][j], kp);

			//Discard the keypoints outside of the annotation rectangle
			int size = kp.size();
			for (int k = 0; k < size; k++){
				if (Dataset.trainingAnnotations[i][j].x + Dataset.trainingAnnotations[i][j].width < kp[k].pt.x || kp[k].pt.x < Dataset.trainingAnnotations[i][j].x || kp[k].pt.y >Dataset.trainingAnnotations[i][j].y + Dataset.trainingAnnotations[i][j].height || kp[k].pt.y < Dataset.trainingAnnotations[i][j].y)
				{
					kp.erase(kp.begin()+k);
				}
				if (size != kp.size()){
					--k;
					size = kp.size();
				}
			}
			// To store the the bag of words histogram
			Mat bow_histogram;
			//Compute the bag of words histogram representation
			bowDE->compute2(Dataset.trainingImages[i][j], kp, bow_histogram);
			//Store the bag of words histogram of I in the corresponding indices of the input
			//variable imageDescriptors
			imageDescriptors[i][j] = bow_histogram;
		}
	}

}

/* Test BoW */
void Test(const Caltech101 &Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors)
{
	int total_number_of_test_images = 0;
	int number_of_correctly_assigned_test_images = 0;
	//create a sift feature detector object
	Ptr<FeatureDetector> FeatureDetector = FeatureDetector::create("SIFT");
	//Create a SIFT descriptor extractor object
	Ptr<DescriptorExtractor> DescriptorExtractor = DescriptorExtractor::create("SIFT");
	Ptr<DescriptorMatcher> DisciptorMatcher = DescriptorMatcher::create("BruteForce");
	Ptr<BOWImgDescriptorExtractor> bowDE(new BOWImgDescriptorExtractor(DescriptorExtractor, DisciptorMatcher));
	
	//2)Set the codebook of the bag of words descriptor extractor object
	bowDE->setVocabulary(codeBook);
	//3)
	for (int i = 0; i < Dataset.testImages.size(); i++){
		for (int j = 0; j < Dataset.testImages[i].size(); j++)
		{			
			++total_number_of_test_images;
			Mat image_sift;
			// KePoints 
			vector<KeyPoint> kp;
			//detect method
			FeatureDetector->detect(Dataset.testImages[i][j], kp);
			//Discard the keypoints outside of the annotation rectangle
			int size = kp.size();
			for (int k = 0; k < size; k++){
				if (Dataset.testAnnotations[i][j].x + Dataset.testAnnotations[i][j].width < kp[k].pt.x || kp[k].pt.x < Dataset.testAnnotations[i][j].x || kp[k].pt.y >Dataset.testAnnotations[i][j].y + Dataset.testAnnotations[i][j].height || kp[k].pt.y < Dataset.testAnnotations[i][j].y)
				{
					kp.erase(kp.begin() + k);
				}
				if (size != kp.size()){
					--k;
					size = kp.size();
				}
			}
			//Compute the bag of words histogram representation
			Mat bow_descriptor;
			bowDE->compute2(Dataset.testImages[i][j], kp, bow_descriptor);
			//Find the best matching histogram amongst the bag of words histogram representation of the training images
			double res = norm(bow_descriptor, imageDescriptors[0][0]);
			//String category_lable = Dataset.categoryNames[0];
			int machedCategoryNumber = -1;
			//
			for (int k = 0; k < imageDescriptors.size(); k++){
				for (int l = 0; l < imageDescriptors[k].size(); l++){
					if (norm(bow_descriptor, imageDescriptors[k][l]) <= res)
					{
						res = norm(bow_descriptor, imageDescriptors[k][l]);
						//Assign the category label of the test image to the category of the closest match
						//category_lable = Dataset.categoryNames[i];
						machedCategoryNumber = k;
					}
				}				
			}
			//check if the matched category is correct
			if (machedCategoryNumber == i){
				//correct match
				++number_of_correctly_assigned_test_images;
			}
		}
	}
	//4) Compare the estimated and the true category of the test images to obtain the recognition ratio
	double compareRatio = 0.0;
	compareRatio = (double)number_of_correctly_assigned_test_images / (double)total_number_of_test_images;
	//5) Print the recognition rate of the method on the console
	cout << "number_of_correctly_assigned_test_images: " << number_of_correctly_assigned_test_images << endl;
	cout << "total_number_of_test_images: " << total_number_of_test_images << endl;
	cout << "recognition rate is: " << compareRatio << endl;
}
