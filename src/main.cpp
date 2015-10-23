#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <cassert>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"


#define DEBUG_FLAG              1   // Debug flag for image channels
#define MICROGLIAL_ROI_FACTOR   20  // ROI of microglial cell = roi factor * mean microglial dia
#define NUM_AREA_BINS           21  // Number of bins
#define BIN_AREA                25  // Bin area
#define NUM_Z_LAYERS_COMBINED   3   // Number of z-layers combined


/* Channel type */
enum class ChannelType : unsigned char {
    BLUE = 0,
    GREEN
};

/* Hierarchy type */
enum class HierarchyType : unsigned char {
    INVALID_CNTR = 0,
    CHILD_CNTR,
    PARENT_CNTR
};

/* Thin sub-iteration 1 */
void thinSubiteration1 (cv::Mat &src, cv::Mat &dst) {

    int rows = src.rows;
    int cols = src.cols;
    src.copyTo(dst);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            if (src.at<float>(i, j) != 1.0f) continue;

            // get 8 neighbors, calculate C(p)
            int neighbor0 = (int) src.at<float>(i-1, j-1);
            int neighbor1 = (int) src.at<float>(i-1, j  );
            int neighbor2 = (int) src.at<float>(i-1, j+1);
            int neighbor3 = (int) src.at<float>(i  , j+1);
            int neighbor4 = (int) src.at<float>(i+1, j+1);
            int neighbor5 = (int) src.at<float>(i+1, j  );
            int neighbor6 = (int) src.at<float>(i+1, j-1);
            int neighbor7 = (int) src.at<float>(i  , j-1);

            int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                    int(~neighbor3 & ( neighbor4 | neighbor5)) +
                    int(~neighbor5 & ( neighbor6 | neighbor7)) +
                    int(~neighbor7 & ( neighbor0 | neighbor1));

            if(C != 1) continue;

            // calculate N
            int N1 =    int(neighbor0 | neighbor1) +
                        int(neighbor2 | neighbor3) +
                        int(neighbor4 | neighbor5) +
                        int(neighbor6 | neighbor7);

            int N2 =    int(neighbor1 | neighbor2) +
                        int(neighbor3 | neighbor4) +
                        int(neighbor5 | neighbor6) +
                        int(neighbor7 | neighbor0);

            int N = std::min(N1,N2);
            if ((N == 2) || (N == 3)) {
                // calculate criteria 3
                int c3 = ( neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
                if (c3 == 0) {
                    dst.at<float>(i, j) = 0.0f;
                }
            }
        }
    }
}

/* Thin sub-iteration 2 */
void thinSubiteration2 (cv::Mat &src, cv::Mat &dst) {

    int rows = src.rows;
    int cols = src.cols;
    src.copyTo(dst);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            if (src.at<float>( i, j) != 1.0f) continue;

            // get 8 neighbors, calculate C(p)
            int neighbor0 = (int) src.at<float>(i-1, j-1);
            int neighbor1 = (int) src.at<float>(i-1, j  );
            int neighbor2 = (int) src.at<float>(i-1, j+1);
            int neighbor3 = (int) src.at<float>(i  , j+1);
            int neighbor4 = (int) src.at<float>(i+1, j+1);
            int neighbor5 = (int) src.at<float>(i+1, j  );
            int neighbor6 = (int) src.at<float>(i+1, j-1);
            int neighbor7 = (int) src.at<float>(i  , j-1);

            int C = int(~neighbor1 & ( neighbor2 | neighbor3)) +
                    int(~neighbor3 & ( neighbor4 | neighbor5)) +
                    int(~neighbor5 & ( neighbor6 | neighbor7)) +
                    int(~neighbor7 & ( neighbor0 | neighbor1));
            
            if(C != 1) continue;

            // calculate N
            int N1 =    int(neighbor0 | neighbor1) +
                        int(neighbor2 | neighbor3) +
                        int(neighbor4 | neighbor5) +
                        int(neighbor6 | neighbor7);

            int N2 =    int(neighbor1 | neighbor2) +
                        int(neighbor3 | neighbor4) +
                        int(neighbor5 | neighbor6) +
                        int(neighbor7 | neighbor0);

            int N = std::min(N1,N2);
            if((N == 2) || (N == 3)) {
                // calculate E
                int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
                if (E == 0) {
                    dst.at<float>(i, j) = 0.0f;
                }
            }
        }
    }
}

/* Skeletonize */
void skeletonize(cv::Mat in, cv::Mat *out) {

    int rows = in.rows;
    int cols = in.cols;
    in.convertTo(in, CV_32FC1);
    in.copyTo(*out);
    out->convertTo(*out, CV_32FC1);

    // pad source
    cv::Mat p_enlarged_src = cv::Mat(rows+2, cols+2, CV_32FC1);
    for (int i = 0; i < rows+2; i++) {
        p_enlarged_src.at<float>(i, 0) = 0.0f;
        p_enlarged_src.at<float>(i, cols+1) = 0.0f;
    }
    for (int j = 0; j < cols+2; j++) {
        p_enlarged_src.at<float>(0, j) = 0.0f;
        p_enlarged_src.at<float>(rows+1, j) = 0.0f;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (in.at<float>(i, j) >= 20.0f) {
                p_enlarged_src.at<float>(i+1, j+1) = 1.0f;
            } else {
                p_enlarged_src.at<float>(i+1, j+1) = 0.0f;
            }
        }
    }

    // start to thin
    cv::Mat p_thin_mat1 = cv::Mat::zeros(rows+2, cols+2, CV_32FC1);
    cv::Mat p_thin_mat2 = cv::Mat::zeros(rows+2, cols+2, CV_32FC1);
    cv::Mat p_cmp       = cv::Mat::zeros(rows+2, cols+2, CV_8UC1);

    int num_non_zero = (rows+2)*(cols+2);                       // initialize
    do {
        thinSubiteration1(p_enlarged_src, p_thin_mat1);         // sub-iteration 1
        thinSubiteration2(p_thin_mat1, p_thin_mat2);            // sub-iteration 2
        compare(p_enlarged_src, p_thin_mat2, p_cmp, CV_CMP_EQ); // compare
        num_non_zero = countNonZero(p_cmp);                     // check
        p_thin_mat2.copyTo(p_enlarged_src);                     // copy

    } while (num_non_zero != (rows+2)*(cols+2));

    for (int i = 0; i < rows; i++) { // copy result
        for (int j = 0; j < cols; j++) {
            out->at<float>(i, j) = p_enlarged_src.at<float>(i+1, j+1);
        }
    }
}

/* Enhance the image */
bool enhanceImage(cv::Mat src, ChannelType channel_type, cv::Mat *dst) {

    // Enhance the image using Gaussian blur and thresholding
    cv::Mat enhanced;

    switch(channel_type) {
        case ChannelType::BLUE: {
            // Enhance the blue channel
            // Sharpen the channel
            cv::Mat gray;
            cv::GaussianBlur(src, gray, cv::Size(3,3), 11);
            cv::addWeighted(src, 1.5, gray, -0.5, 0, enhanced);
            cv::threshold(enhanced, enhanced, 100, 255, cv::THRESH_BINARY);
        } break;

        case ChannelType::GREEN: {
            // Enhance the green channel
            // Sharpen the channel
            cv::Mat denoised, gray;
            cv::threshold(src, denoised, 20, 255, cv::THRESH_TOZERO);
            cv::GaussianBlur(denoised, gray, cv::Size(3,3), 11);
            cv::addWeighted(denoised, 1.5, gray, -0.5, 0, enhanced);
            cv::threshold(enhanced, enhanced, 20, 255, cv::THRESH_BINARY);
        } break;

        default: {
            std::cerr << "Invalid channel type" << std::endl;
            return false;
        }
    }
    *dst = enhanced;
    return true;
}

/* Find the contours in the image */
void contourCalc(cv::Mat src, ChannelType channel_type, 
                    double min_area, cv::Mat *dst, 
                    std::vector<std::vector<cv::Point>> *contours, 
                    std::vector<cv::Vec4i> *hierarchy, 
                    std::vector<HierarchyType> *validity_mask, 
                    std::vector<double> *parent_area) {

    cv::Mat temp_src;
    src.copyTo(temp_src);
    switch(channel_type) {
        case ChannelType::BLUE :
        case ChannelType::GREEN : {
            findContours(temp_src, *contours, *hierarchy, cv::RETR_EXTERNAL, 
                                                        cv::CHAIN_APPROX_SIMPLE);
        } break;

        default: return;
    }

    *dst = cv::Mat::zeros(temp_src.size(), CV_8UC3);
    if (!contours->size()) return;
    validity_mask->assign(contours->size(), HierarchyType::INVALID_CNTR);
    parent_area->assign(contours->size(), 0.0);

    // Keep the contours whose size is >= than min_area
    cv::RNG rng(12345);
    for (int index = 0 ; index < (int)contours->size(); index++) {
        if ((*hierarchy)[index][3] > -1) continue; // ignore child
        auto cntr_external = (*contours)[index];
        double area_external = fabs(contourArea(cv::Mat(cntr_external)));
        if (area_external < min_area) continue;

        std::vector<int> cntr_list;
        cntr_list.push_back(index);

        int index_hole = (*hierarchy)[index][2];
        double area_hole = 0.0;
        while (index_hole > -1) {
            std::vector<cv::Point> cntr_hole = (*contours)[index_hole];
            double temp_area_hole = fabs(contourArea(cv::Mat(cntr_hole)));
            if (temp_area_hole) {
                cntr_list.push_back(index_hole);
                area_hole += temp_area_hole;
            }
            index_hole = (*hierarchy)[index_hole][0];
        }
        double area_contour = area_external - area_hole;
        if (area_contour >= min_area) {
            (*validity_mask)[cntr_list[0]] = HierarchyType::PARENT_CNTR;
            (*parent_area)[cntr_list[0]] = area_contour;
            for (unsigned int i = 1; i < cntr_list.size(); i++) {
                (*validity_mask)[cntr_list[i]] = HierarchyType::CHILD_CNTR;
            }
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0,255), 
                                            rng.uniform(0,255));
            drawContours(*dst, *contours, index, color, CV_FILLED, 8, *hierarchy);
        }
    }
}

/* Process the images inside each directory */
bool processDir(std::string path, std::string image_name, std::string metrics_file) {

    /* Create the data output file for images that were processed */
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::app);
    if (!data_stream.is_open()) {
        std::cerr << "Could not open the data output file." << std::endl;
        return false;
    }

    // Create the output directory
    std::string out_directory = path + "result/";
    struct stat st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }
    out_directory = out_directory + image_name + "/";
    st = {0};
    if (stat(out_directory.c_str(), &st) == -1) {
        mkdir(out_directory.c_str(), 0700);
    }

    // Count the number of images
    std::string dir_name = path + "jpg/" + image_name + "/";
    DIR *read_dir = opendir(dir_name.c_str());
    if (!read_dir) {
        std::cerr << "Could not open directory '" << dir_name << "'" << std::endl;
        return false;
    }
    struct dirent *dir = NULL;
    uint8_t z_count = 0;
    bool collect_name_pattern = false;
    std::string end_pattern;
    while ((dir = readdir(read_dir))) {
        if (!strcmp (dir->d_name, ".") || !strcmp (dir->d_name, "..")) continue;
        if (!collect_name_pattern) {
            std::string delimiter = "c1+";
            end_pattern = dir->d_name;
            size_t pos = end_pattern.find(delimiter);
            end_pattern.erase(0, pos);
            collect_name_pattern = true;
        }
        z_count++;
    }

    std::vector<cv::Mat> blue_list(NUM_Z_LAYERS_COMBINED), green_list(NUM_Z_LAYERS_COMBINED);
    for (uint8_t z_index = 1; z_index <= z_count; z_index++) {

        // Create the input filename and rgb stream output filenames
        std::string in_filename;
        if (z_count < 10) {
            in_filename = dir_name + image_name + 
                                        "_z" + std::to_string(z_index) + end_pattern;
        } else {
            if (z_index < 10) {
                in_filename = dir_name + image_name + 
                                        "_z0" + std::to_string(z_index) + end_pattern;
            } else if (z_index < 100) {
                in_filename = dir_name + image_name + 
                                        "_z" + std::to_string(z_index) + end_pattern;
            } else { // assuming number of z plane layers will never exceed 99
                std::cerr << "Does not support more than 99 z layers curently" << std::endl;
                return false;
            }
        }

        // Extract the bgr streams for each input image
        cv::Mat img = cv::imread(in_filename.c_str(), -1);
        if (img.empty()) return false;

        // Original image
        std::string out_original = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_a_original.jpg";
        cv::imwrite(out_original.c_str(), img);

        std::vector<cv::Mat> channel(3);
        cv::split(img, channel);
        blue_list[(z_index-1)%NUM_Z_LAYERS_COMBINED]  = channel[0];
        green_list[(z_index-1)%NUM_Z_LAYERS_COMBINED] = channel[1];

        // Continue collecting layers if needed
        if (z_index%NUM_Z_LAYERS_COMBINED && (z_index != z_count)) continue;

        // Merge some layers together
        cv::Mat blue  = cv::Mat::zeros(channel[0].size(), CV_8UC1);
        cv::Mat green = cv::Mat::zeros(channel[1].size(), CV_8UC1);
        for (unsigned int merge_index = 0; 
                    merge_index < NUM_Z_LAYERS_COMBINED; merge_index++) {
            bitwise_or(blue, blue_list[merge_index], blue);
            bitwise_or(green, green_list[merge_index], green);
        }

        /** Gather BGR channel information needed for feature extraction **/

        // Blue channel
        std::string out_blue = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_blue.jpg";
        if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue);

        cv::Mat blue_enhanced;
        if(!enhanceImage(blue, ChannelType::BLUE, &blue_enhanced)) return false;
        out_blue.insert(out_blue.find_last_of("."), "_enhanced", 9);
        if (DEBUG_FLAG) cv::imwrite(out_blue.c_str(), blue_enhanced);

        cv::Mat blue_segmented;
        std::vector<std::vector<cv::Point>> contours_blue;
        std::vector<cv::Vec4i> hierarchy_blue;
        std::vector<HierarchyType> blue_contour_mask;
        std::vector<double> blue_contour_area;
        contourCalc(blue_enhanced, ChannelType::BLUE, 1.0, &blue_segmented, 
                &contours_blue, &hierarchy_blue, &blue_contour_mask, &blue_contour_area);

        // Green channel
        std::string out_green = out_directory + "zlayer_" + 
                                        std::to_string(z_index) + "_green.jpg";
        if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green);

        cv::Mat green_enhanced;
        if(!enhanceImage(green, ChannelType::GREEN, &green_enhanced)) return false;
        out_green.insert(out_green.find_last_of("."), "_enhanced", 9);
        if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_enhanced);

        cv::Mat not_blue, green_minus_blue, green_skeletonize;
        bitwise_not(blue_enhanced, not_blue);
        bitwise_and(green_enhanced, not_blue, green_minus_blue);
        skeletonize(green_enhanced, &green_skeletonize);
        out_green.insert(out_green.find_last_of("."), "_skeletonized", 13);
        if (DEBUG_FLAG) cv::imwrite(out_green.c_str(), green_skeletonize);

        cv::Mat green_segmented;
        std::vector<std::vector<cv::Point>> contours_green;
        std::vector<cv::Vec4i> hierarchy_green;
        std::vector<HierarchyType> green_contour_mask;
        std::vector<double> green_contour_area;
        contourCalc(green_enhanced, ChannelType::GREEN, 1.0, &green_segmented, 
                &contours_green, &hierarchy_green, &green_contour_mask, &green_contour_area);
    }
    closedir(read_dir);
    data_stream.close();

    return true;
}

/* Main - create the threads and start the processing */
int main(int argc, char *argv[]) {

    /* Check for argument count */
    if (argc != 2) {
        std::cerr << "Invalid number of arguments." << std::endl;
        return -1;
    }

    /* Read the path to the data */
    std::string path(argv[1]);

    /* Read the list of directories to process */
    std::string image_list_filename = path + "image_list.dat";
    std::vector<std::string> input_images;
    FILE *file = fopen(image_list_filename.c_str(), "r");
    if (!file) {
        std::cerr << "Could not open 'image_list.dat' inside '" << path << "'." << std::endl;
        return -1;
    }
    char line[128];
    while (fgets(line, sizeof(line), file) != NULL) {
        line[strlen(line)-1] = 0;
        std::string temp_str(line);
        input_images.push_back(temp_str);
    }
    fclose(file);

    /* Create and prepare the file for metrics */
    std::string metrics_file = path + "computed_metrics.csv";
    std::ofstream data_stream;
    data_stream.open(metrics_file, std::ios::out);
    if (!data_stream.is_open()) {
        std::cerr << "Could not create the metrics file." << std::endl;
        return -1;
    }
    data_stream << std::endl;
    data_stream.close();

    /* Process each image */
    for (unsigned int index = 0; index < input_images.size(); index++) {
        std::cout << "Processing " << input_images[index] << std::endl;
        if (!processDir(path, input_images[index], metrics_file)) {
            std::cout << "ERROR !!!" << std::endl;
            return -1;
        }
    }

    return 0;
}

