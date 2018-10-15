// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// OpenCV Haar-Cascade based Region-of-Interest Detector.

#include <string>

#include <iostream>  // DEBUG
#include <cstdio>
#include <cstdlib>

#include "opencv2/core/core.hpp"
#include "objdetect.hpp"
#include "imgproc.hpp"

#include "roi_detector_opencv.h"

namespace {
std::string GetHaarCascadeDir() {
  char* env_detectors = getenv("OPENCV_HAARCASCADES_PARENTDIR");
  if (env_detectors == nullptr) {
    return "./haarcascades/";
  }
  return std::string(env_detectors) + "/haarcascades/";
}
}  // namespace

HaarDetector::HaarDetector(const std::vector<std::string>& cascade_files) {
  const std::string& haarcascade_path_prefix = GetHaarCascadeDir();
  for (const auto& cascade_file : cascade_files) {
    std::string cascade_path = haarcascade_path_prefix + cascade_file;
    detectors_.push_back(std::make_unique<cv::CascadeClassifier>());
    std::cout << "Loading Haar Cascade: " << cascade_path << std::endl;
    detectors_.back()->load(cascade_path);
  }
}

HaarDetector::~HaarDetector() {}


std::vector<RegionOfInterest> HaarDetector::Detect(
    size_t width, size_t height, size_t row_bytes, int downscaling,
    void* data_32fc1) const {
  std::vector<RegionOfInterest> ret;
  cv::Mat image_gray_f0(height, width, CV_32FC1, data_32fc1, row_bytes);
  cv::Mat image_gray_f;
  cv::Mat image_gray_8uc1;
  cv::resize(image_gray_f0, image_gray_f, cv::Size(),
             1.0 / downscaling, 1.0 / downscaling);
  image_gray_f.convertTo(image_gray_8uc1, CV_8UC1, 255.0);
  cv::equalizeHist(image_gray_8uc1, image_gray_8uc1);

  for (int num_detector = 0; num_detector < detectors_.size(); ++num_detector) {
    std::vector<cv::Rect> rects;
    detectors_[num_detector]->detectMultiScale(image_gray_8uc1, rects);
    for (const auto& rect : rects) {
      ret.push_back({num_detector,
                     downscaling * rect.x,
                     downscaling * rect.y,
                     downscaling * rect.width,
                     downscaling * rect.height});
    }
  }
  return ret;
}
