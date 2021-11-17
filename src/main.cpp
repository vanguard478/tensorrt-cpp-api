#include "engine.h"
#include <opencv2/opencv.hpp>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


int main() {
    Options options;
    options.optBatchSizes = {2, 4, 8};

    Engine engine(options);

    // TODO: Specify your model here.
    // Must specify a dynamic batch size when exporting the model from onnx.
    const std::string onnxModelpath = "../model.dynamic_batch.onnx";

    bool succ = engine.build(onnxModelpath);
    if (!succ) {
        throw std::runtime_error("Unable to build TRT engine.");
    }

    succ = engine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    std::vector<size_t> batchSizes = {1, 2, 3, 4, 6, 8};
    std::vector<std::vector<cv::Mat>> images;


    const std::string inputImage = "../img.jpg";
    auto img = cv::imread(inputImage);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    for (const auto& batch: batchSizes) {
        std::vector<cv::Mat> imgVec;
        for (size_t i = 0; i < batch; ++i) {
            imgVec.push_back(img);
        }
        images.emplace_back(std::move(imgVec));
    }

    // Discard the first inference time as it takes longer
    std::vector<std::vector<float>> featureVectors;
    succ = engine.runInference(images[0], featureVectors);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    size_t numIterations = 100;

    for (size_t i = 0; i < numIterations; ++i) {
        auto testIdx = i % images.size();
        auto t1 = Clock::now();
        featureVectors.clear();
        engine.runInference(images[testIdx], featureVectors);
        auto t2 = Clock::now();
        double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "Average time per inference: " << totalTime / static_cast<float>(images[testIdx].size()) <<
        " ms, for batch size of: " << images[testIdx].size() << std::endl;
    }


    return 0;
}
