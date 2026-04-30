#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

/*
 * YOLO TensorRT C++ Inference Template
 * -------------------------------------
 * المتطلبات (Requirements):
 * 1. NVIDIA TensorRT
 * 2. CUDA Toolkit
 * 3. OpenCV C++
 *
 * للترجمة (Compilation):
 * nvcc -o yolo_inference inference_tensorrt.cpp `pkg-config --cflags --libs opencv4` -lnvinfer -lcudart
 */

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// تحميل ملف الـ Engine من القرص
std::vector<char> loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error reading engine file: " << path << std::endl;
        return {};
    }
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();
    return trtModelStream;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./yolo_inference <engine_path> <image_path>" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    std::string imagePath = argv[2];

    // 1. قراءة الصورة باستخدام OpenCV
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to read image!" << std::endl;
        return -1;
    }

    // تجهيز الصورة (Resize to 640x640, BGR to RGB, Normalize)
    cv::Mat pr_img;
    cv::resize(img, pr_img, cv::Size(640, 640));
    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
    pr_img.convertTo(pr_img, CV_32FC3, 1.0f / 255.0f);
    
    // تحويل الأبعاد من HWC إلى CHW
    std::vector<float> input_data(1 * 3 * 640 * 640);
    std::vector<cv::Mat> chw(3);
    chw[0] = cv::Mat(640, 640, CV_32FC1, input_data.data());
    chw[1] = cv::Mat(640, 640, CV_32FC1, input_data.data() + 640 * 640);
    chw[2] = cv::Mat(640, 640, CV_32FC1, input_data.data() + 2 * 640 * 640);
    cv::split(pr_img, chw);

    // 2. تحميل الـ Engine
    std::cout << "Loading TensorRT Engine..." << std::endl;
    auto engineData = loadEngine(enginePath);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // 3. حجز الذاكرة في كرت الشاشة (GPU)
    void* buffers[2];
    cudaMalloc(&buffers[0], 1 * 3 * 640 * 640 * sizeof(float)); // Input
    // ملاحظة: الحجم يعتمد على الـ output للموديل، في YOLOv8 عادة يكون [1, 84, 8400]
    cudaMalloc(&buffers[1], 1 * 84 * 8400 * sizeof(float)); // Output

    // نسخ البيانات من المعالج إلى كرت الشاشة
    cudaMemcpy(buffers[0], input_data.data(), 1 * 3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice);

    // 4. تنفيذ الفحص (Inference)
    std::cout << "Running Inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // تنفيذ الـ Engine
    context->executeV2(buffers);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Inference Time: " << duration.count() << " ms" << std::endl;

    // تنظيف الذاكرة
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
