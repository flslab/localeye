#include <opencv2/opencv.hpp>
#include <iostream>
#include "EDCircles.h"
#include <vector>
#include <cmath>
#include <array>

// #include <iomanip>
// #include <memory>
// #include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "LibCamera.h"

// using namespace libcamera;
// using namespace std::chrono_literals;

// #include <Eigen/Dense>  // Using Eigen library for matrix/vector operations


using namespace cv;
using namespace std;


struct ImplicitEllipse {
    double A, B, C, D, E, F;
};


// Function to estimate the center of the sphere from the implicit ellipse parameters
std::pair<std::array<double, 3>, double> implEllipse2implSphereIni(double A, double B, double C, double D, double E,
                                                                   double F, double r) {
    double constMatrix[4][4] = {
        {0, 0, 0, -0.5},
        {1, -1, 0, -0.5},
        {1, 0, -1, -0.5},
        {1 / (r * r), -1 / (r * r), -1 / (r * r), -1 / (r * r)}
    };

    // Create PHI matrix
    double PHI[4] = {A, C, F, B * D / E};

    // Multiply constMatrix by PHI to get X
    double X[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            X[i] += constMatrix[i][j] * PHI[j];
        }
        X[i] = std::abs(X[i]); // Take absolute value as in the MATLAB code
    }

    // Calculate the sphere center (S0)
    std::array<double, 3> S0;
    double scaleFactor = std::sqrt(X[0] / X[3]); // Scale factor to be applied to S0 components
    S0[0] = scaleFactor * std::sqrt(X[0] / X[3]) * (D < 0 ? -1 : 1);
    S0[1] = scaleFactor * std::sqrt(X[1] / X[3]) * (E < 0 ? -1 : 1);
    S0[2] = scaleFactor * std::sqrt(X[2] / X[3]);

    // Return the sphere center S0 and alpha (which is X[3])
    return {S0, X[3]};
}


// Function to convert OpenCV ellipse to implicit form
ImplicitEllipse ellipseToImplicit(const mEllipse &ellipse) {
    // Parameters from cv::RotatedRect

    double a = ellipse.axes.width / 2.0; // Semi-major axis
    double b = ellipse.axes.height / 2.0; // Semi-minor axis
    double x0 = ellipse.center.x; // Center x-coordinate
    double y0 = ellipse.center.y; // Center y-coordinate
    double theta = ellipse.theta; // * CV_PI / 180.0;  // Rotation angle in radians

    // Precompute trigonometric terms
    double cosTheta = std::cos(theta);
    double sinTheta = std::sin(theta);

    // Compute coefficients for implicit ellipse equation:
    // A*u^2 + B*u*v + C*v^2 + D*u + E*v + F = 0
    double A = (cosTheta * cosTheta) / (a * a) + (sinTheta * sinTheta) / (b * b);
    double B = 2 * cosTheta * sinTheta * (1 / (a * a) - 1 / (b * b));
    double C = (sinTheta * sinTheta) / (a * a) + (cosTheta * cosTheta) / (b * b);
    double D = -2 * A * x0 - B * y0;
    double E = -2 * C * y0 - B * x0;
    double F = A * x0 * x0 + B * x0 * y0 + C * y0 * y0 - 1;

    // Return implicit ellipse coefficients
    return {A, B, C, D, E, F};
}


// Function to normalize points using the camera matrix
vector<Point3d> normalizePoints(const vector<Point2d> &points, const Mat &K) {
    vector<Point3d> normalizedPoints;
    Mat K_inv = K.inv();

    for (const auto &pt: points) {
        Mat normalized = K_inv * (Mat_<double>(3, 1) << pt.x, pt.y, 1);
        normalizedPoints.emplace_back(normalized.at<double>(0), normalized.at<double>(1), normalized.at<double>(2));
    }

    return normalizedPoints;
}


// // Main Ell2Sphere algorithm implementation
// Vector3d Ell2Sphere(const Vector2d& e0, double a, double b, double theta, double r) {
//     Vector2d a1 = e0 + Vector2d { a * cos(theta), a * sin(theta) };
//     Vector2d a2 = e0 + Vector2d { -a * cos(theta), -a * sin(theta) };
//
//     // Step 2: Calculate cone's opening angle and direction
//     auto [cos2alpha, w] = ConeFromEllipse(a1, a2);
//
//     // Step 3: Calculate the sphere center
//     return SphereCenterFromEllipse(r, cos2alpha, w);
// }


// Function to calculate the norm of a vector
double norm(const std::vector<double> &v) {
    double sum = 0;
    for (double val: v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// Function to perform matrix-vector multiplication
std::vector<double> matVecMult(const std::vector<std::vector<double> > &matrix, const std::vector<double> &vec) {
    std::vector<double> result(3, 0.0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

// Main function to convert parametric ellipse to implicit sphere
std::array<double, 3> paramEllipse2implSphere(double a, double b, double ex, double ey, double theta, double r) {
    // Transformation matrix
    std::vector<std::vector<double> > T = {
        {a * std::cos(theta), -b * std::sin(theta), ex},
        {a * std::sin(theta), b * std::cos(theta), ey},
        {0, 0, 1}
    };

    // Get the unit vectors directed to the major ellipse axis ending points
    std::vector<double> a1 = matVecMult(T, {-1, 0, 1});
    std::vector<double> a2 = matVecMult(T, {1, 0, 1});

    // Normalize a1 and a2
    std::vector<double> q_a1 = {a1[0] / norm(a1), a1[1] / norm(a1), a1[2] / norm(a1)};
    std::vector<double> q_a2 = {a2[0] / norm(a2), a2[1] / norm(a2), a2[2] / norm(a2)};

    // Angle bisector
    std::vector<double> w = {(q_a2[0] + q_a1[0]) / 2, (q_a2[1] + q_a1[1]) / 2, (q_a2[2] + q_a1[2]) / 2};
    double w_norm = norm(w);
    w = {w[0] / w_norm, w[1] / w_norm, w[2] / w_norm};

    // Cone angle and direction
    double cos2alpha = q_a1[0] * q_a2[0] + q_a1[1] * q_a2[1] + q_a1[2] * q_a2[2];
    double d = std::sqrt(2) * r / std::sqrt(1 - cos2alpha);

    std::vector<double> S = {d * w[0], d * w[1], d * w[2]};
    return {S[0], S[1], S[2]};
}


// std::array<double, 3> paramEllipse2implSphereEigen(double a, double b, double ex, double ey, double theta, double r) {
//     // Define the transformation matrix T
//     Eigen::Matrix3d T;
//     T << a * std::cos(theta), -b * std::sin(theta), ex,
//          a * std::sin(theta),  b * std::cos(theta), ey,
//          0,                   0,                   1;
//
//     // Get the unit vectors directed to the major ellipse axis ending points
//     Eigen::Vector3d a1 = T * Eigen::Vector3d(-1, 0, 1);
//     Eigen::Vector3d a2 = T * Eigen::Vector3d(1, 0, 1);
//
//     // Normalize vectors a1 and a2
//     Eigen::Vector3d q_a1 = a1.normalized();
//     Eigen::Vector3d q_a2 = a2.normalized();
//
//     // Angle bisector
//     Eigen::Vector3d w = (q_a2 + q_a1).normalized();
//
//     // Cone angle and direction
//     double cos2alpha = q_a1.dot(q_a2);
//     double d = std::sqrt(2) * r / std::sqrt(1 - cos2alpha);
//
//     Eigen::Vector3d S = d * w;
//     return {S.x(), S.y(), S.z()};
// }

static std::shared_ptr<Camera> camera;

static void requestComplete(Request *request)
{
    if (request->status() == Request::RequestCancelled)
        return;

    const std::map<const Stream *, FrameBuffer *> &buffers = request->buffers();

    for (auto bufferPair : buffers) {
        FrameBuffer *buffer = bufferPair.second;
        const FrameMetadata &metadata = buffer->metadata();
    }
}

int main(int argc, char **argv) {
    // if (argc != 2) {
    //     std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
    //     return -1;
    // }

    time_t start_time = time(0);
    int frame_count = 0;
    float lens_position = 100;
    float focus_step = 50;
    LibCamera cam;
    uint32_t width = 1280;
    uint32_t height = 720;
    uint32_t stride;
    char key;
    int window_width = 1280;
    int window_height = 720;

    if (width > window_width)
    {
        cv::namedWindow("libcamera-demo", cv::WINDOW_NORMAL);
        cv::resizeWindow("libcamera-demo", window_width, window_height);
    }

    int ret = cam.initCamera();
    cam.configureStill(width, height, formats::RGB888, 1, 0);
    ControlList controls_;
    int64_t frame_time = 1000000 / 10;
    // Set frame rate
	controls_.set(controls::FrameDurationLimits, libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));
    // Adjust the brightness of the output images, in the range -1.0 to 1.0
    // controls_.set(controls::Brightness, 0.5);
    // Adjust the contrast of the output image, where 1.0 = normal contrast
    // controls_.set(controls::Contrast, 1.5);
    // Set the exposure time
    // controls_.set(controls::ExposureTime, 20000);
    cam.set(controls_);
    if (!ret) {
        bool flag;
        LibcameraOutData frameData;
        cam.startCamera();
        cam.VideoStream(&width, &height, &stride);
        while (true) {
            flag = cam.readFrame(&frameData);
            if (!flag)
                continue;
            // CV_8UC3 for color CV_8UC1 for grascale image
            Mat im(height, width, CV_8UC3, frameData.imageData, stride);
            Mat frame(height, width, CV_8UC1, frameData.imageData, stride);

            // cv::circle(im, cv::Point(200, 200), static_cast<int>(40), cv::Scalar(0, 255, 0), 2); // Green circle

            // Detect ellipses
            // Initialize EDLib Circle and Ellipse detector
            EDCircles circleDetector(frame);

            // Detect circles and ellipses
            std::vector<mCircle> circles = circleDetector.getCircles();
            std::vector<mEllipse> ellipses = circleDetector.getEllipses();

            // Draw and print detected circles
            for (size_t i = 0; i < circles.size(); ++i) {
                const auto &circle = circles[i];
                cv::circle(im, circle.center, static_cast<int>(circle.r), cv::Scalar(0, 255, 0), 2); // Green circle

                // Print circle parameters
                std::cout << "Circle " << i + 1 << ": Center = (" << circle.center.x << ", " << circle.center.y
                        << "), Radius = " << circle.r << std::endl;
            }

            // Draw and print detected ellipses
            for (size_t i = 0; i < ellipses.size(); ++i) {
                const auto &ellipse = ellipses[i];
                cv::ellipse(im, ellipse.center, ellipse.axes, ellipse.theta * 180.0 / CV_PI, 0, 360,
                            cv::Scalar(255, 0, 0), 2); // Blue ellipse

                // Print ellipse parameters
                cout << "Ellipse " << i + 1 << ": Center = (" << ellipse.center.x << ", " << ellipse.center.y
                        << "), Axes = (" << ellipse.axes.width << ", " << ellipse.axes.height
                        << "), Orientation = " << ellipse.theta << " radians" << endl;

                double r = 24;

                array<double, 3> sphere_center = paramEllipse2implSphere(ellipse.axes.width, ellipse.axes.height,
                                                                         ellipse.center.x, ellipse.center.y, ellipse.theta, r);
                cout << "Sphere center: [" << sphere_center[0] << ", "
                        << sphere_center[1] << ", " << sphere_center[2] << "]" << endl;
                // Convert the ellipse to implicit form
                // ImplicitEllipse implicit = ellipseToImplicit(ellipse);

                // std::cout << "Implicit Ellipse: " << implicit.A << ", " << implicit.B << ", " << implicit.C << ", " << implicit.D << ", " << implicit.E << ", " << implicit.F << std::endl;
            }


            imshow("libcamera-demo", im);
            key = waitKey(1);
            if (key == 'q') {
                break;
            } else if (key == 'f') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeAuto);
                controls.set(controls::AfTrigger, 0);
                cam.set(controls);
            } else if (key == 'a' || key == 'A') {
                lens_position += focus_step;
            } else if (key == 'd' || key == 'D') {
                lens_position -= focus_step;
            }

            // To use the manual focus function, libcamera-dev needs to be updated to version 0.0.10 and above.
            if (key == 'a' || key == 'A' || key == 'd' || key == 'D') {
                ControlList controls;
                controls.set(controls::AfMode, controls::AfModeManual);
				controls.set(controls::LensPosition, lens_position);
                cam.set(controls);
            }

            frame_count++;
            if ((time(0) - start_time) >= 1){
                printf("fps: %d\n", frame_count);
                frame_count = 0;
                start_time = time(0);
            }
            cam.returnFrameBuffer(frameData);
        }
        destroyAllWindows();
        cam.stopCamera();
    }
    cam.closeCamera();
    return 0;

    // std::unique_ptr<CameraManager> cm = std::make_unique<CameraManager>();
    // cm->start();
    //
    // auto cameras = cm->cameras();
    // if (cameras.empty()) {
    //     std::cout << "No cameras were identified on the system."
    //               << std::endl;
    //     cm->stop();
    //     return EXIT_FAILURE;
    // }
    //
    // std::string cameraId = cameras[0]->id();
    //
    // auto camera = cm->get(cameraId);
    // camera->acquire();
    //
    // std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration( { StreamRole::Viewfinder } );
    //
    // StreamConfiguration &streamConfig = config->at(0);
    // std::cout << "Default viewfinder configuration is: " << streamConfig.toString() << std::endl;
    //
    // streamConfig.size.width = 640;
    // streamConfig.size.height = 480;
    //
    // config->validate();
    // std::cout << "Validated viewfinder configuration is: " << streamConfig.toString() << std::endl;
    // camera->configure(config.get());
    //
    // FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
    //
    // for (StreamConfiguration &cfg : *config) {
    //     int ret = allocator->allocate(cfg.stream());
    //     if (ret < 0) {
    //         std::cerr << "Can't allocate buffers" << std::endl;
    //         return -ENOMEM;
    //     }
    //
    //     size_t allocated = allocator->buffers(cfg.stream()).size();
    //     std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
    // }
    //
    // Stream *stream = streamConfig.stream();
    // const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
    // std::vector<std::unique_ptr<Request>> requests;
    //
    // for (unsigned int i = 0; i < buffers.size(); ++i) {
    //     std::unique_ptr<Request> request = camera->createRequest();
    //     if (!request)
    //     {
    //         std::cerr << "Can't create request" << std::endl;
    //         return -ENOMEM;
    //     }
    //
    //     const std::unique_ptr<FrameBuffer> &buffer = buffers[i];
    //     int ret = request->addBuffer(stream, buffer.get());
    //     if (ret < 0)
    //     {
    //         std::cerr << "Can't set buffer for request"
    //               << std::endl;
    //         return ret;
    //     }
    //
    //     requests.push_back(std::move(request));
    // }
    // camera->requestCompleted.connect(requestComplete);

    // while (true) {
    //
    //     // Get frame data and process it
    //     // Mat frame(Size(1280, 720), CV_8UC1, buffer->planes()[0].mem);
    //
    //     // Initialize EDLib Circle and Ellipse detector
    //     EDCircles circleDetector(frame);
    //
    //     // Detect circles and ellipses
    //     std::vector<mCircle> circles = circleDetector.getCircles();
    //     std::vector<mEllipse> ellipses = circleDetector.getEllipses();
    //
    //     // Draw and print detected circles
    //     for (size_t i = 0; i < circles.size(); ++i) {
    //         const auto &circle = circles[i];
    //         cv::circle(frame, circle.center, static_cast<int>(circle.r), cv::Scalar(0, 255, 0), 2); // Green circle
    //
    //         // Print circle parameters
    //         std::cout << "Circle " << i + 1 << ": Center = (" << circle.center.x << ", " << circle.center.y
    //                 << "), Radius = " << circle.r << std::endl;
    //     }
    //
    //     // Draw and print detected ellipses
    //     for (size_t i = 0; i < ellipses.size(); ++i) {
    //         const auto &ellipse = ellipses[i];
    //         cv::ellipse(frame, ellipse.center, ellipse.axes, ellipse.theta * 180.0 / CV_PI, 0, 360,
    //                     cv::Scalar(255, 0, 0), 2); // Blue ellipse
    //
    //         // Print ellipse parameters
    //         cout << "Ellipse " << i + 1 << ": Center = (" << ellipse.center.x << ", " << ellipse.center.y
    //                 << "), Axes = (" << ellipse.axes.width << ", " << ellipse.axes.height
    //                 << "), Orientation = " << ellipse.theta << " radians" << endl;
    //
    //         double r = 24;
    //
    //         array<double, 3> sphere_center = paramEllipse2implSphere(ellipse.axes.width, ellipse.axes.height,
    //                                                                  ellipse.center.x, ellipse.center.y, ellipse.theta, r);
    //         cout << "Sphere center: [" << sphere_center[0] << ", "
    //                 << sphere_center[1] << ", " << sphere_center[2] << "]" << endl;
    //         // Convert the ellipse to implicit form
    //         // ImplicitEllipse implicit = ellipseToImplicit(ellipse);
    //
    //         // std::cout << "Implicit Ellipse: " << implicit.A << ", " << implicit.B << ", " << implicit.C << ", " << implicit.D << ", " << implicit.E << ", " << implicit.F << std::endl;
    //     }
    //
    //     // Display result
    //     // cv::imshow("Detected Circles and Ellipses", displayImage);
    //     // cv::waitKey(0); // Wait for a key press before closing
    //
    //     // Optionally save the output image
    //     // cv::imwrite("output.png", displayImage);
    //
    //     // Define a Region of Interest (ROI) where the LED is located
    //     Rect roi(100, 100, 100, 100); // Adjust as needed
    //     Mat roi_gray = frame(roi);
    //
    //     // Calculate the average brightness in the ROI
    //     Scalar mean_brightness = mean(roi_gray);
    //     double brightness = mean_brightness[0];
    //
    //     // Detect whether the LED is ON or OFF based on brightness
    //     if (brightness > threshold_value) {
    //         cout << "Frame " << frame_counter << ": LED ON - binary 1" << endl;
    //     } else {
    //         cout << "Frame " << frame_counter << ": LED OFF - binary 0" << endl;
    //     }
    //
    //     // Show the frame with the ROI for debugging (optional)
    //     rectangle(frame, roi, Scalar(0, 255, 0), 2);
    //     imshow("LED Detection", frame);
    //
    //     // Exit the loop if a key is pressed
    //     if (waitKey(30) >= 0) break;
    //
    //     frame_counter++;
    // }

    // // Stop camera
    // camera->stop();
    // allocator->free(stream);
    // delete allocator;
    // camera->release();
    // camera.reset();
    // cm->stop();

    return 0;

    // Load input image
    // cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    // if (image.empty()) {
    //     std::cerr << "Could not load image: " << argv[1] << std::endl;
    //     return -1;
    // }

    // Convert grayscale to BGR for displaying color circles and ellipses
    // cv::Mat displayImage;
    // cv::cvtColor(image, displayImage, cv::COLOR_GRAY2BGR);



}
