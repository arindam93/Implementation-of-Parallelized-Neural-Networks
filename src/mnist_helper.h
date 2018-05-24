#ifndef APT_PROJECT_MNIST_HELPER_H
#define APT_PROJECT_MNIST_HELPER_H

#include <string>
#include <vector>

std::vector<std::vector<double>> read_images(const std::string& filename);
std::vector<double> read_labels(const std::string& filename);

#endif //APT_PROJECT_MNIST_HELPER_H
