#ifndef APT_PROJECT_CONVOLUTIONAL_NEURAL_NETWORK_H
#define APT_PROJECT_CONVOLUTIONAL_NEURAL_NETWORK_H
#include <cstddef>
#include <vector>

typedef std::vector<double> vd1;
typedef std::vector<std::vector<double>> vd2;
typedef std::vector<std::vector<std::vector<double>>> vd3;
typedef std::vector<std::vector<std::vector<std::vector<double>>>> vd4;

// We always use [depth][x][y] for 3D images/filters and [x][y] for 2D
// Convolution filters are 3D, input images are 3D, output image is 2D

class convolutional_neural_network
{
public:
    int mImageWidth; // Input image width and height (square only, 28 for MNIST)
    int mImageDepth; // Input image depth (1 for MNIST)

    int mConvFilter1Width;
    int mConvFilter1Depth;
    int mConvFilter2Width;
    int mConvFilter2Depth;

    int mNumFullyConnected;
    int mNumOutputs;

    vd4 mConvFilters1; // dim: numFilters1 x imageDepth x filterWidth1 x filterHeight1 = 8 x 1 x 5 x 5
    vd1 mConvBiases1;  // dim: numFilters1 = 8
    vd4 mConvFilters2; // dim: numFilters2 x numFilters1 x filterWidth2 x filterHeight2 = 8 x 8 x 5 x 5
    vd1 mConvBiases2;  // dim: numFilters2 = 8
    vd2 mFcWeights;    // dim: numFcNodes x numOutputs = 800 x 10
    vd1 mFcBiases;     // dim: numOutputs

    // Total parameters = 9826

    vd1 predict(const vd3& image) const;
    vd1 compute_gradients(const vd3& image, const vd1& labels, vd4& outDFilt1, vd1& outDBias1, vd4& outDFilt2, vd1& outDBias2, vd2& outDTheta3, vd1& outDBias3) const;
    vd1 train(const vd3& batch, double learningRate, double mu);
    void setup(int imageWidth, int imageDepth, int conv1Num, int conv1Size, int conv2Num, int conv2Size);
    void randomize_weights(long seed);
};

vd2 convolve(const vd3& input, const vd3& filter, double bias);
vd3 max_pool(const vd3& input, int strideX, int strideY);
vd1 flatten(const vd3& input);
vd3 expand(const vd1& input, size_t dims[3]);
vd1 soft_max(const vd1& input);
vd3 activation_relu(const vd3& input);
vd3 activation_derivative_relu(const vd3& input);

vd1 fully_connected(const vd1& input, const vd2& weights, const vd1& bias);

convolutional_neural_network merge_cnns(const std::vector<convolutional_neural_network>& cnns);

#endif //APT_PROJECT_CONVOLUTIONAL_NEURAL_NETWORK_H
