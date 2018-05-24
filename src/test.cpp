#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <omp.h>

#include "neural_network.h"
#include "mnist_helper.h"

int main(int argc, char** argv)
{
    // Load images and labels
    std::cout << "Loading data" << std::endl;
    std::vector<std::vector<double>> trainingImages = read_images("../data/train-images.idx3-ubyte");
    std::vector<double> trainingLabels = read_labels("../data/train-labels.idx1-ubyte");
    std::vector<std::vector<double>> testingImages = read_images("../data/t10k-images.idx3-ubyte");
    std::vector<double> testingLabels = read_labels("../data/t10k-labels.idx1-ubyte");

    if (trainingImages.empty() || trainingLabels.empty() || testingImages.empty() || testingLabels.empty())
    {
        return 1;
    }

    std::cout << "Loaded " << trainingImages.size() << " training images" << std::endl;
    std::cout << "Loaded " << testingImages.size() << " testing images" << std::endl;

    std::vector<std::vector<std::vector<double>>> trainingData;
    for (int i = 0; i < trainingImages.size(); i++)
    {
        std::vector<double> output(10, 0.0);
        output[(unsigned int)((unsigned char)trainingLabels[i])] = 1.0;
        trainingData.push_back({ trainingImages[i], output, { (double) i } });
    }

    std::vector<std::vector<std::vector<double>>> testingData;
    for (int i = 0; i < testingImages.size(); i++)
    {
        std::vector<double> output(10, 0.0);
        output[(unsigned int)((unsigned char)testingLabels[i])] = 1.0;
        testingData.push_back({ testingImages[i], output, { (double) i } });
    }

    std::default_random_engine rng(1259278);

    neural_network nn = create_network(1234, { 784, 100, 10 });
    nn.randomize_weights();

    double learningRate = 0.5;
    int numEpochs = 10;

    int numThreads = omp_get_max_threads();
    int batchLength = 100;

    std::vector<neural_network> nns;
    for (int n = 0; n < numThreads; n++)
    {
        nns.push_back(nn);
    }

    std::cout << "Begin training on " << numThreads << " threads" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        std::cout << "Epoch " << (epoch+1) << " / " << numEpochs << std::endl;
        std::shuffle(trainingData.begin(), trainingData.end(), rng);

        for (int i = 0; i < trainingData.size(); i += numThreads * batchLength)
        {
            for (int n = 0; n < numThreads; n++)
            {
                nns[n] = nn;
            }
#pragma omp parallel for
            for (int n = 0; n < numThreads; n++)
            {
                for (int m = 0; m < batchLength; m++)
                {
                    int idx = i + n * batchLength + m;
                    if (idx < trainingData.size())
                    {
                        nns[n].backpropagate_error(trainingData[idx][0], trainingData[idx][1]);
                        nns[n].update_weights(trainingData[idx][0], learningRate);
                    }
                }
            }
            nn = merge_networks(nns);
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(endTime - startTime);
    std::cout << "Training duration: " << duration.count() << " s" << std::endl;

    std::cout << "Begin testing" << std::endl;
    int top1 = 0;
    for (int i = 0; i < testingData.size(); i++)
    {
        std::vector<double> output = nn.infer(testingData[i][0]);

        int prediction = 0;
        double maxOutput = output[0];
        for (int j = 1; j < 10; j++)
        {
            if (output[j] > maxOutput)
            {
                prediction = j;
                maxOutput = output[j];
            }
        }

        if (testingData[i][1][prediction])
        {
            top1++;
        }
    }

    std::cout << "Top-1 accuracy : " << 100.0 * top1 / testingData.size() << " % ( " << top1 << " / " << testingData.size() << " )" << std::endl;

    return 0;
}