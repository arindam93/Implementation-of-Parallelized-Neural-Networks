#pragma once
#include <vector>
#include <random>

class neural_network
{
public:

  class node
  {
  public:
    std::vector<double> mWeights;
    double mBias;

    double mActivation;
    double mOutput;
    double mDelta;
  };

  class layer
  {
  public:
    std::vector<node> mNodes;
  };

  std::default_random_engine mRng;

  std::vector<layer> mLayers;

  neural_network(unsigned long seed);

  void randomize_weights();

  std::vector<double> infer(const std::vector<double>& expected);

  double transfer(double activation) const;
  double transfer_derivative(double activation) const;

  void backpropagate_error(const std::vector<double>& input, const std::vector<double>& output);
  void update_weights(const std::vector<double>& input, double learningRate);
};

neural_network create_network(long seed, const std::vector<int>& nodesPerLayer);
neural_network merge_networks(const std::vector<neural_network>& nns);