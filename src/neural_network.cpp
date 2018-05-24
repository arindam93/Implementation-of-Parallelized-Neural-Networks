#include "neural_network.h"

#include <iostream>

neural_network::neural_network(unsigned long seed)
{
  mRng.seed(seed);
}

void neural_network::randomize_weights()
{
  std::uniform_real_distribution<double> u01(0, 1);

  for (int layer = 0; layer < mLayers.size(); layer++)
  {
    for (int node = 0; node < mLayers[layer].mNodes.size(); node++)
    {
      for (int i = 0; i < mLayers[layer].mNodes[node].mWeights.size(); i++)
      {
        mLayers[layer].mNodes[node].mWeights[i] = 2 * (u01(mRng) - 0.5);
      }
    }
  }
}

std::vector<double> neural_network::infer(const std::vector<double>& inputActivations)
{
  std::vector<double> input = inputActivations;

  for (int layer = 0; layer < mLayers.size(); layer++)
  {
    std::vector<double> newInput(mLayers[layer].mNodes.size());

    for (int node = 0; node < mLayers[layer].mNodes.size(); node++)
    {
      double activation = mLayers[layer].mNodes[node].mBias;
      for (int i = 0; i < input.size(); i++)
      {
        activation += mLayers[layer].mNodes[node].mWeights[i] * input[i];
      }
      mLayers[layer].mNodes[node].mActivation = activation;
      mLayers[layer].mNodes[node].mOutput = transfer(mLayers[layer].mNodes[node].mActivation);

      newInput[node] = mLayers[layer].mNodes[node].mOutput;
    }

    input = newInput;
  }

  return input;
}

double neural_network::transfer(double activation) const
{
  return 1.0 / (1.0 + std::exp(-activation));
}

double neural_network::transfer_derivative(double output) const
{
  return output * (1.0 - output);
}

void neural_network::backpropagate_error(const std::vector<double>& input, const std::vector<double>& expected)
{
  std::vector<double> output = infer(input);

  for (int layer = mLayers.size() - 1; layer >= 0; layer--)
  {
    for (int node = 0; node < mLayers[layer].mNodes.size(); node++)
    {
      double error;
      if (layer == mLayers.size() - 1)
      {
        error = expected[node] - mLayers[layer].mNodes[node].mOutput;
      }
      else
      {
        error = 0.0;
        for (int i = 0; i < mLayers[layer + 1].mNodes.size(); i++)
        {
          error += mLayers[layer + 1].mNodes[i].mWeights[node] * mLayers[layer + 1].mNodes[i].mDelta;
        }
      }
      mLayers[layer].mNodes[node].mDelta = error * transfer_derivative(mLayers[layer].mNodes[node].mOutput);
    }
  }
}

void neural_network::update_weights(const std::vector<double>& input, double learningRate)
{
  for (int layer = 0; layer < mLayers.size(); layer++)
  {
    std::vector<double> inputs;
    if (layer == 0)
    {
      inputs = input;
    }
    else
    {
      for (int i = 0; i < mLayers[layer - 1].mNodes.size(); i++)
      {
        inputs.push_back(mLayers[layer - 1].mNodes[i].mOutput);
      }
    }

    for (int node = 0; node < mLayers[layer].mNodes.size(); node++)
    {
      for (int i = 0; i < inputs.size(); i++)
      {
        mLayers[layer].mNodes[node].mWeights[i] += learningRate * mLayers[layer].mNodes[node].mDelta * inputs[i];
      }
      mLayers[layer].mNodes[node].mBias += learningRate * mLayers[layer].mNodes[node].mDelta;
    }
  }
}

// Note: nodesPerLayer includes the input layer, but the network does not
neural_network create_network(long seed, const std::vector<int>& nodesPerLayer)
{
  neural_network nn(seed);

  nn.mLayers.resize(nodesPerLayer.size() - 1);
  for (int layer = 0; layer < nn.mLayers.size(); layer++)
  {
    nn.mLayers[layer].mNodes.resize(nodesPerLayer[layer + 1]);
  }

  for (int layer = 0; layer < nn.mLayers.size(); layer++)
  {
    for (int node = 0; node < nn.mLayers[layer].mNodes.size(); node++)
    {
      nn.mLayers[layer].mNodes[node].mBias = 0.0;
      nn.mLayers[layer].mNodes[node].mWeights.resize(nodesPerLayer[layer], 1.0);
    }
  }

  return nn;
}

neural_network merge_networks(const std::vector<neural_network>& nns)
{
  neural_network nn = nns[0];
  for (int i = 0; i < nn.mLayers.size(); i++)
  {
    for (int j = 0; j < nn.mLayers[i].mNodes.size(); j++)
    {
      for (int l = 1; l < nns.size(); l++)
      {
        nn.mLayers[i].mNodes[j].mBias += nns[l].mLayers[i].mNodes[j].mBias;
      }
      nn.mLayers[i].mNodes[j].mBias /= nns.size();
      for (int k = 0; k < nn.mLayers[i].mNodes[j].mWeights.size(); k++)
      {
        for (int l = 1; l < nns.size(); l++)
        {
          nn.mLayers[i].mNodes[j].mWeights[k] += nns[l].mLayers[i].mNodes[j].mWeights[k];
        }
        nn.mLayers[i].mNodes[j].mWeights[k] /= nns.size();
      }
    }
  }
  return nn;
}