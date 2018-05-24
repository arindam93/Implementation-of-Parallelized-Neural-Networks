#include "convolutional_neural_network.h"

#include <cmath>
#include <random>
#include <functional>
#include <iostream>
#include <iomanip>

vd2 convolve(const vd3 &input, const vd3 &filter, double bias)
{
  // input and filter need to have the same depth (first dimension)
  vd2 result(input[0].size() - filter[0].size() + 1, vd1(input[0][0].size() - filter[0][0].size() + 1));
  for (int x = 0; x < result.size(); x++)
  {
    for (int y = 0; y < result[0].size(); y++)
    {
      double val = 0;
      for (int fx = 0; fx < filter[0].size(); fx++)
      {
        for (int fy = 0; fy < filter[0][0].size(); fy++)
        {
          for (int i = 0; i < input.size(); i++)
          {
            val += input[i][x + fx][y + fy] * filter[i][fx][fy];
          }
        }
      }
      result[x][y] = val + bias;
    }
  }
  return result;
}

vd3 max_pool(const vd3 &input, int strideX, int strideY)
{
  // input images width/height need to be multiples of strideX/Y
  vd3 result(input.size(), vd2(input[0].size() / strideX, vd1(input[0][0].size() / strideY)));
  for (int i = 0; i < input.size(); i++)
  {
    for (int x = 0; x < input[0].size(); x += strideX)
    {
      for (int y = 0; y < input[0][0].size(); y += strideY)
      {
        double maxVal = input[i][x][y];
        for (int dx = 0; dx < strideX; dx++)
        {
          for (int dy = 0; dy < strideY; dy++)
          {
            maxVal = std::max(maxVal, input[i][x + dx][y + dy]);
          }
        }
        result[i][x / strideX][y / strideY] = maxVal;
      }
    }
  }
  return result;
}

vd1 flatten(const vd3& input)
{
  vd1 result(input.size() * input[0].size() * input[0][0].size());
  for (int i = 0; i < input.size(); i++)
  {
    for (int x = 0; x < input[0].size(); x++)
    {
      for (int y = 0; y < input[0][0].size(); y++)
      {
        result[i * input[0].size() * input[0][0].size() + y * input[0].size() + x] = input[i][x][y];
      }
    }
  }
  return result;
}

vd3 expand(const vd1& input, size_t dims[3])
{
  vd3 result(dims[0], vd2(dims[1], vd1(dims[2])));
  for (int i = 0; i < dims[0]; i++)
  {
    for (int x = 0; x < dims[1]; x++)
    {
      for (int y = 0; y < dims[2]; y++)
      {
        result[i][x][y] = input[i * dims[1] * dims[2] + y * dims[1] + x];
      }
    }
  }
  return result;
}

vd1 soft_max(const vd1 &input)
{
  vd1 result(input.size());
  double total = 0;
  for (int i = 0; i < input.size(); i++)
  {
    result[i] = std::exp(input[i]);
    total += result[i];
  }
  double totalInv = 1.0 / total;
  for (int i = 0; i < input.size(); i++)
  {
    result[i] *= totalInv;
  }
  return result;
}

vd3 activation_relu(const vd3& input)
{
  vd3 result(input.size(), vd2(input[0].size(), vd1(input[0][0].size(), 0)));
  for (int i = 0; i < input.size(); i++)
  {
    for (int x = 0; x < input[0].size(); x++)
    {
      for (int y = 0; y < input[0][0].size(); y++)
      {
        if (input[i][x][y] > 0)
        {
          result[i][x][y] = input[i][x][y];
        }
      }
    }
  }
  return result;
}

vd3 activation_derivative_relu(const vd3& input)
{
  vd3 result(input.size(), vd2(input[0].size(), vd1(input[0][0].size(), 0)));
  for (int i = 0; i < input.size(); i++)
  {
    for (int x = 0; x < input[0].size(); x++)
    {
      for (int y = 0; y < input[0][0].size(); y++)
      {
        if (input[i][x][y] > 0)
        {
          result[i][x][y] = 1;
        }
      }
    }
  }
  return result;
}

vd1 fully_connected(const vd1& input, const vd2& weights, const vd1& bias)
{
  vd1 result = bias;
  for (int i = 0; i < weights.size(); i++)
  {
    for (int j = 0; j < weights[0].size(); j++)
    {
      result[j] += input[i] * weights[i][j];
    }
  }
  return result;
}

vd1 convolutional_neural_network::predict(const vd3 &image) const
{
  // First convolution

  vd3 conv1;
  for (int i = 0; i < mConvFilters1.size(); i++)
  {
    vd2 conv1i = convolve(image, mConvFilters1[i], mConvBiases1[i]);
    conv1.push_back(conv1i);
  }

  // First relu activation

  conv1 = activation_relu(conv1);

  // Second convolution

  vd3 conv2;
  for (int i = 0; i < mConvFilters2.size(); i++)
  {
    vd2 conv2i = convolve(conv1, mConvFilters2[i], mConvBiases2[i]);
    conv2.push_back(conv2i);
  }

  // Second relu activation

  conv2 = activation_relu(conv2);

  // Max pooling

  vd3 pooled = max_pool(conv2, 2, 2);

  // Fully connected

  vd1 fullyConnected = flatten(pooled);
  vd1 output = fully_connected(fullyConnected, mFcWeights, mFcBiases);
  vd1 probs = soft_max(output);

  return probs;
}

vd1 convolutional_neural_network::compute_gradients(const vd3 &image, const vd1 &labels, vd4 &outDFilt1, vd1 &outDBias1, vd4 &outDFilt2, vd1 &outDBias2, vd2 &outDTheta3, vd1 &outDBias3) const
{

  vd1 out(2);

  // ====================================
  // First, the same code as in "predict". This is duplicated because intermediate variables are needed
  // ====================================

  // First convolution

  vd3 conv1;
  for (int i = 0; i < mConvFilters1.size(); i++)
  {
    vd2 conv1i = convolve(image, mConvFilters1[i], mConvBiases1[i]);
    conv1.push_back(conv1i);
  }

  // First relu activation

  conv1 = activation_relu(conv1);

  // Second convolution

  vd3 conv2;
  for (int i = 0; i < mConvFilters2.size(); i++)
  {
    vd2 conv2i = convolve(conv1, mConvFilters2[i], mConvBiases2[i]);
    conv2.push_back(conv2i);
  }

  // Second relu activation

  conv2 = activation_relu(conv2);

  // Max pooling

  vd3 pooled = max_pool(conv2, 2, 2);

  // Fully connected

  vd1 fullyConnected = flatten(pooled);
  vd1 output = fully_connected(fullyConnected, mFcWeights, mFcBiases);
  vd1 probs = soft_max(output);

  // Check if prediction was correct

  double maxP = 0;
  int prediction = 0;
  double maxLabel = 0;
  int label = 0;
  for (int i = 0; i < probs.size(); i++)
  {
    if (probs[i] > maxP)
    {
      maxP = probs[i];
      prediction = i;
    }
    if (labels[i] > maxLabel)
    {
      maxLabel = labels[i];
      label = i;
    }
  }

  double probability = 0;
  for (int i = 0; i < probs.size(); i++)
  {
    probability += probs[i] * labels[i];
  }

  out[0] = prediction == label;
  out[1] = -std::log(probability);

  // ====================================
  // Second, backpropagate
  // ====================================

  vd1 dOut(probs.size());
  for (int i = 0; i < probs.size(); i++)
  {
    dOut[i] = labels[i] * std::log(probs[i]);
  }

  vd2 dTheta3(fullyConnected.size(), vd1(dOut.size()));
  for (int i = 0; i < fullyConnected.size(); i++)
  {
    for (int j = 0; j < dOut.size(); j++)
    {
      dTheta3[i][j] = dOut[j] * fullyConnected[i];
    }
  }

  vd1 dBias3 = dOut;

  vd1 dFc1(fullyConnected.size(), 0);
  for (int i = 0; i < dOut.size(); i++)
  {
    for (int j = 0; j < fullyConnected.size(); j++)
    {
      dFc1[j] += mFcWeights[j][i] * dOut[i];
    }
  }

  size_t dims[] = { pooled.size(), pooled[0].size(), pooled[0][0].size() };
  vd3 dPool = expand(dFc1, dims);

  // Un-pooling
  vd3 dConv2(conv2.size(), vd2(conv2[0].size(), vd1(conv2[0][0].size())));
  for (int i = 0; i < dConv2.size(); i++)
  {
    for (int x = 0; x < dConv2[0].size(); x += 2)
    {
      for (int y = 0; y < dConv2[0][0].size(); y += 2)
      {
        double maxVal = conv2[i][x][y];
        int maxDx = 0;
        int maxDy = 0;
        for (int dx = 0; dx < 2; dx++)
        {
          for (int dy = 0; dy < 2; dy++)
          {
            if (conv2[i][x + dx][y + dy] > maxVal)
            {
              maxVal = conv2[i][x + dx][y + dy];
              maxDx = dx;
              maxDy = dy;
            }
          }
        }
        dConv2[i][x + maxDx][y + maxDy] = dPool[i][x / 2][y / 2];
      }
    }
  }

  // Relu
  dConv2 = activation_relu(dConv2);

  vd3 dConv1(conv1.size(), vd2(conv1[0].size(), vd1(conv1[0][0].size(), 0)));

  vd4 dFilt2(mConvFilters2.size(), vd3(mConvFilters2[0].size(), vd2(mConvFilters2[0][0].size(), vd1(mConvFilters2[0][0][0].size(), 0))));
  vd1 dBias2(mConvFilters2.size(), 0);

  for (int i = 0; i < mConvFilters2.size(); i++)
  {
    for (int x = 0; x < (image[0].size() - mConvFilters1[0][0].size() - mConvFilters2[0][0].size() + 2); x++)
    {
      for (int y = 0; y < (image[0].size() - mConvFilters1[0][0].size() - mConvFilters2[0][0].size() + 2); y++)
      {
        for (int dx = 0; dx < mConvFilters2[0][0].size(); dx++)
        {
          for (int dy = 0; dy < mConvFilters2[0][0].size(); dy++)
          {
            for (int depth = 0; depth < mConvFilters2[0].size(); depth++)
            {
              dFilt2[i][depth][dx][dy] += dConv2[i][x][y] * conv1[depth][x + dx][y + dy];
              dConv1[depth][x + dx][y + dy] += dConv2[i][x][y] * mConvFilters2[i][depth][dx][dy];
            }
          }
        }
      }
    }
    for (int x = 0; x < dConv2[0].size(); x++)
    {
      for (int y = 0; y < dConv2[0][0].size(); y++)
      {
        dBias2[i] += dConv2[i][x][y];
      }
    }
  }

  dConv1 = activation_relu(dConv1);

  vd4 dFilt1(mConvFilters1.size(), vd3(mConvFilters1[0].size(), vd2(mConvFilters1[0][0].size(), vd1(mConvFilters1[0][0][0].size(), 0))));
  vd1 dBias1(mConvFilters1.size(), 0);

  for (int i = 0; i < mConvFilters1.size(); i++)
  {
    for (int x = 0; x < (image[0].size() - mConvFilters1[0][0].size() + 1); x++)
    {
      for (int y = 0; y < (image[0].size() - mConvFilters1[0][0].size() + 1); y++)
      {
        for (int dx = 0; dx < mConvFilters1[0][0].size(); dx++)
        {
          for (int dy = 0; dy < mConvFilters1[0][0].size(); dy++)
          {
            for (int depth = 0; depth < mConvFilters1[0].size(); depth++)
            {
              dFilt1[i][depth][dx][dy] += dConv1[i][x][y] * image[depth][x + dx][y + dy];
            }
          }
        }
      }
    }
    for (int x = 0; x < dConv1[0].size(); x++)
    {
      for (int y = 0; y < dConv1[0][0].size(); y++)
      {
        dBias1[i] += dConv1[i][x][y];
      }
    }
  }

  outDFilt1 = dFilt1;
  outDBias1 = dBias1;
  outDFilt2 = dFilt2;
  outDBias2 = dBias2;
  outDTheta3 = dTheta3;
  outDBias3 = dBias3;

  return out;
}

vd1 convolutional_neural_network::train(const vd3& batch, double learningRate, double mu) 
{
  size_t dims[] = {mImageDepth, mImageWidth, mImageWidth};

  vd4 X;
  vd2 y;
  for (int i =0 ; i < batch.size(); i++)
  {
    X.push_back(expand(batch[i][0], dims));
    y.push_back(batch[i][1]);
  }
  
  int num_correct = 0;
  double cost_val = 0;

  vd4 dfilt1(mConvFilters1.size(), vd3(mConvFilters1[0].size(), vd2(mConvFilters1[0][0].size(), vd1(mConvFilters1[0][0][0].size(), 0))));
  vd4 dfilt2(mConvFilters2.size(), vd3(mConvFilters2[0].size(), vd2(mConvFilters2[0][0].size(), vd1(mConvFilters2[0][0][0].size(), 0))));

  vd1 dbias1(mConvBiases1.size(), 0);
  vd1 dbias2(mConvBiases2.size(), 0);

  vd4 v1(mConvFilters1.size(), vd3(mConvFilters1[0].size(), vd2(mConvFilters1[0][0].size(), vd1(mConvFilters1[0][0][0].size(), 0))));
  vd4 v2(mConvFilters2.size(), vd3(mConvFilters2[0].size(), vd2(mConvFilters2[0][0].size(), vd1(mConvFilters2[0][0][0].size(), 0))));

  vd1 bv1(mConvBiases1.size(), 0);
  vd1 bv2(mConvBiases2.size(), 0);

  vd2 dweight3(mFcWeights.size(), vd1(mFcWeights[0].size(), 0));
  vd1 dbias3(mFcBiases.size(), 0);
  vd2 v3(mFcWeights.size(), vd1(mFcWeights[0].size(), 0)); 
  vd1 bv3(mFcBiases.size(),0);

  for (int i = 0; i< batch.size(); i++)
  {
    vd4 dfilt1_, dfilt2_;
    vd1 dbias1_, dbias2_;
    vd2 dweight3_;
    vd1 dbias3_;
    vd1 result = compute_gradients(X[i], y[i], dfilt1_, dbias1_, dfilt2_, dbias2_, dweight3_, dbias3_);

    for (int j = 0; j < dfilt2.size(); j++)
    {
      for (int k = 0; k < dfilt2[0].size(); k++)
      {
        for (int l = 0; l < dfilt2[0][0].size(); l++)
        {
          for (int m = 0; m < dfilt2[0][0][0].size(); m++)
          {
            dfilt2[j][k][l][m] += dfilt2_[j][k][l][m];
          }
        }
      }
    }

    for (int j = 0; j < dbias2.size(); j++)
    {
      dbias2[j] += dbias2_[j];
    }

    for (int j = 0; j < dfilt1.size(); j++)
    {
      for (int k = 0; k < dfilt1[0].size(); k++)
      {
        for (int l = 0; l < dfilt1[0][0].size(); l++)
        {
          for (int m = 0; m < dfilt1[0][0][0].size(); m++)
          {
            dfilt1[j][k][l][m] += dfilt1_[j][k][l][m];
          }
        }
      }
    }

    for (int j = 0; j < dbias1.size(); j++)
    {
      dbias1[j] += dbias1_[j];
    }

    for (int l = 0; l < dweight3.size(); l++)
    {
      for (int m = 0; m < dweight3[0].size(); m++)
      {
        dweight3[l][m] += dweight3_[l][m];
      }
    }

    for (int l = 0; l < dbias3.size(); l++)
    {
      dbias3[l] += dbias3_[l];
    }

    cost_val += result[1];
    num_correct += result[0];

  }

  for (int j = 0; j < mConvFilters1.size(); j++)
  {
    for (int k = 0; k < mConvFilters1[0].size(); k++)
    {
      for (int l = 0; l < mConvFilters1[0][0].size(); l++)
      {
        for (int m = 0; m < mConvFilters1[0][0][0].size(); m++)
        {
          v1[j][k][l][m] = mu*v1[j][k][l][m] - learningRate * dfilt1[j][k][l][m] / batch.size();
          mConvFilters1[j][k][l][m] += v1[j][k][l][m];
        }
      }
    }
  }

  for (int j = 0; j < mConvBiases1.size(); j++)
  {
    bv1[j] = mu * bv1[j] - learningRate * dbias1[j] / batch.size();
    mConvBiases1[j] += bv1[j];
  }

  for (int j = 0; j < mConvFilters2.size(); j++)
  {
    for (int k = 0; k < mConvFilters2[0].size(); k++)
    {
      for (int l = 0; l < mConvFilters2[0][0].size(); l++)
      {
        for (int m = 0; m < mConvFilters2[0][0][0].size(); m++)
        {
          v2[j][k][l][m] = mu * v2[j][k][l][m] - learningRate * dfilt2[j][k][l][m] / batch.size();
          mConvFilters2[j][k][l][m] += v2[j][k][l][m];
        }
      }
    }
  }

  for (int j = 0; j < mConvBiases2.size(); j++)
  {
    bv2[j] = mu * bv2[j] - learningRate * dbias2[j] / batch.size();
    mConvBiases2[j] += bv2[j];
  }

  for (int l = 0; l < mFcWeights.size(); l++)
  {
    for (int m = 0; m < mFcWeights[0].size(); m++)
    {
      v3[l][m] = mu * v3[l][m] - learningRate * dweight3[l][m] / batch.size();
      mFcWeights[l][m] += v3[l][m];
    }
  }

  for (int l = 0; l < mFcBiases.size(); l++)
  {
    bv3[l] = mu * bv3[l] - learningRate * dbias3[l] / batch.size();
    mFcBiases[l] += bv3[l];
  }
  

  cost_val = cost_val/batch.size();

  double accuracy = (double) num_correct/batch.size();

  vd1 result = {accuracy, cost_val};

  return result;
}

void convolutional_neural_network::setup(int imageWidth, int imageDepth, int conv1Num, int conv1Size, int conv2Num, int conv2Size)
{
  int size1 = imageWidth - conv1Size + 1; // After 1st convolution
  int size2 = size1 - conv2Size + 1; // After 2nd convolution
  int size3 = size2 / 2; // After pooling
  int numOutputs = 10;

  mImageWidth = imageWidth;
  mImageDepth = imageDepth;

  mConvFilter1Width = conv1Size;
  mConvFilter2Width = conv2Size;

  mConvFilter1Depth = imageDepth;
  mConvFilter2Depth = conv1Num;

  mConvFilters1.resize(conv1Num, vd3(imageDepth, vd2(conv1Size, vd1(conv1Size))));
  mConvBiases1.resize(conv1Num);

  mConvFilters2.resize(conv2Num, vd3(conv1Num, vd2(conv2Size, vd1(conv2Size))));
  mConvBiases2.resize(conv2Num);

  mFcWeights.resize(size3 * size3 * conv2Num, vd1(numOutputs));
  mFcBiases.resize(numOutputs);

  mNumFullyConnected = conv2Num * size3 * size3;
  mNumOutputs = numOutputs;
}

void convolutional_neural_network::randomize_weights(long seed)
{
  std::default_random_engine rng(seed);

  std::uniform_real_distribution<double> uniformTheta(0, 0.01);
  std::normal_distribution<double> normConv1(0, std::sqrt(1.0 / (mConvFilters1[0].size() * mConvFilters1[0][0].size() * mConvFilters1[0][0][0].size())));
  std::normal_distribution<double> normConv2(0, std::sqrt(1.0 / (mConvFilters2[0].size() * mConvFilters2[0][0].size() * mConvFilters2[0][0][0].size())));

  for (int i = 0; i < mConvFilters1.size(); i++)
  {
    for (int j = 0; j < mConvFilters1[0].size(); j++)
    {
      for (int k = 0; k < mConvFilters1[0][0].size(); k++)
      {
        for (int l = 0; l < mConvFilters1[0][0][0].size(); l++)
        {
          mConvFilters1[i][j][k][l] = normConv1(rng);
        }
      }
    }
  }
  for (int i = 0; i < mConvBiases1.size(); i++)
  {
    mConvBiases1[i] = 0;
  }
  for (int i = 0; i < mConvFilters2.size(); i++)
  {
    for (int j = 0; j < mConvFilters2[0].size(); j++)
    {
      for (int k = 0; k < mConvFilters2[0][0].size(); k++)
      {
        for (int l = 0; l < mConvFilters2[0][0][0].size(); l++)
        {
          mConvFilters2[i][j][k][l] = normConv2(rng);
        }
      }
    }
  }
  for (int i = 0; i < mConvBiases2.size(); i++)
  {
    mConvBiases2[i] = 0;
  }
  for (int i = 0; i < mFcWeights.size(); i++)
  {
    for (int j = 0; j < mFcWeights[0].size(); j++)
    {
      mFcWeights[i][j] = uniformTheta(rng);
    }
  }
  for (int i = 0; i < mFcBiases.size(); i++)
  {
    mFcBiases[i] = 0;
  }
}

convolutional_neural_network merge_cnns(const std::vector<convolutional_neural_network>& cnns)
{
  convolutional_neural_network cnn = cnns[0];
  double invNum = 1.0 / cnns.size();

  for (int i = 0; i < cnn.mConvFilters1.size(); i++)
  {
    for (int j = 0; j < cnn.mConvFilters1[0].size(); j++)
    {
      for (int k = 0; k < cnn.mConvFilters1[0][0].size(); k++)
      {
        for (int l = 0; l < cnn.mConvFilters1[0][0][0].size(); l++)
        {
          for (int n = 1; n < cnns.size(); n++)
          {
            cnn.mConvFilters1[i][j][k][l] += cnns[n].mConvFilters1[i][j][k][l];
          }
          cnn.mConvFilters1[i][j][k][l] *= invNum;
        }
      }
    }
  }
  for (int i = 0; i < cnn.mConvBiases1.size(); i++)
  {
    for (int n = 1; n < cnns.size(); n++)
    {
      cnn.mConvBiases1[i] += cnns[n].mConvBiases1[i];
    }
    cnn.mConvBiases1[i] *= invNum;
  }
  for (int i = 0; i < cnn.mConvFilters2.size(); i++)
  {
    for (int j = 0; j < cnn.mConvFilters2[0].size(); j++)
    {
      for (int k = 0; k < cnn.mConvFilters2[0][0].size(); k++)
      {
        for (int l = 0; l < cnn.mConvFilters2[0][0][0].size(); l++)
        {
          for (int n = 1; n < cnns.size(); n++)
          {
            cnn.mConvFilters2[i][j][k][l] += cnns[n].mConvFilters2[i][j][k][l];
          }
          cnn.mConvFilters2[i][j][k][l] *= invNum;
        }
      }
    }
  }
  for (int i = 0; i < cnn.mConvBiases2.size(); i++)
  {
    for (int n = 1; n < cnns.size(); n++)
    {
      cnn.mConvBiases2[i] += cnns[n].mConvBiases2[i];
    }
    cnn.mConvBiases2[i] *= invNum;
  }
  for (int i = 0; i < cnn.mFcWeights.size(); i++)
  {
    for (int j = 0; j < cnn.mFcWeights[0].size(); j++)
    {
      for (int n = 1; n < cnns.size(); n++)
      {
        cnn.mFcWeights[i][j] += cnns[n].mFcWeights[i][j];
      }
      cnn.mFcWeights[i][j] *= invNum;
    }
  }
  for (int i = 0; i < cnn.mFcBiases.size(); i++)
  {
    for (int n = 1; n < cnns.size(); n++)
    {
      cnn.mFcBiases[i] += cnns[n].mFcBiases[i];
    }
    cnn.mFcBiases[i] *= invNum;
  }
  return cnn;
}