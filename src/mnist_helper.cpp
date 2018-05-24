#include <fstream>
#include <iostream>
#include "mnist_helper.h"

std::vector<std::vector<double>> read_images(const std::string& filename)
{
  std::vector<std::vector<char>> imageData;

  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.good())
  {
    std::cout << "Failed to load images file '" << filename << "'\n";
    return std::vector<std::vector<double>>();
  }

  unsigned char magic[4];
  ifs.read((char*)magic, 4);
  unsigned char nImages[4];
  ifs.read((char*)nImages, 4);
  unsigned char nRows[4];
  ifs.read((char*)nRows, 4);
  unsigned char nCols[4];
  ifs.read((char*)nCols, 4);

  int numImages = nImages[0] << 24 | nImages[1] << 16 | nImages[2] << 8 | nImages[3];
  int numRows = nRows[0] << 24 | nRows[1] << 16 | nRows[2] << 8 | nRows[3];
  int numCols = nCols[0] << 24 | nCols[1] << 16 | nCols[2] << 8 | nCols[3];

  int numPixels = numImages * numRows * numCols;
  std::vector<char> pixelData(numPixels);
  ifs.read(pixelData.data(), numImages * numRows * numCols);

  for (int i = 0; i < numImages; i++)
  {
    imageData.push_back(std::vector<char>(pixelData.begin() + (i * numRows * numCols), pixelData.begin() + ((i + 1) * numRows * numCols)));
  }

  std::vector<std::vector<double>> imageDataDouble(imageData.size(), std::vector<double>(imageData[0].size()));
  for (int i = 0; i < imageData.size(); i++)
  {
    for (int j = 0; j < imageData[0].size(); j++)
    {
      imageDataDouble[i][j] = ((double)((unsigned char)imageData[i][j])) / 255.0 - 0.5;
    }
  }

  return imageDataDouble;
}

std::vector<double> read_labels(const std::string& filename)
{
  std::vector<char> labelData;

  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.good())
  {
    std::cout << "Failed to load labels file '" << filename << "'\n";
    return std::vector<double>();
  }

  unsigned char magic[4];
  ifs.read((char*)magic, 4);
  unsigned char nImages[4];
  ifs.read((char*)nImages, 4);

  int numImages = nImages[0] << 24 | nImages[1] << 16 | nImages[2] << 8 | nImages[3];

  labelData.resize(numImages);
  ifs.read(labelData.data(), numImages);

  std::vector<double> labelDataDouble(labelData.size());
  for (int i = 0; i < labelData.size(); i++)
  {
    labelDataDouble[i] = ((double)((unsigned char)labelData[i]));
  }

  return labelDataDouble;
}