//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Test to measure performance of compress/decompression using sycl-compress
// library. This is not run by default in the test suite.

// Takes input the dataset of SPIRV files and (de)compresses them using ZSTD.
// Stores the compression, decompression time in a CSV file.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sycl-compress/sycl-compress.h>
#include <vector>

#define NUM_WORKLOADS 1
#define MAX_WORKLOAD_SIZE 1024 * 1024 * 100 // 100 MB
#define ZSTD_MIN_COMPRESSION_LEVEL 1
#define ZSTD_MAX_COMPRESSION_LEVEL 22

// Generate a random buffer of data with size in the range [1,
// MAX_WORKLOAD_SIZE] Return the buffer and its size (in workloadSize)
const char *GenerateRandonWorkload(size_t &workloadSize) {

  // Get randon size in the range [1, MAX_WORKLOAD_SIZE]
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1, MAX_WORKLOAD_SIZE);
  workloadSize = static_cast<size_t>(dis(gen));

  // Allocate heap buffer.
  char *wokload = static_cast<char *>(malloc(workloadSize));

  // Populate buffer with random data.
  std::generate(wokload, wokload + workloadSize,
                [&]() { return static_cast<char>(dis(gen)); });

  return wokload;
}

// Compress workload using ZSTD and the supplied compression level.
// Returns the time taken to compress the workload and the compressed size.
std::chrono::nanoseconds CompressWorkload(const char *workload,
                                          size_t workloadSize, int level,
                                          char *&compressedData,
                                          size_t &compressedSize) {
  auto start = std::chrono::high_resolution_clock::now();
  char *compressed =
      compressBlob(workload, workloadSize, compressedSize, level);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  compressedData = compressed;
  return duration;
}

// Decompress workload using ZSTD.
// Returns the time taken to decompress the workload.
std::chrono::nanoseconds DecompressWorkload(const char *compressedData,
                                            size_t compressedSize,
                                            char *&decompressedData,
                                            size_t &decompressedSize) {
  auto start = std::chrono::high_resolution_clock::now();
  char *decompressed =
      decompressBlob(compressedData, compressedSize, decompressedSize);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::nanoseconds duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  decompressedData = decompressed;
  return duration;
}

// Measure the performance of compressing and decompressing random workloads
// NUM_WORKLOADS times at a give compression level.
// Returns a vector of vectors where each inner vector contains the time taken
// to compress and decompress a workload, workload size, and the compressed
// size.
std::vector<std::vector<uint64_t>> MeasurePerformance(int level) {
  std::vector<std::vector<uint64_t>> results;
  for (int i = 0; i < NUM_WORKLOADS; i++) {

    // Generate random workload.
    size_t workloadSize;
    const char *workload = GenerateRandonWorkload(workloadSize);

    // Compress.
    size_t compressedSize;
    char *compressedData;
    auto compressDuration = CompressWorkload(workload, workloadSize, level,
                                             compressedData, compressedSize);

    // Decompress.
    size_t decompressedSize;
    char *decompressedData;
    auto decompressDuration = DecompressWorkload(
        compressedData, compressedSize, decompressedData, decompressedSize);

    // Check the size of the decompressed data is same as the original workload.
    if (workloadSize != decompressedSize) {
      std::cerr
          << "Error: Decompressed size is not same as original workload size\n";
      std::cerr << "Workload size: " << workloadSize
                << " Decompressed size: " << decompressedSize << "\n";
      exit(1);
    }

    // Save results
    results.push_back({static_cast<uint64_t>(compressDuration.count()),
                       static_cast<uint64_t>(decompressDuration.count()),
                       static_cast<uint64_t>(workloadSize),
                       static_cast<uint64_t>(compressedSize)});

    free(const_cast<char *>(workload));
    free(compressedData);
    free(decompressedData);
  }
  return results;
}

// Run workloads for different compression levels between
// ZSTD_MIN_COMPRESSION_LEVEL and ZSTD_MAX_COMPRESSION_LEVEL. Saves the result
// in a CSV file, with a user-supplied name, with the following columns:
// Compression level | Workload size | Compressed size | Compress duration |
// Decompress duration.
void RunWorkloads(const std::string &outputFile) {

  // Write results to a CSV file.
  // Clear the file if it already exists.
  std::ofstream file;
  file.open(outputFile, std::ofstream::out | std::ofstream::trunc);

  // Write header.
  file << "Compression level,Workload size,Compressed size,Compress duration,"
          "Decompress duration\n";

  try {
    for (int level = ZSTD_MIN_COMPRESSION_LEVEL;
         level <= ZSTD_MAX_COMPRESSION_LEVEL; level++) {

      std::cout << "Running workloads for compression level: " << level << "\n";
      auto levelResults = MeasurePerformance(level);
      for (const auto &result : levelResults) {
        file << level << "," << result[2] << "," << result[3] << ","
             << result[0] << "," << result[1] << "\n";
      }
      file.flush();
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
  }

  file.close();
}

// Takes a full file path as input, loads it into a buffer and (de)compress it
// with different levels. Returns a vector of vectors containing the
// (de)compression time, workload size, and compressed size, for each level.
std::vector<std::vector<uint64_t>>
MeasurePerformanceOfFileCompression(const std::string &filePath) {
  std::vector<std::vector<uint64_t>> results;
  std::ifstream file(filePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file: " << filePath << "\n";
    exit(1);
  }

  // Get file size.
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Allocate buffer to hold file data.
  char *fileData = static_cast<char *>(malloc(fileSize));
  file.read(fileData, fileSize);
  file.close();

  for (int level = ZSTD_MIN_COMPRESSION_LEVEL;
       level <= ZSTD_MAX_COMPRESSION_LEVEL; level++) {
    std::cout << "Running workloads for compression level: " << level << "\n";

    // Compress.
    size_t compressedSize;
    char *compressedData;
    auto compressDuration = CompressWorkload(fileData, fileSize, level,
                                             compressedData, compressedSize);

    // Decompress.
    size_t decompressedSize;
    char *decompressedData;
    auto decompressDuration = DecompressWorkload(
        compressedData, compressedSize, decompressedData, decompressedSize);

    // Check the size of the decompressed data is same as the original workload.
    if (fileSize != decompressedSize) {
      std::cerr
          << "Error: Decompressed size is not same as original workload size\n";
      std::cerr << "Workload size: " << fileSize
                << " Decompressed size: " << decompressedSize << "\n";
      exit(1);
    }
    assert(level >= 0);
    results.push_back({static_cast<uint64_t>(level), fileSize, compressedSize,
                       static_cast<uint64_t>(compressDuration.count()),
                       static_cast<uint64_t>(decompressDuration.count())});
    free(compressedData);
    free(decompressedData);
  }

  free(fileData);
  return results;
}

// Given a directory and output file name, iterate over all files in the
// directory with extension .spv or .spirv and compress/decompress them with
// different levels. Save the results in a CSV file.
void RunWorkloadsForFiles(const std::string &directory,
                          const std::string &outputFile) {

  // Check validity of the input directory path and output file.
  if (!std::filesystem::exists(directory)) {
    std::cerr << "Error: Directory does not exist: " << directory << "\n";
    exit(1);
  }

  // Write results to a CSV file.
  // Clear the file if it already exists.
  std::ofstream file;
  file.open(outputFile, std::ofstream::out | std::ofstream::trunc);

  // Write header.
  file << "FileName, Compression level,Workload size,Compressed size,Compress "
          "duration,"
          "Decompress duration\n";

  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    std::string filePath = entry.path().string();
    if (filePath.find(".spv") != std::string::npos ||
        filePath.find(".spirv") != std::string::npos) {
      std::cout << "Running workloads for file: " << filePath << "\n";
      auto results = MeasurePerformanceOfFileCompression(filePath);

      for (const auto &result : results) {

        file << filePath << "," << result[0] << "," << result[1] << ","
             << result[2] << "," << result[3] << "," << result[4] << "\n";
      }
      file.flush();
    }
  }

  file.close();
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <output-file> <Spirv dataset>\n";
    return 1;
  }

  RunWorkloadsForFiles(argv[2], argv[1]);
  return 0;
}
