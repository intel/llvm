// REQUIRES: aspect-ext_oneapi_bindless_images
// XFAIL: hip
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19957

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

#include <iostream>
#include <numeric>

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc, const std::vector<float> &dataIn1,
    const std::vector<float> &dataIn2, sycl::device dev, sycl::queue q,
    std::vector<float> &out) {

  // Check that output image is double size of input images.
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc1(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemSrc2(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of dataIn1 to first quarter of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, {dataInDesc.width, 0, 0},
                    imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc, copyExtent);

  // Copy second half of dataIn1 to second quarter of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {dataInDesc.width / 2, 0, 0},
                    {dataInDesc.width, 0, 0}, imgMemSrc1.get_handle(),
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, copyExtent);

  // Copy first half of dataIn2 to third quarter of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, {dataInDesc.width, 0, 0},
                    imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc, copyExtent);

  // Copy second half of dataIn2 to fourth quarter of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {dataInDesc.width / 2, 0, 0},
                    {dataInDesc.width, 0, 0}, imgMemSrc2.get_handle(),
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device.
  // Copy first half of imgMemSrc1 to first quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc,
                    imgMemDst.get_handle(), {0, 0, 0}, outDesc, copyExtent);

  // Copy second half of imgMemSrc1 to second quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst.get_handle(),
                    {outDesc.width / 4, 0, 0}, outDesc, copyExtent);

  // Copy first half of imgMemSrc2 to third quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc,
                    imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    copyExtent);

  // Copy second half of imgMemSrc2 to fourth quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst.get_handle(),
                    {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);

  q.wait_and_throw();

  // Copy device data back to host.
  // Extent to copy.
  copyExtent = {outDesc.width / 2, 1, 1};

  // Copy first half of imgMemDst to first half of out data.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, 0, 0}, outDesc, out.data(),
                    {0, 0, 0}, {outDesc.width / 2, 0, 0}, copyExtent);

  // Copy second half of imgMemDst to second half of out data.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    out.data(), {outDesc.width / 2, 0, 0},
                    {outDesc.width / 2, 0, 0}, copyExtent);

  q.wait_and_throw();
}

void copy_image_mem_handle_to_usm(const syclexp::image_descriptor &dataInDesc,
                                  const syclexp::image_descriptor &outDesc,
                                  const std::vector<float> &dataIn1,
                                  const std::vector<float> &dataIn2,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {

  // Check that output image is double size of input images.
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc1(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemSrc2(dataInDesc, dev, q.get_context());

  // Allocate 1D device USM memory. Pitch set to zero as it is a 1D image.
  size_t pitch = 0;
  size_t elements = outDesc.width * outDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elements, q);

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of dataIn1 to first half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, {dataInDesc.width, 0, 0},
                    imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc, copyExtent);

  // Copy second half of dataIn1 to second half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {dataInDesc.width / 2, 0, 0},
                    {dataInDesc.width, 0, 0}, imgMemSrc1.get_handle(),
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, copyExtent);

  // Copy first half of dataIn2 to first half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, {dataInDesc.width, 0, 0},
                    imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc, copyExtent);

  // Copy second half of dataIn2 to second half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {dataInDesc.width / 2, 0, 0},
                    {dataInDesc.width, 0, 0}, imgMemSrc2.get_handle(),
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device.
  // Copy first half of imgMemSrc1 to first quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc, imgMemDst,
                    {0, 0, 0}, outDesc, pitch, copyExtent);

  // Copy second half of imgMemSrc1 to second quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst, {outDesc.width / 4, 0, 0}, outDesc,
                    pitch, copyExtent);

  // Copy first half of imgMemSrc2 to third quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc, imgMemDst,
                    {outDesc.width / 2, 0, 0}, outDesc, pitch, copyExtent);

  // Copy second half of imgMemSrc2 to fourth quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                    outDesc, pitch, copyExtent);

  q.wait_and_throw();

  // Copy device data back to host.
  // Extent to copy.
  copyExtent = {outDesc.width / 2, 1, 1};

  // Copy first half of imgMemDst to first half of out data.
  q.ext_oneapi_copy(imgMemDst, {0, 0, 0}, out.data(), {0, 0, 0}, outDesc, pitch,
                    {outDesc.width, 0, 0}, copyExtent);

  // Copy second half of imgMemDst to second half of out data.
  q.ext_oneapi_copy(imgMemDst, {outDesc.width / 2, 0, 0}, out.data(),
                    {outDesc.width / 2, 0, 0}, outDesc, pitch,
                    {outDesc.width, 0, 0}, copyExtent);

  q.wait_and_throw();

  sycl::free(imgMemDst, q);
}

void copy_usm_to_image_mem_handle(const syclexp::image_descriptor &dataInDesc,
                                  const syclexp::image_descriptor &outDesc,
                                  const std::vector<float> &dataIn1,
                                  const std::vector<float> &dataIn2,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {

  // Check that output image is double size of input images.
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitchSrc1 = 0;
  size_t pitchSrc2 = 0;
  size_t elements = outDesc.width * outDesc.num_channels;
  void *imgMemSrc1 = sycl::malloc_device<float>(elements, q);
  void *imgMemSrc2 = sycl::malloc_device<float>(elements, q);

  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of dataIn1 to first half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, imgMemSrc1, {0, 0, 0},
                    dataInDesc, pitchSrc1, {dataInDesc.width, 0, 0},
                    copyExtent);

  // Copy second half of dataIn1 to second half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {dataInDesc.width / 2, 0, 0}, imgMemSrc1,
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, pitchSrc1,
                    {dataInDesc.width, 0, 0}, copyExtent);

  // Copy first half of dataIn2 to first half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, imgMemSrc2, {0, 0, 0},
                    dataInDesc, pitchSrc2, {dataInDesc.width, 0, 0},
                    copyExtent);

  // Copy second half of dataIn2 to second half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {dataInDesc.width / 2, 0, 0}, imgMemSrc2,
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, pitchSrc2,
                    {dataInDesc.width, 0, 0}, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device.
  // Copy first half of imgMemSrcOne to first quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1, {0, 0, 0}, dataInDesc, pitchSrc1,
                    imgMemDst.get_handle(), {0, 0, 0}, outDesc, copyExtent);

  // Copy second half of imgMemSrcOne to second quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc1, imgMemDst.get_handle(),
                    {outDesc.width / 4, 0, 0}, outDesc, copyExtent);

  // Copy first half of imgMemSrcTwo to third quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2, {0, 0, 0}, dataInDesc, pitchSrc2,
                    imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    copyExtent);

  // Copy second half of imgMemSrcTwo to fourth quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc2, imgMemDst.get_handle(),
                    {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);

  q.wait_and_throw();

  // Copy device data back to host.
  // Extent to copy.
  copyExtent = {outDesc.width / 2, 1, 1};

  // Copy first half of imgMemDst to first half of out data.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, 0, 0}, outDesc, out.data(),
                    {0, 0, 0}, {outDesc.width, 0, 0}, copyExtent);

  // Copy second half of imgMemDst to second half of out data.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    out.data(), {outDesc.width / 2, 0, 0},
                    {outDesc.width, 0, 0}, copyExtent);

  q.wait_and_throw();

  sycl::free(imgMemSrc1, q);
  sycl::free(imgMemSrc2, q);
}

void copy_usm_to_usm(const syclexp::image_descriptor &dataInDesc,
                     const syclexp::image_descriptor &outDesc,
                     const std::vector<float> &dataIn1,
                     const std::vector<float> &dataIn2, sycl::device dev,
                     sycl::queue q, std::vector<float> &out) {

  // Check that output image is double size of input images.
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitchSrc1 = 0;
  size_t pitchSrc2 = 0;
  size_t elementsSrc = outDesc.width * outDesc.num_channels;
  void *imgMemSrc1 = sycl::malloc_device<float>(elementsSrc, q);
  void *imgMemSrc2 = sycl::malloc_device<float>(elementsSrc, q);

  size_t pitchDst = 0;
  size_t elementsDst = outDesc.width * outDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elementsDst, q);

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of dataIn1 to first half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, imgMemSrc1, {0, 0, 0},
                    dataInDesc, pitchSrc1, {dataInDesc.width, 0, 0},
                    copyExtent);

  // Copy second half of dataIn1 to second half of imgMemSrc1.
  q.ext_oneapi_copy(dataIn1.data(), {dataInDesc.width / 2, 0, 0}, imgMemSrc1,
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, pitchSrc1,
                    {dataInDesc.width, 0, 0}, copyExtent);

  // Copy first half of dataIn2 to first half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, imgMemSrc2, {0, 0, 0},
                    dataInDesc, pitchSrc2, {dataInDesc.width, 0, 0},
                    copyExtent);

  // Copy second half of dataIn2 to second half of imgMemSrc2.
  q.ext_oneapi_copy(dataIn2.data(), {dataInDesc.width / 2, 0, 0}, imgMemSrc2,
                    {dataInDesc.width / 2, 0, 0}, dataInDesc, pitchSrc2,
                    {dataInDesc.width, 0, 0}, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device.
  // Copy first half of imgMemSrc1 to first quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1, {0, 0, 0}, dataInDesc, pitchSrc1, imgMemDst,
                    {0, 0, 0}, outDesc, pitchDst, copyExtent);

  // Copy second half of imgMemSrc1 to second quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc1, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc1, imgMemDst, {outDesc.width / 4, 0, 0}, outDesc,
                    pitchDst, copyExtent);

  // Copy first half of imgMemSrc2 to third quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2, {0, 0, 0}, dataInDesc, pitchSrc2, imgMemDst,
                    {outDesc.width / 2, 0, 0}, outDesc, pitchDst, copyExtent);

  // Copy second half of imgMemSrc2 to fourth quarter of imgMemDst.
  q.ext_oneapi_copy(imgMemSrc2, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc2, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                    outDesc, pitchDst, copyExtent);

  q.wait_and_throw();

  // Copy device data back to host.
  // Extent to copy.
  copyExtent = {outDesc.width / 2, 1, 1};

  // Copy first half of imgMemDst to first half of out data.
  q.ext_oneapi_copy(imgMemDst, {0, 0, 0}, out.data(), {0, 0, 0}, outDesc,
                    pitchDst, {outDesc.width, 0, 0}, copyExtent);

  // Copy second half of imgMemDst to second half of out data.
  q.ext_oneapi_copy(imgMemDst, {outDesc.width / 2, 0, 0}, out.data(),
                    {outDesc.width / 2, 0, 0}, outDesc, pitchDst,
                    {outDesc.width, 0, 0}, copyExtent);

  q.wait_and_throw();

  sycl::free(imgMemSrc1, q);
  sycl::free(imgMemSrc2, q);
  sycl::free(imgMemDst, q);
}

bool image_mem_handle_to_image_mem_handle_out_of_bounds_copy(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), dataInDesc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {(dataInDesc.width / 2) + 1, 1, 1};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(imgMemSrc.get_handle(), {dataInDesc.width / 2, 0, 0},
                      dataInDesc, imgMemDst.get_handle(),
                      {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);
  } catch (sycl::exception e) {
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return false;
  }

  return false;
}

bool image_mem_handle_to_usm_out_of_bounds_copy(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc(dataInDesc, dev, q.get_context());

  size_t pitch = 0;
  size_t elements = outDesc.width * outDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elements, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), dataInDesc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {(dataInDesc.width / 2) + 1, 1, 1};

  try {
    // Perform out of bounds copy!
    q.ext_oneapi_copy(imgMemSrc.get_handle(), {dataInDesc.width / 2, 0, 0},
                      dataInDesc, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                      outDesc, pitch, copyExtent);
  } catch (sycl::exception e) {
    sycl::free(imgMemDst, q);
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
  }

  sycl::free(imgMemDst, q);
  return false;
}

bool usm_to_image_mem_handle_out_of_bounds_copy(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitch = 0;
  size_t elements = dataInDesc.width * dataInDesc.num_channels;
  void *imgMemSrc = sycl::malloc_device<float>(elements, q);

  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, dataInDesc, pitch);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {(dataInDesc.width / 2) + 1, 1, 1};

  try {
    // Perform out of bounds copy!
    q.ext_oneapi_copy(imgMemSrc, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                      pitch, imgMemDst.get_handle(),
                      {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);
  } catch (sycl::exception e) {
    sycl::free(imgMemSrc, q);
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
  }

  sycl::free(imgMemSrc, q);
  return false;
}

bool usm_to_usm_out_of_bounds_copy(const syclexp::image_descriptor &dataInDesc,
                                   const syclexp::image_descriptor &outDesc,
                                   const std::vector<float> &dataIn,
                                   sycl::device dev, sycl::queue q) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitchSrc = 0;
  size_t elementsSrc = dataInDesc.width * dataInDesc.num_channels;
  void *imgMemSrc = sycl::malloc_device<float>(elementsSrc, q);

  size_t pitchDst = 0;
  size_t elementsDst = dataInDesc.width * dataInDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elementsDst, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, dataInDesc, pitchSrc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {(dataInDesc.width / 2) + 1, 1, 1};

  try {
    // Perform out of bounds copy!
    q.ext_oneapi_copy(imgMemSrc, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                      pitchSrc, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                      outDesc, pitchDst, copyExtent);
  } catch (sycl::exception e) {
    sycl::free(imgMemSrc, q);
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
  }

  sycl::free(imgMemSrc, q);
  return false;
}

bool check_test(const std::vector<float> &out,
                const std::vector<float> &expected) {
  assert(out.size() == expected.size());
  bool validated = true;
  for (int i = 0; i < out.size(); i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
      std::cout << "Result mismatch at index " << i
                << "! Expected: " << expected[i] << ", Actual: " << out[i]
                << std::endl;
#ifndef VERBOSE_PRINT
      // In CI, only display the first mismatched index
      break;
#endif
    }
  }
  return validated;
}

template <int channelNum, sycl::image_channel_type channelType>
bool run_copy_test(sycl::device &dev, sycl::queue &q, sycl::range<1> dims) {
  std::vector<float> dataIn1(dims.size() / 2);
  std::vector<float> dataIn2(dims.size() / 2);
  std::vector<float> out(dims.size(), 0);

  std::vector<float> expected(dims.size());

  // Create two sets of input data. Each half the size of the output
  // and one beginning sequentually after the other.
  std::iota(dataIn1.begin(), dataIn1.end(), 0);
  std::iota(dataIn2.begin(), dataIn2.end(), (dataIn2.size()));

  // Set expected to be sequential
  std::iota(expected.begin(), expected.end(), 0);

  syclexp::image_descriptor outDesc =
      syclexp::image_descriptor(dims, channelNum, channelType);
  syclexp::image_descriptor dataInDesc =
      syclexp::image_descriptor(dims / 2, channelNum, channelType);

  bool validated = true;

  // Perform copy checks
  copy_image_mem_handle_to_image_mem_handle(dataInDesc, outDesc, dataIn1,
                                            dataIn2, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_image_mem_handle_to_image_mem_handle test failed"
              << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_image_mem_handle_to_usm(dataInDesc, outDesc, dataIn1, dataIn2, dev, q,
                               out);
  if (!check_test(out, expected)) {
    std::cout << "copy_image_mem_handle_to_usm test failed" << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_usm_to_image_mem_handle(dataInDesc, outDesc, dataIn1, dataIn2, dev, q,
                               out);
  if (!check_test(out, expected)) {
    std::cout << "copy_usm_to_image_mem_handle test failed" << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_usm_to_usm(dataInDesc, outDesc, dataIn1, dataIn2, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_usm_to_usm test failed" << std::endl;
    validated = false;
  }

  // Perform out of bounds copy checks
  if (!image_mem_handle_to_image_mem_handle_out_of_bounds_copy(
          dataInDesc, outDesc, dataIn1, dev, q)) {
    std::cout
        << "image_mem_handle_to_image_mem_handle_out_of_bounds_copy test failed"
        << std::endl;
    validated = false;
  }

  if (!image_mem_handle_to_usm_out_of_bounds_copy(dataInDesc, outDesc, dataIn1,
                                                  dev, q)) {
    std::cout << "image_mem_handle_to_usm_out_of_bounds_copy test failed"
              << std::endl;
    validated = false;
  }

  if (!usm_to_image_mem_handle_out_of_bounds_copy(dataInDesc, outDesc, dataIn1,
                                                  dev, q)) {
    std::cout << "usm_to_image_mem_handle_out_of_bounds_copy test failed"
              << std::endl;
    validated = false;
  }

  if (!usm_to_usm_out_of_bounds_copy(dataInDesc, outDesc, dataIn1, dev, q)) {
    std::cout << "usm_to_usm_out_of_bounds_copy test failed" << std::endl;
    validated = false;
  }

  return validated;
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);

  bool validated =
      run_copy_test<1, sycl::image_channel_type::fp32>(dev, q, {12});

  if (!validated) {
    std::cout << "Tests failed\n";
    return 1;
  }

  return 0;
}
