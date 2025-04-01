// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc, const std::vector<float> &dataIn1,
    const std::vector<float> &dataIn2, sycl::device dev, sycl::queue q,
    std::vector<float> &out) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc1(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemSrc2(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn1.data(), imgMemSrc1.get_handle(), dataInDesc);
  q.ext_oneapi_copy(dataIn2.data(), imgMemSrc2.get_handle(), dataInDesc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of imgMemSrcOne to first quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc,
                    imgMemDst.get_handle(), {0, 0, 0}, outDesc, copyExtent);

  // Copy second half of imgMemSrcOne to second quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst.get_handle(),
                    {outDesc.width / 4, 0, 0}, outDesc, copyExtent);

  // Copy first half of imgMemSrcTwo to third quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc,
                    imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    copyExtent);

  // Copy second half of imgMemSrcTwo to fourth quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst.get_handle(),
                    {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst.get_handle(), out.data(), outDesc);

  q.wait_and_throw();
}

void copy_image_mem_handle_to_usm(const syclexp::image_descriptor &dataInDesc,
                                  const syclexp::image_descriptor &outDesc,
                                  const std::vector<float> &dataIn1,
                                  const std::vector<float> &dataIn2,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  syclexp::image_mem imgMemSrc1(dataInDesc, dev, q.get_context());
  syclexp::image_mem imgMemSrc2(dataInDesc, dev, q.get_context());

  // Allocate 1D device USM memory. Pitch set to zero as it is a 1D image
  size_t pitch = 0;
  size_t elements = outDesc.width * outDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elements, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn1.data(), imgMemSrc1.get_handle(), dataInDesc);
  q.ext_oneapi_copy(dataIn2.data(), imgMemSrc2.get_handle(), dataInDesc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of imgMemSrcOne to first quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {0, 0, 0}, dataInDesc, imgMemDst,
                    {0, 0, 0}, outDesc, pitch, copyExtent);

  // Copy second half of imgMemSrcOne to second quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst, {outDesc.width / 4, 0, 0}, outDesc,
                    pitch, copyExtent);

  // Copy first half of imgMemSrcTwo to third quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {0, 0, 0}, dataInDesc, imgMemDst,
                    {outDesc.width / 2, 0, 0}, outDesc, pitch, copyExtent);

  // Copy second half of imgMemSrcTwo to fourth quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2.get_handle(), {dataInDesc.width / 2, 0, 0},
                    dataInDesc, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                    outDesc, pitch, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst, out.data(), outDesc, pitch);

  q.wait_and_throw();

  sycl::free(imgMemDst, q);
}

void copy_usm_to_image_mem_handle(const syclexp::image_descriptor &dataInDesc,
                                  const syclexp::image_descriptor &outDesc,
                                  const std::vector<float> &dataIn1,
                                  const std::vector<float> &dataIn2,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitchSrc1 = 0;
  size_t pitchSrc2 = 0;
  size_t elements = outDesc.width * outDesc.num_channels;
  void *imgMemSrc1 = sycl::malloc_device<float>(elements, q);
  void *imgMemSrc2 = sycl::malloc_device<float>(elements, q);

  syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn1.data(), imgMemSrc1, dataInDesc, pitchSrc1);
  q.ext_oneapi_copy(dataIn2.data(), imgMemSrc2, dataInDesc, pitchSrc2);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of imgMemSrcOne to first quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1, {0, 0, 0}, dataInDesc, pitchSrc1,
                    imgMemDst.get_handle(), {0, 0, 0}, outDesc, copyExtent);

  // Copy second half of imgMemSrcOne to second quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc1, imgMemDst.get_handle(),
                    {outDesc.width / 4, 0, 0}, outDesc, copyExtent);

  // Copy first half of imgMemSrcTwo to third quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2, {0, 0, 0}, dataInDesc, pitchSrc2,
                    imgMemDst.get_handle(), {outDesc.width / 2, 0, 0}, outDesc,
                    copyExtent);

  // Copy second half of imgMemSrcTwo to fourth quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc2, imgMemDst.get_handle(),
                    {(outDesc.width / 4) * 3, 0, 0}, outDesc, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst.get_handle(), out.data(), outDesc);

  q.wait_and_throw();

  sycl::free(imgMemSrc1, q);
  sycl::free(imgMemSrc2, q);
}

void copy_usm_to_usm(const syclexp::image_descriptor &dataInDesc,
                     const syclexp::image_descriptor &outDesc,
                     const std::vector<float> &dataIn1,
                     const std::vector<float> &dataIn2, sycl::device dev,
                     sycl::queue q, std::vector<float> &out) {

  // Check that output image is double size of input images
  assert(outDesc.width == dataInDesc.width * 2);

  size_t pitchSrc1 = 0;
  size_t pitchSrc2 = 0;
  size_t elementsSrc = outDesc.width * outDesc.num_channels;
  void *imgMemSrc1 = sycl::malloc_device<float>(elementsSrc, q);
  void *imgMemSrc2 = sycl::malloc_device<float>(elementsSrc, q);

  // syclexp::image_mem imgMemDst(outDesc, dev, q.get_context());

  size_t pitchDst = 0;
  size_t elementsDst = outDesc.width * outDesc.num_channels;
  void *imgMemDst = sycl::malloc_device<float>(elementsDst, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn1.data(), imgMemSrc1, dataInDesc, pitchSrc1);
  q.ext_oneapi_copy(dataIn2.data(), imgMemSrc2, dataInDesc, pitchSrc2);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {dataInDesc.width / 2, 1, 1};

  // Copy first half of imgMemSrcOne to first quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1, {0, 0, 0}, dataInDesc, pitchSrc1, imgMemDst,
                    {0, 0, 0}, outDesc, pitchDst, copyExtent);

  // Copy second half of imgMemSrcOne to second quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc1, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc1, imgMemDst, {outDesc.width / 4, 0, 0}, outDesc,
                    pitchDst, copyExtent);

  // Copy first half of imgMemSrcTwo to third quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2, {0, 0, 0}, dataInDesc, pitchSrc2, imgMemDst,
                    {outDesc.width / 2, 0, 0}, outDesc, pitchDst, copyExtent);

  // Copy second half of imgMemSrcTwo to fourth quarter of imgMemDst
  q.ext_oneapi_copy(imgMemSrc2, {dataInDesc.width / 2, 0, 0}, dataInDesc,
                    pitchSrc2, imgMemDst, {(outDesc.width / 4) * 3, 0, 0},
                    outDesc, pitchDst, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst, out.data(), outDesc, pitchDst);

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
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
#else
      break;
#endif
    }
  }
  return validated;
}

template <int channelNum, sycl::image_channel_type channelType,
          syclexp::image_type type = syclexp::image_type::standard>
bool run_copy_test(sycl::device &dev, sycl::queue &q, sycl::range<1> dims) {
  std::vector<float> dataIn1(dims.size() / 2);
  std::vector<float> dataIn2(dims.size() / 2);
  std::vector<float> out(dims.size());

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

  validated = validated && check_test(out, expected);

  copy_image_mem_handle_to_usm(dataInDesc, outDesc, dataIn1, dataIn2, dev, q,
                               out);

  validated = validated && check_test(out, expected);

  copy_usm_to_image_mem_handle(dataInDesc, outDesc, dataIn1, dataIn2, dev, q,
                               out);

  validated = validated && check_test(out, expected);

  copy_usm_to_usm(dataInDesc, outDesc, dataIn1, dataIn2, dev, q, out);

  validated = validated && check_test(out, expected);

  // Perform out of bounds copy checks
  validated =
      validated && image_mem_handle_to_image_mem_handle_out_of_bounds_copy(
                       dataInDesc, outDesc, dataIn1, dev, q);

  validated = validated && image_mem_handle_to_usm_out_of_bounds_copy(
                               dataInDesc, outDesc, dataIn1, dev, q);

  validated = validated && usm_to_image_mem_handle_out_of_bounds_copy(
                               dataInDesc, outDesc, dataIn1, dev, q);

  validated = validated && usm_to_usm_out_of_bounds_copy(dataInDesc, outDesc,
                                                         dataIn1, dev, q);

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

  std::cout << "Tests passed\n";

  return 0;
}
