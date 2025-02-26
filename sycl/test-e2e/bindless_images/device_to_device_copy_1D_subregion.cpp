// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    const syclexp::image_descriptor &dataInDesc,
    const syclexp::image_descriptor &outDesc,
    const std::vector<float> &dataIn1, const std::vector<float> &dataIn2,
    sycl::device dev, sycl::queue q, std::vector<float> &out) {

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

bool out_of_bounds_copy(const syclexp::image_descriptor &dataInDesc,
                        const syclexp::image_descriptor &outDesc,
                        const std::vector<float> &dataIn, sycl::device dev,
                        sycl::queue q) {

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

  // Perform copy
  copy_image_mem_handle_to_image_mem_handle(dataInDesc, outDesc, dataIn1,
                                            dataIn2, dev, q, out);

  bool copyValidated = check_test(out, expected);

  bool exceptionValidated =
      out_of_bounds_copy(dataInDesc, outDesc, dataIn1, dev, q);

  return copyValidated && exceptionValidated;
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
