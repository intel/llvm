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
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q, std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), desc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  // Copy four quarters of square into output image
  q.ext_oneapi_copy(imgMemSrc.get_handle(), {0, 0, 0}, desc,
                    imgMemDst.get_handle(), {0, 0, 0}, desc, copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {desc.width / 2, 0, 0}, desc,
                    imgMemDst.get_handle(), {desc.width / 2, 0, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {0, desc.height / 2, 0}, desc,
                    imgMemDst.get_handle(), {0, desc.height / 2, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc,
                    imgMemDst.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst.get_handle(), out.data(), desc);

  q.wait_and_throw();
}

void copy_image_mem_handle_to_usm(const syclexp::image_descriptor &desc,
                                  const std::vector<float> &dataIn,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());

  // Allocate 2D device USM memory
  size_t pitch = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitch, desc, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), desc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  // Copy four quarters of square into output image
  q.ext_oneapi_copy(imgMemSrc.get_handle(), {0, 0, 0}, desc, imgMemDst,
                    {0, 0, 0}, desc, pitch, copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {desc.width / 2, 0, 0}, desc,
                    imgMemDst, {desc.width / 2, 0, 0}, desc, pitch, copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {0, desc.height / 2, 0}, desc,
                    imgMemDst, {0, desc.height / 2, 0}, desc, pitch,
                    copyExtent);

  q.ext_oneapi_copy(
      imgMemSrc.get_handle(), {desc.width / 2, desc.height / 2, 0}, desc,
      imgMemDst, {desc.width / 2, desc.height / 2, 0}, desc, pitch, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst, out.data(), desc, pitch);

  q.wait_and_throw();

  sycl::free(imgMemDst, q);
}

void copy_usm_to_image_mem_handle(const syclexp::image_descriptor &desc,
                                  const std::vector<float> &dataIn,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {
  // Allocate 2D device USM memory
  size_t pitch = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitch, desc, q);

  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, desc, pitch);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  // Copy four quarters of square into output image
  q.ext_oneapi_copy(imgMemSrc, {0, 0, 0}, desc, pitch, imgMemDst.get_handle(),
                    {0, 0, 0}, desc, copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, 0, 0}, desc, pitch,
                    imgMemDst.get_handle(), {desc.width / 2, 0, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {0, desc.height / 2, 0}, desc, pitch,
                    imgMemDst.get_handle(), {0, desc.height / 2, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                    pitch, imgMemDst.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst.get_handle(), out.data(), desc);

  q.wait_and_throw();

  sycl::free(imgMemSrc, q);
}

void copy_usm_to_usm(const syclexp::image_descriptor &desc,
                     const std::vector<float> &dataIn, sycl::device dev,
                     sycl::queue q, std::vector<float> &out) {
  // Allocate 2D device USM memory
  size_t pitchSrc = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitchSrc, desc, q);

  size_t pitchDst = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitchDst, desc, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, desc, pitchSrc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  // Copy four quarters of square into output image
  q.ext_oneapi_copy(imgMemSrc, {0, 0, 0}, desc, pitchSrc, imgMemDst, {0, 0, 0},
                    desc, pitchDst, copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, 0, 0}, desc, pitchSrc,
                    imgMemDst, {desc.width / 2, 0, 0}, desc, pitchDst,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {0, desc.height / 2, 0}, desc, pitchSrc,
                    imgMemDst, {0, desc.height / 2, 0}, desc, pitchDst,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                    pitchSrc, imgMemDst, {desc.width / 2, desc.height / 2, 0},
                    desc, pitchDst, copyExtent);

  q.wait_and_throw();

  // Copy out data to host
  q.ext_oneapi_copy(imgMemDst, out.data(), desc, pitchDst);

  q.wait_and_throw();

  sycl::free(imgMemSrc, q);
  sycl::free(imgMemDst, q);
}

bool image_mem_handle_to_image_mem_handle_out_of_bounds_copy(
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), desc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(
        imgMemSrc.get_handle(), {desc.width / 2, desc.height / 2, 0}, desc,
        imgMemDst.get_handle(), {desc.width / 2, (desc.height / 2) + 1, 0},
        desc, copyExtent);
  } catch (sycl::exception e) {
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return false;
  }

  return false;
}

bool image_mem_handle_to_usm_out_of_bounds_copy(
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());

  size_t pitch = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitch, desc, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), desc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(imgMemSrc.get_handle(),
                      {desc.width / 2, desc.height / 2, 0}, desc, imgMemDst,
                      {desc.width / 2, (desc.height / 2) + 1, 0}, desc, pitch,
                      copyExtent);
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
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q) {

  size_t pitch = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitch, desc, q);

  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, desc, pitch);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                      pitch, imgMemDst.get_handle(),
                      {desc.width / 2, (desc.height / 2) + 1, 0}, desc,
                      copyExtent);
  } catch (sycl::exception e) {
    sycl::free(imgMemSrc, q);
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
  }

  sycl::free(imgMemSrc, q);

  return false;
}

bool usm_to_usm_out_of_bounds_copy(const syclexp::image_descriptor &desc,
                                   const std::vector<float> &dataIn,
                                   sycl::device dev, sycl::queue q) {

  size_t pitchSrc = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitchSrc, desc, q);

  size_t pitchDst = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitchDst, desc, q);

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc, desc, pitchSrc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                      pitchSrc, imgMemDst,
                      {desc.width / 2, (desc.height / 2) + 1, 0}, desc,
                      pitchDst, copyExtent);
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
bool run_copy_test(sycl::device &dev, sycl::queue &q, sycl::range<2> dims) {
  std::vector<float> dataIn(dims.size());
  std::iota(dataIn.begin(), dataIn.end(), 0);

  std::vector<float> expected(dims.size());
  std::iota(expected.begin(), expected.end(), 0);

  std::vector<float> out(dims.size());

  syclexp::image_descriptor desc =
      syclexp::image_descriptor(dims, channelNum, channelType);

  bool validated = true;

  // Perform copy checks
  copy_image_mem_handle_to_image_mem_handle(desc, dataIn, dev, q, out);

  validated = validated && check_test(out, expected);

  copy_image_mem_handle_to_usm(desc, dataIn, dev, q, out);

  validated = validated && check_test(out, expected);

  copy_usm_to_image_mem_handle(desc, dataIn, dev, q, out);

  validated = validated && check_test(out, expected);

  copy_usm_to_usm(desc, dataIn, dev, q, out);

  validated = validated && check_test(out, expected);

  // Perform out of bounds copy checks
  validated =
      validated && image_mem_handle_to_image_mem_handle_out_of_bounds_copy(
                       desc, dataIn, dev, q);

  validated = validated &&
              image_mem_handle_to_usm_out_of_bounds_copy(desc, dataIn, dev, q);

  validated = validated &&
              usm_to_image_mem_handle_out_of_bounds_copy(desc, dataIn, dev, q);

  validated = validated && usm_to_usm_out_of_bounds_copy(desc, dataIn, dev, q);

  return validated;
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);

  bool validated =
      run_copy_test<1, sycl::image_channel_type::fp32>(dev, q, {12, 12});

  if (!validated) {
    std::cout << "Tests failed\n";
    return 1;
  }

  std::cout << "Tests passed\n";

  return 0;
}
