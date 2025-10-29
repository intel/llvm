// REQUIRES: aspect-ext_oneapi_bindless_images
// UNSUPPORTED: arch-intel_gpu_bmg_g21 || gpu-intel-dg2 || arch-intel_gpu_lnl_m
// UNSUPPORTED-INTENDED: sporadic failure in CI
//                       https://github.com/intel/llvm/issues/20006
// XFAIL: linux && arch-intel_gpu_acm_g10 && level_zero_v2_adapter
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20004
// XFAIL: hip
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/19957

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q, std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};
  sycl::range hostExtent = {desc.width, desc.height, 0};

  // Copy four quarters of input data into device image memory.
  q.ext_oneapi_copy(dataIn.data(), {0, 0, 0}, hostExtent,
                    imgMemSrc.get_handle(), {0, 0, 0}, desc, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, 0, 0}, hostExtent,
                    imgMemSrc.get_handle(), {desc.width / 2, 0, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {0, desc.height / 2, 0}, hostExtent,
                    imgMemSrc.get_handle(), {0, desc.height / 2, 0}, desc,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, desc.height / 2, 0},
                    hostExtent, imgMemSrc.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device, using four sub-region copies.
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

  // Copy device data back to host.
  // Copy four quarters of device imgMemDst data to host out.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, 0, 0}, desc, out.data(),
                    {0, 0, 0}, hostExtent, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(), {desc.width / 2, 0, 0}, desc,
                    out.data(), {desc.width / 2, 0, 0}, hostExtent, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, desc.height / 2, 0}, desc,
                    out.data(), {0, desc.height / 2, 0}, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(
      imgMemDst.get_handle(), {desc.width / 2, desc.height / 2, 0}, desc,
      out.data(), {desc.width / 2, desc.height / 2, 0}, hostExtent, copyExtent);

  q.wait_and_throw();
}

void copy_image_mem_handle_to_usm(const syclexp::image_descriptor &desc,
                                  const std::vector<float> &dataIn,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());

  // Allocate 2D device USM memory.
  size_t pitch = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitch, desc, q);

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};

  // Copy four quarters of input data into device image memory.
  q.ext_oneapi_copy(dataIn.data(), {0, 0, 0}, {desc.width, desc.height, 0},
                    imgMemSrc.get_handle(), {0, 0, 0}, desc, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, 0, 0},
                    {desc.width, desc.height, 0}, imgMemSrc.get_handle(),
                    {desc.width / 2, 0, 0}, desc, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {0, desc.height / 2, 0},
                    {desc.width, desc.height, 0}, imgMemSrc.get_handle(),
                    {0, desc.height / 2, 0}, desc, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, desc.height / 2, 0},
                    {desc.width, desc.height, 0}, imgMemSrc.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device, using four sub-region copies.
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

  // Copy device data back to host.
  // Copy four quarters of device imgMemDst data to host out.
  q.ext_oneapi_copy(imgMemDst, {0, 0, 0}, out.data(), {0, 0, 0}, desc, pitch,
                    {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst, {desc.width / 2, 0, 0}, out.data(),
                    {desc.width / 2, 0, 0}, desc, pitch,
                    {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst, {0, desc.height / 2, 0}, out.data(),
                    {0, desc.height / 2, 0}, desc, pitch,
                    {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst, {desc.width / 2, desc.height / 2, 0}, out.data(),
                    {desc.width / 2, desc.height / 2, 0}, desc, pitch,
                    {desc.width, desc.height, 0}, copyExtent);

  q.wait_and_throw();

  sycl::free(imgMemDst, q);
}

void copy_usm_to_image_mem_handle(const syclexp::image_descriptor &desc,
                                  const std::vector<float> &dataIn,
                                  sycl::device dev, sycl::queue q,
                                  std::vector<float> &out) {
  // Allocate 2D device USM memory.
  size_t pitch = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitch, desc, q);

  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};
  sycl::range hostExtent = {desc.width, desc.height, 0};

  // Copy four quarters of input data into device image memory.
  q.ext_oneapi_copy(dataIn.data(), {0, 0, 0}, imgMemSrc, {0, 0, 0}, desc, pitch,
                    hostExtent, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, 0, 0}, imgMemSrc,
                    {desc.width / 2, 0, 0}, desc, pitch, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {0, desc.height / 2, 0}, imgMemSrc,
                    {0, desc.height / 2, 0}, desc, pitch, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, desc.height / 2, 0},
                    imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                    pitch, hostExtent, copyExtent);

  q.wait_and_throw();

  // Copy data from device to device, using four sub-region copies.
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

  // Copy device data back to host.
  // Copy four quarters of device imgMemDst data to host out.
  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, 0, 0}, desc, out.data(),
                    {0, 0, 0}, {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(), {desc.width / 2, 0, 0}, desc,
                    out.data(), {desc.width / 2, 0, 0},
                    {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, desc.height / 2, 0}, desc,
                    out.data(), {0, desc.height / 2, 0},
                    {desc.width, desc.height, 0}, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(),
                    {desc.width / 2, desc.height / 2, 0}, desc, out.data(),
                    {desc.width / 2, desc.height / 2, 0},
                    {desc.width, desc.height, 0}, copyExtent);

  q.wait_and_throw();

  sycl::free(imgMemSrc, q);
}

void copy_usm_to_usm(const syclexp::image_descriptor &desc,
                     const std::vector<float> &dataIn, sycl::device dev,
                     sycl::queue q, std::vector<float> &out) {
  // Allocate 2D device USM memory.
  size_t pitchSrc = 0;
  void *imgMemSrc = syclexp::pitched_alloc_device(&pitchSrc, desc, q);

  size_t pitchDst = 0;
  void *imgMemDst = syclexp::pitched_alloc_device(&pitchDst, desc, q);

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, 1};
  sycl::range hostExtent = {desc.width, desc.height, 0};

  q.ext_oneapi_copy(dataIn.data(), {0, 0, 0}, imgMemSrc, {0, 0, 0}, desc,
                    pitchSrc, hostExtent, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, 0, 0}, imgMemSrc,
                    {desc.width / 2, 0, 0}, desc, pitchSrc, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {0, desc.height / 2, 0}, imgMemSrc,
                    {0, desc.height / 2, 0}, desc, pitchSrc, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, desc.height / 2, 0},
                    imgMemSrc, {desc.width / 2, desc.height / 2, 0}, desc,
                    pitchSrc, hostExtent, copyExtent);

  q.wait_and_throw();

  // Copy four quarters of square into output image.
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

  // Copy device data back to host.
  // Copy four quarters of device imgMemDst data to host out.
  q.ext_oneapi_copy(imgMemDst, {0, 0, 0}, out.data(), {0, 0, 0}, desc, pitchDst,
                    hostExtent, copyExtent);

  q.ext_oneapi_copy(imgMemDst, {desc.width / 2, 0, 0}, out.data(),
                    {desc.width / 2, 0, 0}, desc, pitchDst, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(imgMemDst, {0, desc.height / 2, 0}, out.data(),
                    {0, desc.height / 2, 0}, desc, pitchDst, hostExtent,
                    copyExtent);

  q.ext_oneapi_copy(imgMemDst, {desc.width / 2, desc.height / 2, 0}, out.data(),
                    {desc.width / 2, desc.height / 2, 0}, desc, pitchDst,
                    hostExtent, copyExtent);

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

template <int channelNum, sycl::image_channel_type channelType,
          syclexp::image_type type = syclexp::image_type::standard>
bool run_copy_test(sycl::device &dev, sycl::queue &q, sycl::range<2> dims) {
  std::vector<float> dataIn(dims.size());
  std::iota(dataIn.begin(), dataIn.end(), 0);

  std::vector<float> expected(dims.size());
  std::iota(expected.begin(), expected.end(), 0);

  std::vector<float> out(dims.size(), 0);

  syclexp::image_descriptor desc =
      syclexp::image_descriptor(dims, channelNum, channelType);

  bool validated = true;

  // Perform copy checks
  copy_image_mem_handle_to_image_mem_handle(desc, dataIn, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_image_mem_handle_to_image_mem_handle test failed"
              << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_image_mem_handle_to_usm(desc, dataIn, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_image_mem_handle_to_usm test failed" << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_usm_to_image_mem_handle(desc, dataIn, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_usm_to_image_mem_handle test failed" << std::endl;
    validated = false;
  }

  std::fill(out.begin(), out.end(), 0);

  copy_usm_to_usm(desc, dataIn, dev, q, out);
  if (!check_test(out, expected)) {
    std::cout << "copy_usm_to_usm test failed" << std::endl;
    validated = false;
  }

  // Perform out of bounds copy checks
  if (!image_mem_handle_to_image_mem_handle_out_of_bounds_copy(desc, dataIn,
                                                               dev, q)) {
    std::cout
        << "image_mem_handle_to_image_mem_handle_out_of_bounds_copy test failed"
        << std::endl;
    validated = false;
  }

  if (!image_mem_handle_to_usm_out_of_bounds_copy(desc, dataIn, dev, q)) {
    std::cout << "image_mem_handle_to_usm_out_of_bounds_copy test failed"
              << std::endl;
    validated = false;
  }

  if (!usm_to_image_mem_handle_out_of_bounds_copy(desc, dataIn, dev, q)) {
    std::cout << "usm_to_image_mem_handle_out_of_bounds_copy test failed"
              << std::endl;
    validated = false;
  }

  if (!usm_to_usm_out_of_bounds_copy(desc, dataIn, dev, q)) {
    std::cout << "usm_to_usm_out_of_bounds_copy test failed" << std::endl;
    validated = false;
  }

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

  return 0;
}
