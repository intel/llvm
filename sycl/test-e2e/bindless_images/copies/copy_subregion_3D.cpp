// REQUIRES: aspect-ext_oneapi_bindless_images

// RUN: %{build} -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    const syclexp::image_descriptor &desc, const std::vector<float> &dataIn,
    sycl::device dev, sycl::queue q, std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy host input data to device.
  // Extent to copy.
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, desc.depth / 2};
  sycl::range hostExtent = {desc.width, desc.height, desc.depth};

  // Copy eight quadrants of input data into device image memory.
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

  q.ext_oneapi_copy(dataIn.data(), {0, 0, desc.depth / 2}, hostExtent,
                    imgMemSrc.get_handle(), {0, 0, desc.depth / 2}, desc,
                    copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {desc.width / 2, 0, desc.depth / 2},
                    hostExtent, imgMemSrc.get_handle(),
                    {desc.width / 2, 0, desc.depth / 2}, desc, copyExtent);

  q.ext_oneapi_copy(dataIn.data(), {0, desc.height / 2, desc.depth / 2},
                    hostExtent, imgMemSrc.get_handle(),
                    {0, desc.height / 2, desc.depth / 2}, desc, copyExtent);

  q.ext_oneapi_copy(
      dataIn.data(), {desc.width / 2, desc.height / 2, desc.depth / 2},
      hostExtent, imgMemSrc.get_handle(),
      {desc.width / 2, desc.height / 2, desc.depth / 2}, desc, copyExtent);

  q.wait_and_throw();

  // Copy eight quadrants of square into output image
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

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {0, 0, desc.depth / 2}, desc,
                    imgMemDst.get_handle(), {0, 0, desc.depth / 2}, desc,
                    copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(), {desc.width / 2, 0, desc.depth / 2},
                    desc, imgMemDst.get_handle(),
                    {desc.width / 2, 0, desc.depth / 2}, desc, copyExtent);

  q.ext_oneapi_copy(imgMemSrc.get_handle(),
                    {0, desc.height / 2, desc.depth / 2}, desc,
                    imgMemDst.get_handle(),
                    {0, desc.height / 2, desc.depth / 2}, desc, copyExtent);

  q.ext_oneapi_copy(
      imgMemSrc.get_handle(), {desc.width / 2, desc.height / 2, desc.depth / 2},
      desc, imgMemDst.get_handle(),
      {desc.width / 2, desc.height / 2, desc.depth / 2}, desc, copyExtent);

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

  q.ext_oneapi_copy(imgMemDst.get_handle(), {0, 0, desc.depth / 2}, desc,
                    out.data(), {0, 0, desc.depth / 2}, hostExtent, copyExtent);

  q.ext_oneapi_copy(imgMemDst.get_handle(), {desc.width / 2, 0, desc.depth / 2},
                    desc, out.data(), {desc.width / 2, 0, desc.depth / 2},
                    hostExtent, copyExtent);

  q.ext_oneapi_copy(
      imgMemDst.get_handle(), {0, desc.height / 2, desc.depth / 2}, desc,
      out.data(), {0, desc.height / 2, desc.depth / 2}, hostExtent, copyExtent);

  q.ext_oneapi_copy(
      imgMemDst.get_handle(), {desc.width / 2, desc.height / 2, desc.depth / 2},
      desc, out.data(), {desc.width / 2, desc.height / 2, desc.depth / 2},
      hostExtent, copyExtent);

  q.wait_and_throw();
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

bool out_of_bounds_copy(const syclexp::image_descriptor &desc,
                        const std::vector<float> &dataIn, sycl::device dev,
                        sycl::queue q) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  // Copy input data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemSrc.get_handle(), desc);

  q.wait_and_throw();

  // Extent to copy
  sycl::range copyExtent = {desc.width / 2, desc.height / 2, desc.depth / 2};

  try {
    // Perform out of bound copy!
    q.ext_oneapi_copy(imgMemSrc.get_handle(),
                      {desc.width / 2, desc.height / 2, (desc.depth / 2) + 1},
                      desc, imgMemDst.get_handle(),
                      {desc.width / 2, desc.height / 2, desc.depth / 2}, desc,
                      copyExtent);
  } catch (sycl::exception e) {
    return true;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return false;
  }

  return false;
}

template <int channelNum, sycl::image_channel_type channelType,
          syclexp::image_type type = syclexp::image_type::standard>
bool run_copy_test(sycl::device &dev, sycl::queue &q, sycl::range<3> dims) {
  std::vector<float> dataIn(dims.size());
  std::iota(dataIn.begin(), dataIn.end(), 0);

  std::vector<float> expected(dims.size());
  std::iota(expected.begin(), expected.end(), 0);

  std::vector<float> out(dims.size());

  syclexp::image_descriptor desc =
      syclexp::image_descriptor(dims, channelNum, channelType);

  // Perform copy
  copy_image_mem_handle_to_image_mem_handle(desc, dataIn, dev, q, out);

  bool copyValidated = check_test(out, expected);

  bool exceptionValidated = out_of_bounds_copy(desc, dataIn, dev, q);

  return copyValidated && exceptionValidated;
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);

  bool validated =
      run_copy_test<1, sycl::image_channel_type::fp32>(dev, q, {12, 12, 12});

  if (!validated) {
    std::cout << "Tests failed\n";
    return 1;
  }

  return 0;
}
