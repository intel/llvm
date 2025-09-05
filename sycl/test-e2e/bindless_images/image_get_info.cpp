// REQUIRES: aspect-ext_oneapi_bindless_images

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Image channels queries not working correctly on HIP.

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

void printString(std::string name) {
#ifdef VERBOSE_PRINT
  std::cout << name;
#endif
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  size_t height = 13;
  size_t width = 7;
  size_t depth = 11;

  bool validated = true;

  try {
    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, 1, sycl::image_channel_type::signed_int32);

    // Extension: returns the device pointer to the allocated memory
    // Input images memory
    sycl::ext::oneapi::experimental::image_mem imgMem(desc, dev, ctxt);

    auto rangeMem = imgMem.get_range();
    auto range = sycl::ext::oneapi::experimental::get_image_range(
        imgMem.get_handle(), dev, ctxt);
    if (rangeMem != range) {
      printString("handle and mem object disagree on image dimensions!\n");
      validated = false;
    }
    if (range[0] == width) {
      printString("width is correct!\n");
    } else {
      printString("width is NOT correct!\n");
      validated = false;
    }
    if (range[1] == height) {
      printString("height is correct!\n");
    } else {
      printString("height is NOT correct!\n");
      validated = false;
    }
    if (range[2] == depth) {
      printString("depth is correct!\n");
    } else {
      printString("depth is NOT correct!\n");
      validated = false;
    }

    auto type = imgMem.get_type();
    if (type == sycl::ext::oneapi::experimental::image_type::standard) {
      printString("image type is correct!\n");
    } else {
      printString("image type is NOT correct!\n");
      validated = false;
    }

    auto ctypeMem = imgMem.get_channel_type();
    auto ctype = sycl::ext::oneapi::experimental::get_image_channel_type(
        imgMem.get_handle(), dev, ctxt);
    if (ctypeMem != ctype) {
      printString("handle and mem object disagree on image channel type!\n");
      validated = false;
    }
    if (ctype == sycl::image_channel_type::signed_int32) {
      printString("channel type is correct!\n");
    } else {
      printString("channel type is NOT correct!\n");
      validated = false;
    }

    auto numchannelsMem = imgMem.get_num_channels();
    auto numchannels = sycl::ext::oneapi::experimental::get_image_num_channels(
        imgMem.get_handle(), dev, ctxt);
    if (numchannelsMem != numchannels) {
      printString("handle and mem object disagree on number of channels!\n");
      validated = false;
    }
    if (numchannels == 1) {
      printString("num channels is correct!\n");
    } else {
      printString("num channels is NOT correct!\n");
      validated = false;
    }

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor mipDesc(
        {width, height}, 4, sycl::image_channel_type::signed_int32,
        sycl::ext::oneapi::experimental::image_type::mipmap, 3);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mipMem(mipDesc, q);

    auto numChannelsMipMem = mipMem.get_num_channels();
    auto numChannelsMip =
        sycl::ext::oneapi::experimental::get_image_num_channels(
            mipMem.get_handle(), dev, ctxt);

    if (numChannelsMipMem != numChannelsMip) {
      printString(
          "mipmap handle and mem object disagree on number of channels!\n");
      validated = false;
    }
    if (numChannelsMip == 4) {
      printString("mipmap num channels is correct!\n");
    } else {
      printString("mipmap num channels is NOT correct!\n");
      validated = false;
    }

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  if (validated) {
    std::cout << "Test Passed!\n";
    return 0;
  }

  std::cout << "Test Failed!" << std::endl;
  return 3;
}
