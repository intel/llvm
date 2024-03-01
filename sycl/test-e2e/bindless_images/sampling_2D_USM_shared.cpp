// REQUIRES: linux
// REQUIRES: cuda
// REQUIRES: aspect-ext_oneapi_bindless_images_shared_usm

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  size_t width = 5;
  size_t height = 6;
  size_t N = width * height;
  size_t widthInBytes = width * sizeof(float);
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn(N);

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = i + (width * j);
      dataIn[i + (width * j)] = i + (width * j);
    }
  }

  try {
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, sycl::image_channel_order::r,
        sycl::image_channel_type::fp32);

    auto devicePitchAlign = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_row_pitch_align>();
    auto deviceMaxPitch =
        dev.get_info<sycl::ext::oneapi::experimental::info::device::
                         max_image_linear_row_pitch>();

    // Pitch requirements:
    //  - pitch % devicePitchAlign == 0
    //  - pitch >= widthInBytes
    //  - pitch <= deviceMaxPitch
    size_t pitch = devicePitchAlign *
                   std::ceil(float(widthInBytes) / float(devicePitchAlign));
    assert(pitch <= deviceMaxPitch);

    // Shared USM allocation
    auto imgMem = sycl::aligned_alloc_shared(devicePitchAlign, (pitch * height),
                                             dev, ctxt);

    if (imgMem == nullptr) {
      std::cerr << "Error allocating images!" << std::endl;
      return 1;
    }

    // Copy to shared USM and incorporate pitch
    for (size_t i = 0; i < height; i++) {
      memcpy(static_cast<float *>(imgMem) + (i * pitch / sizeof(float)),
             dataIn.data() + (i * width), widthInBytes);
    }

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem, pitch, samp, desc,
                                                      dev, ctxt);

    sycl::buffer<float, 2> buf((float *)out.data(),
                               sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;

            // Extension: sample image data from handle
            float px = sycl::ext::oneapi::experimental::sample_image<float>(
                imgHandle, sycl::float2(fdim0, fdim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px;
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle, dev, ctxt);
    sycl::free(imgMem, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < N; i++) {
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
  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cout << "Test failed!" << std::endl;
  return 3;
}
