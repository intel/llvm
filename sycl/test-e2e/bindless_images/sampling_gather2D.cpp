// REQUIRES: aspect-ext_oneapi_bindless_images_2d_usm
// REQUIRES: aspect-ext_oneapi_bindless_images_gather

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename type, sycl::image_channel_type channelType>
int test(sycl::queue &q) {

  size_t width = 8;
  size_t height = 4;
  size_t N = width * height;
  // Only using numChannels = 4 is supported
  size_t numChannels = 4;

  std::vector<type> out(N * numChannels);
  std::vector<type> expected(N);
  std::vector<type> dataIn(N);

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      expected[i + (width * j)][0] =
          i + (width * (j + 1)) -
          (sycl::floor(float(j) / float(height - 1))) * width * height;
      expected[i + (width * j)][1] =
          i + 1 - (sycl::floor(float(i) / float(width - 1))) * width +
          (width * (j + 1)) -
          (sycl::floor(float(j) / float(height - 1))) * width * height;
      expected[i + (width * j)][2] =
          i + 1 - (sycl::floor(float(i) / float(width - 1))) * width +
          (width * j);
      expected[i + (width * j)][3] = i + (width * j);

      dataIn[i + (width * j)] = {i + (width * j), i + (width * j),
                                 i + (width * j), i + (width * j)};
    }
  }

  try {
    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    syclexp::image_descriptor desc({width, height}, numChannels, channelType,
                                   syclexp::image_type::gather);

    syclexp::image_mem imgMem(desc, q);
    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(imgMem, samp, desc, q);

    sycl::buffer<type, 3> bufR(out.data(),
                               sycl::range<3>{numChannels, height, width});

    q.submit([&](sycl::handler &cgh) {
      auto outAcc = bufR.template get_access<sycl::access_mode::write>(
          cgh, sycl::range<3>{numChannels, height, width});

      cgh.parallel_for(sycl::nd_range<2>{{width, height}, {width, height}},
                       [=](sycl::nd_item<2> it) {
                         size_t dim0 = it.get_local_id(0);
                         size_t dim1 = it.get_local_id(1);
                         // Normalize coordinates -- +0.5 to look  centre of
                         // pixel
                         float fdim0 = (dim0 + 0.5) / width;
                         float fdim1 = (dim1 + 0.5) / height;

                         for (size_t j = 0; j < numChannels; j++) {
                           type out = syclexp::gather_image<type>(
                               imgHandle, sycl::float2(fdim0, fdim1), j);
                           for (int i = 0; i < 4; i++) {
                             outAcc[sycl::id<3>{j, dim1, dim0}][i] = out[i];
                           }
                         }
                       });
    });

    q.wait_and_throw();

    syclexp::destroy_image_handle(imgHandle, q);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  bool validated = true;
  for (int k = 0; k < numChannels; k++) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < N; i++) {
        bool mismatch = false;
        if (out[i + k * N][j] != expected[i][j]) {
          mismatch = true;
          validated = false;
        }

        if (mismatch) {
#ifdef VERBOSE_PRINT
          std::cout << "Result mismatch! Expected: " << expected[i][0]
                    << ", Actual: " << out[i + k * N][j]
                    << ", channel index: " << k << std::endl;
#else
          break;
#endif
        }
      }
    }
  }
  if (!validated) {
    std::cout << "Test failed!"
              << "type:" << typeid(type).name() << std::endl;
    return 1;
  }
  return 0;
}

int main() {

  sycl::queue _q{};
  int check = 0;

  check += test<sycl::float4, sycl::image_channel_type::fp32>(_q);
  check += test<sycl::int4, sycl::image_channel_type::signed_int32>(_q);
  check += test<sycl::uint4, sycl::image_channel_type::unsigned_int32>(_q);

  return check;
}
