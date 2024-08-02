// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

template <typename DType, sycl::image_channel_type CType> class kernel;

template <typename DType, sycl::image_channel_type CType> bool runTest() {
  using VecType = sycl::vec<DType, 4>;

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // skip half tests if not supported
  if constexpr (std::is_same_v<DType, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#ifdef VERBOSE_PRINT
      std::cout << "Test skipped due to lack of device support for fp16\n";
#endif
      return false;
    }
  }

  // declare image data
  size_t width = 16;
  size_t height = 16;
  size_t N = width * height;
  std::vector<DType> out(N);
  std::vector<DType> expected(N);
  std::vector<VecType> dataIn1(N);
  std::vector<VecType> dataIn2(N / 4);
  std::vector<VecType> dataIn3(N / 16);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      dataIn1[i + (width * j)] = {i + (width * j), 0, 0, 0};
    }
  }
  for (int i = 0; i < (N / 4); i++) {
    dataIn2[i] = VecType(i);
  }
  for (int i = 0; i < (N / 16); i++) {
    dataIn3[i] = VecType(i);
  }
  // Expected each x and y will repeat twice
  // since mipmap level 1 is half in size
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float norm_coord_x = ((i + 0.5f) / (float)width);
      int x = norm_coord_x * (width >> 1);
      float norm_coord_y = ((j + 0.5f) / (float)height);
      int y = norm_coord_y * (height >> 1);
      expected[j + (width * i)] = dataIn2[y + (width / 2 * x)][0];
    }
  }

  try {

    size_t numLevels = 3;

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 4, CType,
        sycl::ext::oneapi::experimental::image_type::mipmap, numLevels);

    // Extension: define a sampler object -- extended mipmap attributes
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest, sycl::filtering_mode::nearest, 0.0f,
        (float)numLevels, 8.0f);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mipMem(desc, q);

    // Extension: copy data to device at all levels -- copy func handles desc
    // sizing
    q.ext_oneapi_copy(dataIn1.data(), mipMem.get_mip_level_mem_handle(0),
                      desc.get_mip_level_desc(0));
    q.ext_oneapi_copy(dataIn1.data(), mipMem.get_mip_level_mem_handle(1),
                      desc.get_mip_level_desc(1));
    q.ext_oneapi_copy(dataIn3.data(), mipMem.get_mip_level_mem_handle(2),
                      desc.get_mip_level_desc(2));
    q.wait_and_throw();

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(mipMem, samp, desc, q);

    sycl::buffer<DType, 2> buf((DType *)out.data(),
                               sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.template get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<kernel<DType, CType>>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;

            // Extension: sample mipmap level 1 with LOD
            VecType px2 =
                sycl::ext::oneapi::experimental::sample_mipmap<VecType>(
                    mipHandle, sycl::float2(fdim0, fdim1), 1.0f);

            outAcc[sycl::id<2>{dim1, dim0}] = px2[0];
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(mipHandle, q);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    std::cout << "Test failed!" << std::endl;
    exit(1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    std::cout << "Test failed!" << std::endl;
    exit(2);
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
    return false;
  }

  return true;
}

int main() {

  int failed = 0;

  failed += runTest<int, sycl::image_channel_type::signed_int32>();

  failed += runTest<unsigned int, sycl::image_channel_type::unsigned_int32>();

  failed += runTest<float, sycl::image_channel_type::fp32>();

  failed += runTest<short, sycl::image_channel_type::signed_int16>();

  failed += runTest<unsigned short, sycl::image_channel_type::unsigned_int16>();

  failed += runTest<char, sycl::image_channel_type::signed_int8>();

  failed += runTest<unsigned char, sycl::image_channel_type::unsigned_int8>();

  failed += runTest<sycl::half, sycl::image_channel_type::fp16>();

  if (failed) {
    std::cout << "Test failed!" << std::endl;
  } else {
    std::cout << "Test passed!" << std::endl;
  }

  return failed;
}
