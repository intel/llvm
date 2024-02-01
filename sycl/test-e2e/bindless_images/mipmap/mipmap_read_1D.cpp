// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/sycl.hpp>

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
  constexpr size_t N = 15;
  std::vector<DType> out(N);
  std::vector<DType> expected(N);
  std::vector<VecType> dataIn1(N);
  std::vector<VecType> dataIn2(N / 2);
  std::vector<VecType> copyOut(N / 2);

  for (int i = 0; i < N; i++) {
    // Populate input data (to-be mipmap image layers)
    dataIn1[i] = VecType(i);
    if (i < (N / 2)) {
      dataIn2[i] = VecType(i + 10);
      copyOut[i] = VecType(0);
    }

    // Calculate expected output data
    float norm_coord = ((i + 0.5f) / (float)N);
    int x = norm_coord * (N >> 1);
    expected[i] = dataIn1[i][0] + dataIn2[x][0];
  }

  try {

    size_t width = N;
    unsigned int numLevels = 2;

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::rgba, CType,
        sycl::ext::oneapi::experimental::image_type::mipmap, numLevels);

    // Extension: allocate mipmap memory on device
    sycl::ext::oneapi::experimental::image_mem mipMem(desc, dev, ctxt);

    // Extension: retrieve level 0
    sycl::ext::oneapi::experimental::image_mem_handle imgMem1 =
        mipMem.get_mip_level_mem_handle(0);

    // Extension: copy over data to device at level 0
    q.ext_oneapi_copy(dataIn1.data(), imgMem1, desc);

    // Extension: copy data to device at level 1
    q.ext_oneapi_copy(dataIn2.data(), mipMem.get_mip_level_mem_handle(1),
                      desc.get_mip_level_desc(1));
    q.wait_and_throw();

    // Extension: define a sampler object -- extended mipmap attributes
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::mirrored_repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::nearest, sycl::filtering_mode::nearest, 0.0f,
        (float)numLevels, 8.0f);

    // Extension: create a sampled image handle to represent the mipmap
    sycl::ext::oneapi::experimental::sampled_image_handle mipHandle =
        sycl::ext::oneapi::experimental::create_image(mipMem, samp, desc, dev,
                                                      ctxt);

    sycl::buffer<DType, 1> buf((DType *)out.data(), N);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.template get_access<sycl::access_mode::write>(cgh, N);

      cgh.parallel_for<kernel<DType, CType>>(N, [=](sycl::id<1> id) {
        DType sum = 0;
        float x = float(id[0] + 0.5f) / (float)N;
        // Extension: read mipmap level 0 with anisotropic filtering and level 1
        // with LOD
        VecType px1 = sycl::ext::oneapi::experimental::read_mipmap<VecType>(
            mipHandle, x, 0.0f);
        VecType px2 = sycl::ext::oneapi::experimental::read_mipmap<VecType>(
            mipHandle, x, 1.0f);

        sum = px1[0] + px2[0];
        outAcc[id] = sum;
      });
    });

    q.wait_and_throw();

    // Extension: copy data from device
    q.ext_oneapi_copy(mipMem.get_mip_level_mem_handle(1), copyOut.data(),
                      desc.get_mip_level_desc(1));
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(mipHandle, dev, ctxt);

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
    return 0;
  }

  return 1;
}

int main() {

  int failed = 0;

  failed += runTest<int, sycl::image_channel_type::signed_int32>();

  failed += runTest<uint, sycl::image_channel_type::unsigned_int32>();

  failed += runTest<float, sycl::image_channel_type::fp32>();

  failed += runTest<short, sycl::image_channel_type::signed_int16>();

  failed += runTest<ushort, sycl::image_channel_type::unsigned_int16>();

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
