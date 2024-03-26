// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "user_types_common.hpp"
#include <iostream>
#include <sycl/sycl.hpp>

// Returns true if test case was passed and validated
template <typename MyType, int NElems, typename OutType,
          sycl::image_channel_order ChannelOrder,
          sycl::image_channel_type ChannelType, typename KernelName>
bool run_test() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // skip sycl::half tests if fp16 not supported
  if constexpr (std::is_same_v<typename OutType::element_type, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#ifdef VERBOSE_PRINT
      std::cout
          << "sycl::half test skipped due to lack of device support for fp16\n";
#endif
      return true;
    }
  }

  // declare image data
  size_t width = 16;
  size_t height = 16;
  size_t N = width * height;
  std::vector<OutType> dataOut(N);
  std::vector<OutType> expected(N);
  std::vector<MyType> dataIn1(N);
  std::vector<MyType> dataIn2(N / 4);
  std::vector<MyType> dataIn3(N / 16);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      dataIn1[i + (width * j)].set_all(i + (width * j));
    }
  }
  for (int i = 0; i < (N / 4); i++) {
    dataIn2[i].set_all(i);
  }
  for (int i = 0; i < (N / 16); i++) {
    dataIn3[i].set_all(i);
  }
  // Expected each x and y will repeat twice
  // since mipmap level 1 is half in size
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float normCoordX = ((i + 0.5f) / (float)width);
      int x = normCoordX * (width >> 1);
      float normCoordY = ((j + 0.5f) / (float)height);
      int y = normCoordY * (height >> 1);
      // expected[j + (width * i)].set_all(dataIn2[y + (width / 2 * x)]);
      expected[j + (width * i)][0] = dataIn2[y + (width / 2 * x)].x;
      expected[j + (width * i)][1] = dataIn2[y + (width / 2 * x)].y;
      if constexpr (NElems == 4) {
        expected[j + (width * i)][2] = dataIn2[y + (width / 2 * x)].z;
        expected[j + (width * i)][3] = dataIn2[y + (width / 2 * x)].w;
      }
    }
  }

  try {

    size_t numLevels = 3;

    // Extension: image descriptor -- number of levels
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, ChannelOrder, ChannelType,
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

    sycl::buffer<MyType, 2> buf((MyType *)dataOut.data(),
                                sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.template get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<KernelName>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;
            float fdim1 = float(dim1 + 0.5f) / (float)height;

            // Extension: sample mipmap level 1 with LOD
            MyType pixel =
                sycl::ext::oneapi::experimental::sample_mipmap<MyType, OutType>(
                    mipHandle, sycl::float2(fdim0, fdim1), 1.0f);

            outAcc[sycl::id<2>{dim1, dim0}] = pixel;
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
  bool mismatch = false;

  for (int i = 0; i < N; i++) {
    int j = 0;
    for (; j < NElems; ++j) {
      if (dataOut[i][j] != expected[i][j]) {
        validated = false;
        mismatch = true;
        break;
      }
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i][j]
                << ", Actual[" << i << "][" << j << "]: " << dataOut[i][j]
                << std::endl;
#else
      break;
#endif
    }
  }

  return validated;
}

int main() {

  bool validated = true;

  // User-defined float structs
  printTestName("Running my_float4");
  validated &= run_test<my_float4, 4, sycl::vec<float, 4>,
                        sycl::image_channel_order::rgba,
                        sycl::image_channel_type::fp32, class myfloat_4>();
  printTestName("Running my_float2");
  validated &=
      run_test<my_float2, 2, sycl::vec<float, 2>, sycl::image_channel_order::rg,
               sycl::image_channel_type::fp32, class myfloat_2>();

  // User-defined uint structs
  printTestName("Running my_uint4");
  validated &=
      run_test<my_uint4, 4, sycl::vec<uint32_t, 4>,
               sycl::image_channel_order::rgba,
               sycl::image_channel_type::unsigned_int32, class myuint_4>();
  printTestName("Running my_uint2");
  validated &=
      run_test<my_uint2, 2, sycl::vec<uint32_t, 2>,
               sycl::image_channel_order::rg,
               sycl::image_channel_type::unsigned_int32, class myuint_2>();

  printTestName("Running my_short4");
  validated &=
      run_test<my_ushort4, 4, sycl::vec<uint16_t, 4>,
               sycl::image_channel_order::rgba,
               sycl::image_channel_type::unsigned_int16, class myushort_4>();

  printTestName("Running my_short2");
  validated &=
      run_test<my_ushort2, 2, sycl::vec<uint16_t, 2>,
               sycl::image_channel_order::rg,
               sycl::image_channel_type::unsigned_int16, class myushort_2>();

  printTestName("Running my_char4");
  validated &=
      run_test<my_uchar4, 4, sycl::vec<uint8_t, 4>,
               sycl::image_channel_order::rgba,
               sycl::image_channel_type::unsigned_int8, class myuchar_4>();
  printTestName("Running my_char2");
  validated &=
      run_test<my_uchar2, 2, sycl::vec<uint8_t, 2>,
               sycl::image_channel_order::rg,
               sycl::image_channel_type::unsigned_int8, class myuchar_2>();

  // User-defined sycl::half structs
  printTestName("Running my_half4");
  validated &= run_test<my_half4, 4, sycl::vec<sycl::half, 4>,
                        sycl::image_channel_order::rgba,
                        sycl::image_channel_type::fp16, class myhalf_4>();
  printTestName("Running my_half4");
  validated &= run_test<my_half2, 2, sycl::vec<sycl::half, 2>,
                        sycl::image_channel_order::rg,
                        sycl::image_channel_type::fp16, class myhalf_2>();

  if (validated) {
    std::cout << "All tests passed!" << std::endl;
    return 0;
  } else {
    std::cout << "A test case failed!" << std::endl;
    return 1;
  }
}
