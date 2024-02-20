// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "user_types_common.hpp"
#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

template <typename MyType, int NElems, typename OutType,
          sycl::image_channel_order ChannelOrder,
          sycl::image_channel_type ChannelType, typename KernelName>
bool run_test() {

  sycl::device dev;
  sycl::queue q(dev);

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

  // Define and populate array for the output data
  OutType dataOut[256];
  MyType dataIn[256];
  for (int i = 0; i < 256; i++) {
    dataIn[i].set_all(1.f);
  }

  try {

    namespace syclexp = sycl::ext::oneapi::experimental;

    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    syclexp::image_descriptor desc(sycl::range<2>{16, 16}, ChannelOrder,
                                   ChannelType);

    syclexp::image_mem imgMemoryIn1(desc, q);
    syclexp::image_mem imgMemoryIn2(desc, q);
    syclexp::image_mem imgMemoryOut(desc, q);

    syclexp::unsampled_image_handle unsampledImgIn =
        syclexp::create_image(imgMemoryIn1, desc, q);
    syclexp::sampled_image_handle sampledImgIn =
        syclexp::create_image(imgMemoryIn2, samp, desc, q);
    syclexp::unsampled_image_handle imgOut =
        syclexp::create_image(imgMemoryOut, desc, q);

    q.ext_oneapi_copy(dataIn, imgMemoryIn1.get_handle(), desc);
    q.ext_oneapi_copy(dataIn, imgMemoryIn2.get_handle(), desc);
    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<KernelName>(sycl::range{16, 16}, [=](sycl::id<2> id) {
        sycl::int2 coords = sycl::int2(id[0], id[1]);
        sycl::float2 floatCoords =
            sycl::float2(float(id[0]) + 0.5f, float(id[1]) + 0.5f);

        MyType myPixel{};

        // Unsampled read
        myPixel = syclexp::read_image<MyType, OutType>(unsampledImgIn, coords);

        // Sampled read
        myPixel +=
            syclexp::read_image<MyType, OutType>(sampledImgIn, floatCoords);

        syclexp::write_image(imgOut, coords, myPixel);
      });
    });
    q.wait_and_throw();

    q.ext_oneapi_copy(imgMemoryOut.get_handle(), dataOut, desc);
    q.wait_and_throw();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return false;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return false;
  }

  float expected = 2.0f;
  bool validated = true;
  bool mismatch = false;
  for (int i = 0; i < 256; ++i) {
    int j = 0;
    for (; j < NElems; ++j) {
      if (dataOut[i][j] != expected) {
        validated = false;
        mismatch = true;
        break;
      }
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected << ", Actual[" << i
                << "][" << j << "]: " << dataOut[i][j] << std::endl;
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
  printTestName("Running my_half2");
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
