// REQUIRES: cuda,aspect-ext_oneapi_cubemap

// RUN: %{build} -o %t.out
// RUN: %t.out

#include "../helpers/common.hpp"
#include <iostream>
#include <random>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <type_traits>

namespace syclexp = sycl::ext::oneapi::experimental;

static sycl::device dev;

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

// Helpers and utilities.
struct util {
  template <typename DType, int NChannels, typename KernelName>
  static void run_ndim_test(sycl::queue q, sycl::range<3> globalSize,
                            sycl::range<3> localSize,
                            syclexp::unsampled_image_handle input_0,
                            syclexp::unsampled_image_handle input_1,
                            syclexp::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            sycl::nd_range<3>{globalSize, localSize}, [=](sycl::nd_item<3> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);

              if constexpr (NChannels >= 1) {
                VecType px1 = syclexp::fetch_cubemap<VecType>(
                    input_0, sycl::int2(dim0, dim1), int(dim2));
                VecType px2 = syclexp::fetch_cubemap<VecType>(
                    input_1, sycl::int2(dim0, dim1), int(dim2));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_cubemap<VecType>(output, sycl::int2(dim0, dim1),
                                                int(dim2), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_cubemap<DType>(
                    input_0, sycl::int2(dim0, dim1), int(dim2));
                DType px2 = syclexp::fetch_cubemap<DType>(
                    input_1, sycl::int2(dim0, dim1), int(dim2));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_cubemap<DType>(output, sycl::int2(dim0, dim1),
                                              int(dim2), DType(sum));
              }
            });
      });
    } catch (sycl::exception e) {
      std::cout << "\tKernel submission failed! " << e.what() << std::endl;
      exit(-1);
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
      exit(-1);
    }
  }
};

template <typename DType, int NChannels, sycl::image_channel_type CType,
          typename KernelName>
bool run_test(sycl::range<2> dims, sycl::range<3> localSize,
              unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::queue q(dev);

  // Skip half tests if not supported.
  if constexpr (std::is_same_v<DType, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#ifdef VERBOSE_PRINT
      std::cout << "Test skipped due to lack of device support for fp16\n";
#endif
      return false;
    }
  }

  size_t num_elems = dims.size() * 6;

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> input_1(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  bindless_helpers::fill_rand(input_0, seed);
  bindless_helpers::fill_rand(input_1, seed);
  bindless_helpers::add_host(input_0, input_1, expected);

  try {
    syclexp::image_descriptor desc(dims, NChannels, CType,
                                   syclexp::image_type::cubemap, 1, 6);

    // Extension: allocate memory on device and create the handle.
    syclexp::image_mem img_mem_0(desc, q);
    syclexp::image_mem img_mem_1(desc, q);
    syclexp::image_mem img_mem_2(desc, q);

    auto img_input_0 = syclexp::create_image(img_mem_0, desc, q);
    auto img_input_1 = syclexp::create_image(img_mem_1, desc, q);
    auto img_output = syclexp::create_image(img_mem_2, desc, q);

    // Extension: copy over data to device.
    q.ext_oneapi_copy(input_0.data(), img_mem_0.get_handle(), desc);
    q.ext_oneapi_copy(input_1.data(), img_mem_1.get_handle(), desc);
    q.wait();
    {
      sycl::range<3> globalSize = {dims[0], dims[1], 6};
      util::run_ndim_test<DType, NChannels, KernelName>(
          q, globalSize, localSize, img_input_0, img_input_1, img_output);
      q.wait();
      q.ext_oneapi_copy(img_mem_2.get_handle(), actual.data(), desc);
      q.wait();
    }

    // Cleanup.
    syclexp::destroy_image_handle(img_input_0, q);
    syclexp::destroy_image_handle(img_input_1, q);
    syclexp::destroy_image_handle(img_output, q);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
  }

  // Collect and validate output.
  bool validated = true;
  for (int i = 0; i < num_elems; i++) {
    for (int j = 0; j < NChannels; ++j) {
      bool mismatch = false;
      if (actual[i][j] != expected[i][j]) {
        mismatch = true;
        validated = false;
      }
      if (mismatch) {
#ifdef VERBOSE_PRINT
        std::cout << "\tResult mismatch at [" << i << "][" << j
                  << "] Expected: " << +expected[i][j]
                  << ", Actual: " << +actual[i][j] << std::endl;
#else
        break;
#endif
      }
    }
  }
#ifdef VERBOSE_PRINT
  if (validated) {
    std::cout << "\tTest passed!" << std::endl;
  } else {
    std::cout << "\tTest failed!\n";
  }
#endif

  return !validated;
}

void printTestName(std::string name) {
#ifdef VERBOSE_PRINT
  std::cout << name;
#endif
}

int main() {

  unsigned int seed = 0;
  bool failed = false;

  printTestName("Running cube int\n");
  failed |= run_test<int32_t, 1, sycl::image_channel_type::signed_int32,
                     class int_cube>({32, 32}, {16, 16, 2}, seed);
  printTestName("Running cube int2\n");
  failed |= run_test<int32_t, 2, sycl::image_channel_type::signed_int32,
                     class int2_cube>({128, 128}, {16, 16, 3}, seed);
  printTestName("Running cube int4\n");
  failed |= run_test<int32_t, 4, sycl::image_channel_type::signed_int32,
                     class int4_cube>({64, 64}, {32, 16, 1}, seed);

  printTestName("Running cube unsigned int\n");
  failed |= run_test<uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                     class uint_cube>({15, 15}, {5, 3, 1}, seed);
  printTestName("Running cube unsigned int2\n");
  failed |= run_test<uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                     class uint2_cube>({90, 90}, {10, 9, 3}, seed);
  printTestName("Running cube unsigned int4\n");
  failed |= run_test<uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                     class uint4_cube>({1024, 1024}, {16, 16, 2}, seed);

  printTestName("Running cube short\n");
  failed |= run_test<short, 1, sycl::image_channel_type::signed_int16,
                     class short_cube>({8, 8}, {2, 2, 1}, seed);
  printTestName("Running cube short2\n");
  failed |= run_test<short, 2, sycl::image_channel_type::signed_int16,
                     class short2_cube>({8, 8}, {4, 4, 2}, seed);
  printTestName("Running cube short4\n");
  failed |= run_test<short, 4, sycl::image_channel_type::signed_int16,
                     class short4_cube>({8, 8}, {8, 8, 3}, seed);

  printTestName("Running cube unsigned short\n");
  failed |=
      run_test<unsigned short, 1, sycl::image_channel_type::unsigned_int16,
               class ushort_cube>({75, 75}, {25, 5, 1}, seed);
  printTestName("Running cube unsigned short2\n");
  failed |=
      run_test<unsigned short, 2, sycl::image_channel_type::unsigned_int16,
               class ushort2_cube>({75, 75}, {15, 3, 2}, seed);
  printTestName("Running cube unsigned short4\n");
  failed |=
      run_test<unsigned short, 4, sycl::image_channel_type::unsigned_int16,
               class ushort4_cube>({75, 75}, {5, 25, 3}, seed);

  printTestName("Running cube char\n");
  failed |= run_test<signed char, 1, sycl::image_channel_type::signed_int8,
                     class char_cube>({60, 60}, {10, 6, 1}, seed);
  printTestName("Running cube char2\n");
  failed |= run_test<signed char, 2, sycl::image_channel_type::signed_int8,
                     class char2_cube>({60, 60}, {5, 3, 2}, seed);
  printTestName("Running cube char4\n");
  failed |= run_test<signed char, 4, sycl::image_channel_type::signed_int8,
                     class char4_cube>({60, 60}, {6, 10, 3}, seed);

  printTestName("Running cube unsigned char\n");
  failed |= run_test<unsigned char, 1, sycl::image_channel_type::unsigned_int8,
                     class uchar_cube>({128, 128}, {16, 16, 3}, seed);
  printTestName("Running cube unsigned char2\n");
  failed |= run_test<unsigned char, 2, sycl::image_channel_type::unsigned_int8,
                     class uchar2_cube>({128, 128}, {16, 16, 3}, seed);
  printTestName("Running cube unsigned char4\n");
  failed |= run_test<unsigned char, 4, sycl::image_channel_type::unsigned_int8,
                     class uchar4_cube>({128, 128}, {16, 16, 3}, seed);

  printTestName("Running cube float\n");
  failed |=
      run_test<float, 1, sycl::image_channel_type::fp32, class float_cube>(
          {1024, 1024}, {16, 16, 1}, seed);
  printTestName("Running cube float2\n");
  failed |=
      run_test<float, 2, sycl::image_channel_type::fp32, class float2_cube>(
          {1024, 1024}, {16, 16, 3}, seed);
  printTestName("Running cube float4\n");
  failed |=
      run_test<float, 4, sycl::image_channel_type::fp32, class float4_cube>(
          {1024, 1024}, {16, 16, 2}, seed);

  printTestName("Running cube half\n");
  failed |=
      run_test<sycl::half, 1, sycl::image_channel_type::fp16, class half_cube>(
          {48, 48}, {8, 8, 1}, seed);
  printTestName("Running cube half2\n");
  failed |=
      run_test<sycl::half, 2, sycl::image_channel_type::fp16, class half2_cube>(
          {48, 48}, {8, 8, 3}, seed);
  printTestName("Running cube half4\n");
  failed |=
      run_test<sycl::half, 4, sycl::image_channel_type::fp16, class half4_cube>(
          {48, 48}, {8, 8, 2}, seed);

  if (failed) {
    std::cerr << "An error has occured!\n";
    return 1;
  }

  std::cout << "All tests passed!\n";
  return 0;
}
