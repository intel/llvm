// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../bindless_helpers.hpp"
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>
#include <type_traits>

static sycl::device dev;

// Uncomment to print additional test information
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

// Helpers and utilities.
struct util {
  // parallel_for 3D.
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 3>>
  static void run_ndim_test(sycl::queue q, sycl::range<3> globalSize,
                            sycl::range<3> localSize,
                            syclexp::unsampled_image_handle input_0,
                            syclexp::unsampled_image_handle input_1,
                            syclexp::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            sycl::nd_range<NDims>{globalSize, localSize},
            [=](sycl::nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);

              if constexpr (NChannels >= 1) {
                VecType px1 = syclexp::fetch_image_array<VecType>(
                    input_0, sycl::int2(dim0, dim1), int(dim2));
                VecType px2 = syclexp::fetch_image_array<VecType>(
                    input_1, sycl::int2(dim0, dim1), int(dim2));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image_array<VecType>(
                    output, sycl::int2(dim0, dim1), int(dim2), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image_array<DType>(
                    input_0, sycl::int2(dim0, dim1), int(dim2));
                DType px2 = syclexp::fetch_image_array<DType>(
                    input_1, sycl::int2(dim0, dim1), int(dim2));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image_array<DType>(
                    output, sycl::int2(dim0, dim1), int(dim2), DType(sum));
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

  // parallel_for 2D.
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 2>>
  static void run_ndim_test(sycl::queue q, sycl::range<2> globalSize,
                            sycl::range<2> localSize,
                            syclexp::unsampled_image_handle input_0,
                            syclexp::unsampled_image_handle input_1,
                            syclexp::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            sycl::nd_range<NDims>{globalSize, localSize},
            [=](sycl::nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              if constexpr (NChannels >= 1) {
                VecType px1 = syclexp::fetch_image_array<VecType>(
                    input_0, int(dim0), int(dim1));
                VecType px2 = syclexp::fetch_image_array<VecType>(
                    input_1, int(dim0), int(dim1));

                auto sum = VecType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image_array<VecType>(output, int(dim0),
                                                    int(dim1), VecType(sum));
              } else {
                DType px1 = syclexp::fetch_image_array<DType>(
                    input_0, int(dim0), int(dim1));
                DType px2 = syclexp::fetch_image_array<DType>(
                    input_1, int(dim0), int(dim1));

                auto sum = DType(
                    bindless_helpers::add_kernel<DType, NChannels>(px1, px2));
                syclexp::write_image_array<DType>(output, int(dim0), int(dim1),
                                                  DType(sum));
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

template <int NDims, typename DType, int NChannels,
          sycl::image_channel_type CType, sycl::image_channel_order COrder,
          typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> localSize,
              unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // skip half tests if not supported.
  if constexpr (std::is_same_v<DType, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#ifdef VERBOSE_PRINT
      std::cout << "Test skipped due to lack of device support for fp16\n";
#endif
      return false;
    }
  }

  size_t num_elems = dims.size();

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> input_1(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  bindless_helpers::fill_rand(input_0, seed);
  bindless_helpers::fill_rand(input_1, seed);
  bindless_helpers::add_host(input_0, input_1, expected);

  try {
    syclexp::image_descriptor desc({dims[0], NDims > 2 ? dims[1] : 0}, COrder,
                                   CType, syclexp::image_type::array, 1,
                                   NDims > 2 ? dims[2] : dims[1]);

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
      sycl::range<NDims> globalSize = dims;
      q.wait();
      util::run_ndim_test<NDims, DType, NChannels, KernelName>(
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

  // collect and validate output.
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

  printTestName("Running 1D int\n");
  failed |= run_test<2, int32_t, 1, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::r, class int_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int\n");
  failed |= run_test<3, int32_t, 1, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::r, class int_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D int2\n");
  failed |= run_test<2, int32_t, 2, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::rg, class int2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int2\n");
  failed |= run_test<3, int32_t, 2, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::rg, class int2_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D int4\n");
  failed |= run_test<2, int32_t, 4, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::rgba, class int4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int4\n");
  failed |= run_test<3, int32_t, 4, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::rgba, class int4_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned int\n");
  failed |= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::r, class uint_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int\n");
  failed |= run_test<3, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::r, class uint_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned int2\n");
  failed |= run_test<2, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::rg, class uint2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int2\n");
  failed |= run_test<3, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::rg, class uint2_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned int4\n");
  failed |= run_test<2, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::rgba, class uint4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int4\n");
  failed |= run_test<3, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                     sycl::image_channel_order::rgba, class uint4_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D short\n");
  failed |= run_test<2, short, 1, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::r, class short_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short\n");
  failed |= run_test<3, short, 1, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::r, class short_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D short2\n");
  failed |= run_test<2, short, 2, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::rg, class short2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short2\n");
  failed |= run_test<3, short, 2, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::rg, class short2_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D short4\n");
  failed |= run_test<2, short, 4, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::rgba, class short4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short4\n");
  failed |= run_test<3, short, 4, sycl::image_channel_type::signed_int16,
                     sycl::image_channel_order::rgba, class short4_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned short\n");
  failed |=
      run_test<2, unsigned short, 1, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::r, class ushort_1d>({2816, 32},
                                                              {32, 32}, seed);
  printTestName("Running 2D unsigned short\n");
  failed |=
      run_test<3, unsigned short, 1, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::r, class ushort_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned short2\n");
  failed |=
      run_test<2, unsigned short, 2, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::rg, class ushort2_1d>({2816, 32},
                                                                {32, 32}, seed);
  printTestName("Running 2D unsigned short2\n");
  failed |=
      run_test<3, unsigned short, 2, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::rg, class ushort2_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned short4\n");
  failed |=
      run_test<2, unsigned short, 4, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::rgba, class ushort4_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned short4\n");
  failed |=
      run_test<3, unsigned short, 4, sycl::image_channel_type::unsigned_int16,
               sycl::image_channel_order::rgba, class ushort4_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D char\n");
  failed |= run_test<2, signed char, 1, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::r, class char_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char\n");
  failed |= run_test<3, signed char, 1, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::r, class char_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D char2\n");
  failed |= run_test<2, signed char, 2, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::rg, class char2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char2\n");
  failed |= run_test<3, signed char, 2, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::rg, class char2_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D char4\n");
  failed |= run_test<2, signed char, 4, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::rgba, class char4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char4\n");
  failed |= run_test<3, signed char, 4, sycl::image_channel_type::signed_int8,
                     sycl::image_channel_order::rgba, class char4_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned char\n");
  failed |=
      run_test<2, unsigned char, 1, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::r, class uchar_1d>({2816, 32},
                                                             {32, 32}, seed);
  printTestName("Running 2D unsigned char\n");
  failed |=
      run_test<3, unsigned char, 1, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::r, class uchar_2d>({48, 128, 32},
                                                             {16, 16, 4}, seed);
  printTestName("Running 1D unsigned char2\n");
  failed |=
      run_test<2, unsigned char, 2, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::rg, class uchar2_1d>({2816, 32},
                                                               {32, 32}, seed);
  printTestName("Running 2D unsigned char2\n");
  failed |=
      run_test<3, unsigned char, 2, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::rg, class uchar2_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned char4\n");
  failed |=
      run_test<2, unsigned char, 4, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::rgba, class uchar4_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned char4\n");
  failed |=
      run_test<3, unsigned char, 4, sycl::image_channel_type::unsigned_int8,
               sycl::image_channel_order::rgba, class uchar4_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D float\n");
  failed |= run_test<2, float, 1, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::r, class float_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float\n");
  failed |= run_test<3, float, 1, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::r, class float_2d>(
      {1024, 832, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D float2\n");
  failed |= run_test<2, float, 2, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::rg, class float2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float2\n");
  failed |= run_test<3, float, 2, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::rg, class float2_2d>(
      {832, 1024, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D float4\n");
  failed |= run_test<2, float, 4, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::rgba, class float4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float4\n");
  failed |= run_test<3, float, 4, sycl::image_channel_type::fp32,
                     sycl::image_channel_order::rgba, class float4_2d>(
      {1024, 1024, 16}, {16, 16, 4}, seed);

  printTestName("Running 1D half\n");
  failed |= run_test<2, sycl::half, 1, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::r, class half_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half\n");
  failed |= run_test<3, sycl::half, 1, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::r, class half_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D half2\n");
  failed |= run_test<2, sycl::half, 2, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::rg, class half2_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half2\n");
  failed |= run_test<3, sycl::half, 2, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::rg, class half2_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D half4\n");
  failed |= run_test<2, sycl::half, 4, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::rgba, class half4_1d>(
      {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half4\n");
  failed |= run_test<3, sycl::half, 4, sycl::image_channel_type::fp16,
                     sycl::image_channel_order::rgba, class half4_2d>(
      {48, 128, 32}, {16, 16, 4}, seed);

  if (failed) {
    std::cerr << "An error has occured!\n";
    return 1;
  }

  std::cout << "All tests passed!\n";
  return 0;
}
