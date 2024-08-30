// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "../helpers/common.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static sycl::device dev;

// Uncomment to print additional test information
// #define VERBOSE_PRINT

// Helpers and utilities.
struct util {
  // parallel_for 3D: 2D image + 1D array index.
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 3>>
  static void
  run_ndim_test(sycl::queue q, sycl::range<NDims> globalSize,
                sycl::range<NDims> localSize,
                syclexp::sampled_image_handle input,
                sycl::buffer<sycl::vec<DType, NChannels>, NDims> output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](sycl::handler &cgh) {
        auto acc_output = output.template get_access<sycl::access_mode::write>(
            cgh, bindless_helpers::reverse_dims(globalSize));

        cgh.parallel_for<KernelName>(
            sycl::nd_range<NDims>{globalSize, localSize},
            [=](sycl::nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);

              VecType px1 = syclexp::fetch_image_array<VecType>(
                  input, sycl::int2(dim0, dim1), int(dim2));

              acc_output[sycl::id<3>{dim2, dim1, dim0}] = px1;
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

  // parallel_for 2D: 1D image + 1D array index.
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 2>>
  static void
  run_ndim_test(sycl::queue q, sycl::range<NDims> globalSize,
                sycl::range<NDims> localSize,
                syclexp::sampled_image_handle input,
                sycl::buffer<sycl::vec<DType, NChannels>, NDims> &output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](sycl::handler &cgh) {
        auto acc_output = output.template get_access<sycl::access_mode::write>(
            cgh, bindless_helpers::reverse_dims(globalSize));

        cgh.parallel_for<KernelName>(
            sycl::nd_range<NDims>{globalSize, localSize},
            [=](sycl::nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              VecType px1 = syclexp::fetch_image_array<VecType>(
                  input, int(dim0), int(dim1));

              acc_output[sycl::id<NDims>{dim1, dim0}] = px1;
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
          sycl::image_channel_type CType, typename KernelName>
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
  auto image_array_dims = bindless_helpers::ImageArrayDims<NDims>(dims);

  std::vector<VecType> input(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  bindless_helpers::fill_rand(input, seed);
  expected = input;

  try {
    syclexp::image_descriptor desc(image_array_dims.array_dims, NChannels,
                                   CType, syclexp::image_type::array, 1,
                                   image_array_dims.array_count);

    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::filtering_mode::nearest);

    // Extension: allocate memory on device and create the handle.
    syclexp::image_mem img_mem(desc, q);

    auto img_input = syclexp::create_image(img_mem, samp, desc, q);

    // Extension: copy over data to device.
    q.ext_oneapi_copy(input.data(), img_mem.get_handle(), desc);
    q.wait();

    {
      sycl::buffer<VecType, NDims> buf_output(
          actual.data(), bindless_helpers::reverse_dims(dims));
      util::run_ndim_test<NDims, DType, NChannels, KernelName>(
          q, dims, localSize, img_input, buf_output);
      q.wait();
    }

    // Cleanup.
    syclexp::destroy_image_handle(img_input, q);
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

  failed |= run_test<2, int32_t, 1, sycl::image_channel_type::signed_int32,
                     class int_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int\n");
  failed |= run_test<3, int32_t, 1, sycl::image_channel_type::signed_int32,
                     class int_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D int2\n");
  failed |= run_test<2, int32_t, 2, sycl::image_channel_type::signed_int32,
                     class int2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int2\n");
  failed |= run_test<3, int32_t, 2, sycl::image_channel_type::signed_int32,
                     class int2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D int4\n");
  failed |= run_test<2, int32_t, 4, sycl::image_channel_type::signed_int32,
                     class int4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D int4\n");
  failed |= run_test<3, int32_t, 4, sycl::image_channel_type::signed_int32,
                     class int4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned int\n");
  failed |= run_test<2, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                     class uint_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int\n");
  failed |= run_test<3, uint32_t, 1, sycl::image_channel_type::unsigned_int32,
                     class uint_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned int2\n");
  failed |= run_test<2, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                     class uint2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int2\n");
  failed |= run_test<3, uint32_t, 2, sycl::image_channel_type::unsigned_int32,
                     class uint2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned int4\n");
  failed |= run_test<2, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                     class uint4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned int4\n");
  failed |= run_test<3, uint32_t, 4, sycl::image_channel_type::unsigned_int32,
                     class uint4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D short\n");
  failed |= run_test<2, short, 1, sycl::image_channel_type::signed_int16,
                     class short_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short\n");
  failed |= run_test<3, short, 1, sycl::image_channel_type::signed_int16,
                     class short_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D short2\n");
  failed |= run_test<2, short, 2, sycl::image_channel_type::signed_int16,
                     class short2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short2\n");
  failed |= run_test<3, short, 2, sycl::image_channel_type::signed_int16,
                     class short2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D short4\n");
  failed |= run_test<2, short, 4, sycl::image_channel_type::signed_int16,
                     class short4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D short4\n");
  failed |= run_test<3, short, 4, sycl::image_channel_type::signed_int16,
                     class short4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned short\n");
  failed |=
      run_test<2, unsigned short, 1, sycl::image_channel_type::unsigned_int16,
               class ushort_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned short\n");
  failed |=
      run_test<3, unsigned short, 1, sycl::image_channel_type::unsigned_int16,
               class ushort_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned short2\n");
  failed |=
      run_test<2, unsigned short, 2, sycl::image_channel_type::unsigned_int16,
               class ushort2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned short2\n");
  failed |=
      run_test<3, unsigned short, 2, sycl::image_channel_type::unsigned_int16,
               class ushort2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned short4\n");
  failed |=
      run_test<2, unsigned short, 4, sycl::image_channel_type::unsigned_int16,
               class ushort4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned short4\n");
  failed |=
      run_test<3, unsigned short, 4, sycl::image_channel_type::unsigned_int16,
               class ushort4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D char\n");
  failed |= run_test<2, signed char, 1, sycl::image_channel_type::signed_int8,
                     class char_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char\n");
  failed |= run_test<3, signed char, 1, sycl::image_channel_type::signed_int8,
                     class char_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D char2\n");
  failed |= run_test<2, signed char, 2, sycl::image_channel_type::signed_int8,
                     class char2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char2\n");
  failed |= run_test<3, signed char, 2, sycl::image_channel_type::signed_int8,
                     class char2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D char4\n");
  failed |= run_test<2, signed char, 4, sycl::image_channel_type::signed_int8,
                     class char4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D char4\n");
  failed |= run_test<3, signed char, 4, sycl::image_channel_type::signed_int8,
                     class char4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D unsigned char\n");
  failed |=
      run_test<2, unsigned char, 1, sycl::image_channel_type::unsigned_int8,
               class uchar_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned char\n");
  failed |=
      run_test<3, unsigned char, 1, sycl::image_channel_type::unsigned_int8,
               class uchar_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned char2\n");
  failed |=
      run_test<2, unsigned char, 2, sycl::image_channel_type::unsigned_int8,
               class uchar2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned char2\n");
  failed |=
      run_test<3, unsigned char, 2, sycl::image_channel_type::unsigned_int8,
               class uchar2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D unsigned char4\n");
  failed |=
      run_test<2, unsigned char, 4, sycl::image_channel_type::unsigned_int8,
               class uchar4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D unsigned char4\n");
  failed |=
      run_test<3, unsigned char, 4, sycl::image_channel_type::unsigned_int8,
               class uchar4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  printTestName("Running 1D float\n");
  failed |=
      run_test<2, float, 1, sycl::image_channel_type::fp32, class float_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float\n");
  failed |=
      run_test<3, float, 1, sycl::image_channel_type::fp32, class float_2d>(
          {1024, 832, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D float2\n");
  failed |=
      run_test<2, float, 2, sycl::image_channel_type::fp32, class float2_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float2\n");
  failed |=
      run_test<3, float, 2, sycl::image_channel_type::fp32, class float2_2d>(
          {832, 1024, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D float4\n");
  failed |=
      run_test<2, float, 4, sycl::image_channel_type::fp32, class float4_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D float4\n");
  failed |=
      run_test<3, float, 4, sycl::image_channel_type::fp32, class float4_2d>(
          {1024, 1024, 16}, {16, 16, 4}, seed);

  printTestName("Running 1D half\n");
  failed |=
      run_test<2, sycl::half, 1, sycl::image_channel_type::fp16, class half_1d>(
          {2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half\n");
  failed |=
      run_test<3, sycl::half, 1, sycl::image_channel_type::fp16, class half_2d>(
          {48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D half2\n");
  failed |= run_test<2, sycl::half, 2, sycl::image_channel_type::fp16,
                     class half2_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half2\n");
  failed |= run_test<3, sycl::half, 2, sycl::image_channel_type::fp16,
                     class half2_2d>({48, 128, 32}, {16, 16, 4}, seed);
  printTestName("Running 1D half4\n");
  failed |= run_test<2, sycl::half, 4, sycl::image_channel_type::fp16,
                     class half4_1d>({2816, 32}, {32, 32}, seed);
  printTestName("Running 2D half4\n");
  failed |= run_test<3, sycl::half, 4, sycl::image_channel_type::fp16,
                     class half4_2d>({48, 128, 32}, {16, 16, 4}, seed);

  if (failed) {
    std::cerr << "An error has occured!\n";
    return 1;
  }

  std::cout << "All tests passed!\n";
  return 0;
}
