// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that reading and writing with channel order swaps the pixel channels
// correctly.

#include "bindless_helpers.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

static sycl::device dev;

// Uncomment to print additional test information.
// #define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

// Helpers and utilities.
struct util {
  template <sycl::image_channel_order order, typename DType, int NChannels>
  static sycl::vec<DType, NChannels>
  swap_order(const sycl::vec<DType, NChannels> &in) {

    if constexpr (NChannels == 4) {
      switch (order) {
      case sycl::image_channel_order::rgba:
        return in;
      case sycl::image_channel_order::argb:
        return sycl::vec<DType, NChannels>(in[3], in[0], in[1], in[2]);
      case sycl::image_channel_order::bgra:
        return sycl::vec<DType, NChannels>(in[2], in[1], in[0], in[3]);
      case sycl::image_channel_order::abgr:
        return sycl::vec<DType, NChannels>(in[3], in[2], in[1], in[0]);
      }
    }
    // r and rg channel orders do not require order of channels to be changed.
    return in;
  }

  template <sycl::image_channel_order ReadOrder,
            sycl::image_channel_order WriteOrder, typename DType, int NChannels>
  static void add_host(const std::vector<sycl::vec<DType, NChannels>> &in_0,
                       const std::vector<sycl::vec<DType, NChannels>> &in_1,
                       std::vector<sycl::vec<DType, NChannels>> &out) {
    for (int i = 0; i < out.size(); ++i) {
      out[i] = swap_order<WriteOrder>(swap_order<ReadOrder>(in_0[i]) +
                                      swap_order<ReadOrder>(in_1[i]));
    }
  }
};

template <typename DType, int NChannels, sycl::image_channel_type CType,
          sycl::image_channel_order ReadOrder,
          sycl::image_channel_order WriteOrder, typename KernelName>
bool run_test(sycl::range<1> dims, sycl::range<1> localSize,
              unsigned int seed = 0) {
  using VecType = sycl::vec<DType, NChannels>;

  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Skip half tests if not supported.
  if constexpr (std::is_same_v<DType, sycl::half>) {
    if (!dev.has(sycl::aspect::fp16)) {
#ifdef VERBOSE_PRINT
      std::cout << "Test skipped due to lack of device support for fp16.\n";
#endif
      return false;
    }
  }

  size_t num_elems = dims[0];

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> input_1(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(seed);
  bindless_helpers::fill_rand(input_0, seed);
  bindless_helpers::fill_rand(input_1, seed);
  util::add_host<ReadOrder, WriteOrder>(input_0, input_1, expected);

  try {
    syclexp::image_descriptor desc(dims, NChannels, CType);

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
    q.wait_and_throw();

    {
      sycl::range<1> globalSize = dims;

      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            sycl::nd_range<1>{globalSize, localSize}, [=](sycl::nd_item<1> it) {
              size_t dim0 = it.get_global_id(0);

              VecType px1 = syclexp::fetch_image<VecType, VecType, ReadOrder>(
                  img_input_0, int(dim0));
              VecType px2 = syclexp::fetch_image<VecType, VecType, ReadOrder>(
                  img_input_1, int(dim0));

              auto sum = VecType(
                  bindless_helpers::add_kernel<DType, NChannels>(px1, px2));

              syclexp::write_image<VecType, WriteOrder>(img_output, int(dim0),
                                                        VecType(sum));
            });
      });

      q.wait_and_throw();

      q.ext_oneapi_copy(img_mem_2.get_handle(), actual.data(), desc);
      q.wait_and_throw();
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
                  << "] Expected: " << expected[i][j]
                  << ", Actual: " << actual[i][j] << std::endl;
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

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::rgba, sycl::image_channel_order::rgba,
               class int_1d1>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::argb, sycl::image_channel_order::rgba,
               class int_1d2>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::bgra, sycl::image_channel_order::rgba,
               class int_1d3>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::abgr, sycl::image_channel_order::rgba,
               class int_1d4>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::rgba, sycl::image_channel_order::argb,
               class int_1d5>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::argb, sycl::image_channel_order::argb,
               class int_1d6>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::bgra, sycl::image_channel_order::argb,
               class int_1d7>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::abgr, sycl::image_channel_order::argb,
               class int_1d8>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::rgba, sycl::image_channel_order::bgra,
               class int_1d9>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::argb, sycl::image_channel_order::bgra,
               class int_1d10>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::bgra, sycl::image_channel_order::bgra,
               class int_1d11>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::abgr, sycl::image_channel_order::bgra,
               class int_1d12>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::rgba, sycl::image_channel_order::abgr,
               class int_1d13>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::argb, sycl::image_channel_order::abgr,
               class int_1d14>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::bgra, sycl::image_channel_order::abgr,
               class int_1d15>({32}, {2}, seed);

  failed |=
      run_test<int, 4, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::abgr, sycl::image_channel_order::abgr,
               class int_1d16>({32}, {2}, seed);

  failed |=
      run_test<int, 2, sycl::image_channel_type::signed_int32,
               sycl::image_channel_order::rg, sycl::image_channel_order::rg,
               class int_1d17>({32}, {2}, seed);

  failed |= run_test<int, 1, sycl::image_channel_type::signed_int32,
                     sycl::image_channel_order::r, sycl::image_channel_order::r,
                     class int_1d18>({32}, {2}, seed);

  if (failed) {
    std::cerr << "An error has occurred!\n";
    return 1;
  }

  std::cout << "All tests passed!\n";
  return 0;
}
