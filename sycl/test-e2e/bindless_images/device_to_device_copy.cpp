// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
#define VERBOSE_PRINT

namespace syclexp = sycl::ext::oneapi::experimental;

void copy_image_mem_handle_to_image_mem_handle(
    syclexp::image_descriptor &desc, const std::vector<float> &testData,
    sycl::device dev, sycl::queue q, std::vector<float> &out) {
  syclexp::image_mem imgMemSrc(desc, dev, q.get_context());
  syclexp::image_mem imgMemDst(desc, dev, q.get_context());

  q.ext_oneapi_copy((void *)testData.data(), imgMemSrc.get_handle(), desc);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemSrc.get_handle(), imgMemDst.get_handle(), desc);
  q.wait_and_throw();

  q.ext_oneapi_copy(imgMemDst.get_handle(), (void *)out.data(), desc);
  q.wait_and_throw();
}

bool check_test(const std::vector<float> &out,
                const std::vector<float> &expected) {
  assert(out.size() == expected.size());
  bool validated = true;
  for (int i = 0; i < out.size(); i++) {
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
  return validated;
}

template <sycl::image_channel_order channelOrder,
          sycl::image_channel_type channelType, int dim>
bool run_copy_test_with(sycl::device &dev, sycl::queue &q,
                        sycl::range<dim> dims) {
  std::vector<float> dataSequence(dims.size());
  std::vector<float> out(dims.size());

  std::vector<float> expected(dims.size());

  std::iota(dataSequence.begin(), dataSequence.end(), 0);
  std::iota(expected.begin(), expected.end(), 0);

  syclexp::image_descriptor desc(dims, channelOrder, channelType);

  copy_image_mem_handle_to_image_mem_handle(desc, dataSequence, dev, q, out);

  return check_test(out, expected);
}

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  bool validated =
      run_copy_test_with<sycl::image_channel_order::r,
                         sycl::image_channel_type::fp32, 1>(dev, q, {4});

  validated &=
      run_copy_test_with<sycl::image_channel_order::r,
                         sycl::image_channel_type::fp32, 2>(dev, q, {4, 4});

  validated &=
      run_copy_test_with<sycl::image_channel_order::r,
                         sycl::image_channel_type::fp32, 3>(dev, q, {4, 4, 4});

  if (!validated) {
    std::cout << "Tests failed";
    return 1;
  }

  std::cout << "Tests passed";

  return 0;
}
