#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>

namespace sycl_imf = sycl::ext::intel::math;

int main(int, char **) {
  sycl::queue device_queue(sycl::default_selector_v);
  std::initializer_list<float> input_vals = {
      -0x1.4p+3, -0x1p+3,  -0x1.8p+2, -0x1p+1,  -0x1.8p-1,
      -0x1p-1,   -0x1p-2,  0x0p+0,    0x1p-2,   0x1p-1,
      0x1p+2,    0x1.8p+2, 0x1p+3,    0x1.4p+3, 0x1.8p+3};
  test(device_queue, input_vals, F(sycl_imf::sigmoid));

  std::initializer_list<sycl::half> input_vals_fp16 = {
      -0x1p+3, -0x1.8p+2, -0x1p+1, -0x1.8p-1, -0x1p-1,  -0x1p-2,
      0x0p+0,  0x1p-2,    0x1p-1,  0x1p+2,    0x1.8p+2, 0x1p+3};
  test(device_queue, input_vals_fp16, F(sycl_imf::sigmoid));

  std::initializer_list<sycl::ext::oneapi::bfloat16> input_vals_bf16 = {
      -0x1p+3, -0x1.8p+2, -0x1p+1, -0x1.8p-1, -0x1p-1,  -0x1p-2,
      0x0p+0,  0x1p-2,    0x1p-1,  0x1p+2,    0x1.8p+2, 0x1p+3};
  test(device_queue, input_vals_bf16, F(sycl_imf::sigmoid));
  return 0;
}
