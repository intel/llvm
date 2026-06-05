// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out

// All __imf_* bf16 functions are implemented via fp32 emulation, so we don't
// need to check whether underlying device supports bf16 or not.
#include "imf_utils.hpp"
#include <sycl/ext/intel/math.hpp>
int check_nan_convert(std::initializer_list<float> Inputs,
                      const std::vector<uint16_t> &Outputs) {
  assert(Inputs.size() == Outputs.size());
  size_t Idx = 0;
  bool Signaling = false;
  for (const float *It = Inputs.begin(); It != Inputs.end(); ++It) {
    uint16_t bf16_bits = Outputs[Idx];
    if (is_bfloat16_nan(bf16_bits, Signaling)) {
      if (Signaling != is_signaling_nan(*It))
        return 1;
    } else
      return 1;
    ++Idx;
  }

  return 0;
}

int main() {
  sycl::queue device_queue(sycl::default_selector_v);
  std::cout << "Running on "
            << device_queue.get_device().get_info<sycl::info::device::name>()
            << "\n";

  {
    std::initializer_list<float> input_vals = {
        __builtin_bit_cast(float, 0x0),        // +0
        __builtin_bit_cast(float, 0x80000000), // -0
        __builtin_bit_cast(float, 0x1),        // min positive subnormal
        __builtin_bit_cast(float, 0x7FFFFF),   // max positive subnormal
        __builtin_bit_cast(float, 0x5A6BFC),   // positive subnormal
        __builtin_bit_cast(float, 0x80000001), // max negative subnormal
        __builtin_bit_cast(float, 0x807FFFFF), // min negative subnormal
        __builtin_bit_cast(float, 0x805A6FED), // negative subnormal
        __builtin_bit_cast(float, 0x7F800000), // +inf
        __builtin_bit_cast(float, 0xFF800000), // -inf
        __builtin_bit_cast(float, 0x2E05CBA9), // positive normal
        __builtin_bit_cast(float, 0x7E5A8935), // positive normal
        __builtin_bit_cast(float, 0xAE4411FC), // negative normal
        __builtin_bit_cast(float, 0xFA84C773), // negative normal
        __builtin_bit_cast(float, 0x7F7FFFFF), // max positive normal
        __builtin_bit_cast(float, 0x765FCEED), // positive normal
        __builtin_bit_cast(float, 0xFF7FFFFF), // min negative normal
        __builtin_bit_cast(float, 0xAC763561), // negative normal
    };

    std::initializer_list<uint16_t> ref_vals = {
        0x0,    0x8000, 0x0,    0x80,   0x5a,   0x8000, 0x8080, 0x805A, 0x7F80,
        0xFF80, 0x2E06, 0x7E5B, 0xAE44, 0xFA85, 0x7F80, 0x7660, 0xFF80, 0xAC76};

    std::initializer_list<uint16_t> ref_vals_rd = {
        0x0,    0x8000, 0x0,    0x7F,   0x5A,   0x8001, 0x8080, 0x805B, 0x7F80,
        0xFF80, 0x2E05, 0x7E5A, 0xAE45, 0xFA85, 0x7F7F, 0x765F, 0xFF80, 0xAC77};

    std::initializer_list<uint16_t> ref_vals_ru = {
        0x0,    0x8000, 0x1,    0x80,   0x5B,   0x8000, 0x807F, 0x805A, 0x7F80,
        0xFF80, 0x2E06, 0x7E5B, 0xAE44, 0xFA84, 0x7F80, 0x7660, 0xFF7F, 0xAC76};

    std::initializer_list<uint16_t> ref_vals_rz = {
        0x0,    0x8000, 0x0,    0x7F,   0x5A,   0x8000, 0x807F, 0x805A, 0x7F80,
        0xFF80, 0x2E05, 0x7E5A, 0xAE44, 0xFA84, 0x7F7F, 0x765F, 0xFF7F, 0xAC76};

    test_host(input_vals, ref_vals,
              FT(uint16_t, sycl::ext::intel::math::float2bfloat16));
    test_host(input_vals, ref_vals_rd,
              FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rd));
    test_host(input_vals, ref_vals,
              FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rn));
    test_host(input_vals, ref_vals_ru,
              FT(uint16_t, sycl::ext::intel::math::float2bfloat16_ru));
    test_host(input_vals, ref_vals_rz,
              FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rz));
    test(device_queue, input_vals, ref_vals,
         FT(uint16_t, sycl::ext::intel::math::float2bfloat16));
    test(device_queue, input_vals, ref_vals_rd,
         FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rd));
    test(device_queue, input_vals, ref_vals,
         FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rn));
    test(device_queue, input_vals, ref_vals_ru,
         FT(uint16_t, sycl::ext::intel::math::float2bfloat16_ru));
    test(device_queue, input_vals, ref_vals_rz,
         FT(uint16_t, sycl::ext::intel::math::float2bfloat16_rz));
  }

  int ret = 0;
  // Test float2bfloat16 for nan inputs
  {
    std::initializer_list<float> input_vals = {
        __builtin_bit_cast(float, 0x7F800001),  // sNan
        __builtin_bit_cast(float, 0x7F8000F1),  // sNan
        __builtin_bit_cast(float, 0x7FC00000),  // qNan
        __builtin_bit_cast(float, 0x7FC00010)}; // qNan
    std::vector<uint16_t> output_vals(input_vals.size());
    run_on_device(device_queue, input_vals, output_vals,
                  FT(uint16_t, sycl::ext::intel::math::float2bfloat16));
    ret = check_nan_convert(input_vals, output_vals);
  }
  return ret;
}
