// Test the round-to-nearest-even saturating float->int8 conversions. On NVPTX
// these lower to `cvt.rni.sat.{s8,u8}.f32`; other targets use a portable round
// + clamp sequence. The numerical result is identical, so this validates both
// paths against a host reference (including saturation, ties-to-even and NaN).

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/saturating_conversion.hpp>
#include <sycl/usm.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

static int8_t ref_s8(float x) {
  if (std::isnan(x))
    return 0;
  float r = std::nearbyint(x); // round to nearest even (default FP env)
  if (r < -128.0f)
    r = -128.0f;
  if (r > 127.0f)
    r = 127.0f;
  return static_cast<int8_t>(r);
}

static uint8_t ref_u8(float x) {
  if (std::isnan(x))
    return 0;
  float r = std::nearbyint(x);
  if (r < 0.0f)
    r = 0.0f;
  if (r > 255.0f)
    r = 255.0f;
  return static_cast<uint8_t>(r);
}

int main() {
  queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  const float Inputs[] = {0.0f,    -0.0f,  0.5f,    1.5f,   2.5f,   -0.5f,
                          -1.5f,   -2.5f,  0.49f,   0.51f,  126.5f, 127.4f,
                          127.5f,  127.6f, 128.0f,  200.0f, 300.0f, -128.4f,
                          -128.5f, -128.6f, -300.0f, 254.5f, 255.5f,
                          std::numeric_limits<float>::quiet_NaN()};
  constexpr int N = std::size(Inputs);

  float *in = malloc_shared<float>(N, Q);
  int8_t *outS = malloc_shared<int8_t>(N, Q);
  uint8_t *outU = malloc_shared<uint8_t>(N, Q);
  int32_t *packed = malloc_shared<int32_t>(N / 4, Q);
  for (int i = 0; i < N; ++i)
    in[i] = Inputs[i];

  Q.single_task([=]() {
     for (int i = 0; i < N; ++i) {
       outS[i] = float_to_int8_rn(in[i]);
       outU[i] = float_to_uint8_rn(in[i]);
     }
     for (int i = 0; i < N / 4; ++i)
       packed[i] = float4_to_int8x4_rn(
           vec<float, 4>(in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]));
   }).wait();

  int errors = 0;
  for (int i = 0; i < N; ++i) {
    if (outS[i] != ref_s8(in[i])) {
      std::cout << "s8 mismatch at " << in[i] << ": got " << (int)outS[i]
                << " expected " << (int)ref_s8(in[i]) << "\n";
      ++errors;
    }
    if (outU[i] != ref_u8(in[i])) {
      std::cout << "u8 mismatch at " << in[i] << ": got " << (int)outU[i]
                << " expected " << (int)ref_u8(in[i]) << "\n";
      ++errors;
    }
  }
  for (int i = 0; i < N / 4; ++i) {
    const int8_t *b = reinterpret_cast<const int8_t *>(&packed[i]);
    for (int j = 0; j < 4; ++j) {
      int8_t exp = ref_s8(in[4 * i + j]);
      if (b[j] != exp) {
        std::cout << "packed mismatch at lane " << (4 * i + j) << ": got "
                  << (int)b[j] << " expected " << (int)exp << "\n";
        ++errors;
      }
    }
  }

  free(in, Q);
  free(outS, Q);
  free(outU, Q);
  free(packed, Q);

  if (errors) {
    std::cout << errors << " errors\nFAILED\n";
    return 1;
  }
  std::cout << "PASSED\n";
  return 0;
}
