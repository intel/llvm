// RUN: %{build} -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %{run} %t.out

// Tests oneapi extension native tanh math function for sycl::vec and
// sycl::marray float cases.

#include "ext_native_math_common.hpp"

int main() {

  sycl::queue q;

  const float tv[16] = {-2.0, -1.5, -1.0, 0.0, 2.0,  1.5, 1.0,   0.0,
                        -1.7, 1.7,  -1.2, 1.2, -3.0, 3.0, -10.0, 10.0};
  const float tl[16] = {-0.97, -0.91, -0.77, -0.1, 0.95, 0.89, 0.75,  -0.1,
                        -0.94, 0.92,  -0.84, 0.82, -1.0, 0.98, -1.10, 0.98};
  const float tu[16] = {-0.95, -0.89, -0.75, 0.1,  0.97,  0.91, 0.77,  0.1,
                        -0.92, 0.94,  -0.82, 0.84, -0.98, 1.00, -0.98, 1.10};

  native_tanh_tester<float>(q, tv[0], tl[0], tu[0]);
  native_tanh_tester<sycl::float2>(q, {tv[0], tv[1]}, {tl[0], tl[1]},
                                   {tu[0], tu[1]});
  native_tanh_tester<sycl::float3>(
      q, {tv[0], tv[1], tv[2]}, {tl[0], tl[1], tl[2]}, {tu[0], tu[1], tu[2]});

  native_tanh_tester<sycl::float4>(q, {tv[0], tv[1], tv[2], tv[3]},
                                   {tl[0], tl[1], tl[2], tl[3]},
                                   {tu[0], tu[1], tu[2], tu[3]});
  native_tanh_tester<sycl::marray<float, 3>>(
      q, {tv[0], tv[1], tv[2]}, {tl[0], tl[1], tl[2]}, {tu[0], tu[1], tu[2]});
  native_tanh_tester<sycl::marray<float, 4>>(q, {tv[0], tv[1], tv[2], tv[3]},
                                             {tl[0], tl[1], tl[2], tl[3]},
                                             {tu[0], tu[1], tu[2], tu[3]});
  native_tanh_tester<sycl::float8>(
      q, {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7]},
      {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7]},
      {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7]});
  native_tanh_tester<sycl::float16>(
      q,
      {tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7], tv[8], tv[9],
       tv[10], tv[11], tv[12], tv[13], tv[14], tv[15]},
      {tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7], tl[8], tl[9],
       tl[10], tl[11], tl[12], tl[13], tl[14], tl[15]},
      {tu[0], tu[1], tu[2], tu[3], tu[4], tu[5], tu[6], tu[7], tu[8], tu[9],
       tu[10], tu[11], tu[12], tu[13], tu[14], tu[15]});

  return 0;
}
