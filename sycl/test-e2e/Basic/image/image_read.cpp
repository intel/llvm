// UNSUPPORTED: hip || gpu-intel-pvc
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "image_read.h"

int main() {

  s::queue myQueue(s::default_selector_v);

  bool passed = true;

  // Float image
  if (!test<s::float4, s::image_channel_type::fp32>(myQueue))
    passed = false;

  // 32-bit signed integer image
  if (!test<s::int4, s::image_channel_type::signed_int32>(myQueue))
    passed = false;

  // 32-bit unsigned integer image
  if (!test<s::uint4, s::image_channel_type::unsigned_int32>(myQueue))
    passed = false;

  return passed ? 0 : -1;
}
