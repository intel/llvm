// REQUIRES: aspect-ext_intel_legacy_image
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: hip
// TODO: re-enable on cuda device.
// See https://github.com/intel/llvm/issues/1542#issuecomment-707877817 for more
// details.

#include "image_write.h"

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
