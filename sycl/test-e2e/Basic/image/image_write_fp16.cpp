// REQUIRES: aspect-fp16, aspect-ext_intel_legacy_image
// UNSUPPORTED: hip || cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "image_write.h"

int main() {
  s::queue myQueue(s::default_selector_v);

  // Half image
  if (!test<s::half4, s::image_channel_type::fp16>(myQueue))
    return -1;

  return 0;
}
