// REQUIRES: aspect-fp16, aspect-ext_intel_legacy_image

// UNSUPPORTED: cuda, hip

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "image_write.h"

int main() {
  s::queue myQueue(s::default_selector_v);

  // Half image
  if (!test<s::half4, s::image_channel_type::fp16>(myQueue))
    return -1;

  return 0;
}
