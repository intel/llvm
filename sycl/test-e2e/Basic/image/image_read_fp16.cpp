// REQUIRES: aspect-fp16, aspect-ext_intel_legacy_image

// UNSUPPORTED: hip

// Temporarily add explicit '-O2' to avoid GPU hang issue with O0 optimization.
// RUN: %{build} -O2 -o %t.out
// RUN: %{run} %t.out

#include "image_read.h"

int main() {
  s::queue myQueue;

  // Half image
  if (!test<s::half4, s::image_channel_type::fp16>(myQueue))
    return -1;

  return 0;
}
