// UNSUPPORTED: hip || cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include "image_write.h"

int main() {
  s::default_selector selector;
  s::queue myQueue(selector);

  // Device doesn't support cl_khr_fp16 extension - skip.
  if (!myQueue.get_device().has(sycl::aspect::fp16))
    return 0;

  // Half image
  if (!test<s::half4, s::image_channel_type::fp16>(myQueue))
    return -1;

  return 0;
}
