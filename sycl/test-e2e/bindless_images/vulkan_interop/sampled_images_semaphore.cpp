// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: target-nvidia || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// UNSUPPORTED: true
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21122

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "sampled_images.cpp"
