// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: target-nvidia || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "sampled_images.cpp"
