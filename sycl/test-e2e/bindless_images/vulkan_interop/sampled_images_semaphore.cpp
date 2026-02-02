// REQUIRES: aspect-ext_oneapi_bindless_images && aspect-ext_oneapi_external_semaphore_import
// REQUIRES: target-nvidia || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "sampled_images.cpp"
