// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "unsampled_images.cpp"
