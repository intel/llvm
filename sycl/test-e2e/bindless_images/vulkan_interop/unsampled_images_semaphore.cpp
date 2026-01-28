// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: vulkan

// UNSUPPORTED: linux && run-mode
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21133

// XFAIL: linux && gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21136

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "unsampled_images.cpp"
