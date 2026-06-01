// Semaphore coverage version of the Vulkan/SYCL 1D unsampled write interop
// test.

// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: CMPLRLLVM-73525

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// clang-format off
// RUN-IF: !level_zero, %{run} %t.out --type float --channels 1 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type half --channels 2 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type int32 --channels 4 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type uint32 --channels 1 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type int16 --channels 2 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type uint16 --channels 4 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type uint8 --channels 1 32 --semaphores
// RUN-IF: !level_zero, %{run} %t.out --type int8 --channels 4 32 --semaphores
// CUDA doesn't support unorm8, level_zero has issues with semaphores
// XXX-IF: !cuda, %{run} %t.out --type unorm8 --channels 2 32 --semaphores
// clang-format on

#include "./vulkan_sycl_image_interop_write_1d_unsampled_common.hpp"
