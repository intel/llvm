// Semaphore version of the Vulkan/SYCL 1D image read interop test.

// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// clang-format off
// RUN: %{run} %t.out --type float --channels 1 32 --semaphores
// RUN: %{run} %t.out --type float --channels 2 32 --semaphores
// RUN: %{run} %t.out --type float --channels 4 32 --semaphores
// RUN: %{run} %t.out --type half --channels 1 32 --semaphores
// RUN: %{run} %t.out --type int32 --channels 2 32 --semaphores
// RUN: %{run} %t.out --type uint32 --channels 4 32 --semaphores
// RUN: %{run} %t.out --type int16 --channels 1 32 --semaphores
// RUN: %{run} %t.out --type uint16 --channels 2 32 --semaphores
// RUN: %{run} %t.out --type uint8 --channels 4 32 --semaphores
// RUN: %{run} %t.out --type int8 --channels 1 32 --semaphores
// RUN-IF: !cuda, %{run} %t.out --type unorm8 --channels 2 32 --semaphores
// RUN: %{run} %t.out --type float --channels 4 --sampled 32 --semaphores
// RUN: %{run} %t.out --type int16 --channels 4 --sampled 32 --semaphores
// RUN: %{run} %t.out --type int8 --channels 4 --sampled 32 --semaphores
// RUN-IF: !cuda, %{run} %t.out --type unorm8 --channels 4 --sampled 32 --semaphores
// clang-format on

#include "./vulkan_sycl_image_interop_read_1d_common.hpp"
