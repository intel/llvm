// Sampled-only version of the Vulkan/SYCL 2D image read interop test.

// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// UNSUPPORTED: linux
// UNSUPPORTED-TRACKER: GSD-12357

// clang-format off
// RUN: %{run} %t.out --type float --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 --sampled 32x33
// clang-format on

#include "./vulkan_sycl_image_interop_read_2d_common.hpp"
