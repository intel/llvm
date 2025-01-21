// REQUIRES: cuda
// REQUIRES: vulkan
// REQUIRES: build-and-run-mode

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} %t.out

#define TEST_SEMAPHORE_IMPORT
#include "unsampled_images.cpp"
