// REQUIRES: windows && level_zero && gpu-intel-dg2
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#define TEST_L0_SUPPORTED_VK_FORMAT
#include "sampled_images.cpp"
