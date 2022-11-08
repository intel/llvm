// The test checks that invoke_simd implementation performs proper conversions
// on the actual arguments of 'double' type.

// TODO: enable on Windows once driver is ready
// REQUIRES: gpu && linux
// UNSUPPORTED: cuda || hip

// RUN: %clangxx -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr %s -o %t.out
// RUN: env IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 %GPU_RUN_PLACEHOLDER %t.out

#define TEST_DOUBLE_TYPE
#include "invoke_simd_conv.cpp"
