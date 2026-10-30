// NVPTX sm_80+ uses inline asm, which must not be reached in constant
// evaluation. The default CUDA arch is below sm_80, so it is set explicitly.
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -nocudalib -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict -std=c++17
// REQUIRES: cuda
// expected-no-diagnostics
#include "constexpr-bfloat16.cpp"
