// NVPTX sm_80+ uses inline asm, which must not be reached in constant
// evaluation. The default CUDA arch is below sm_80, so it is set explicitly.
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80 -nocudalib -fno-sycl-libspirv -Wno-unsafe-libspirv-not-linked -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning -Wno-sycl-strict -std=c++17
// REQUIRES: cuda
// expected-no-diagnostics
#include <sycl/sycl.hpp>

#if !defined(__SYCL_BITCAST_IS_CONSTEXPR)
#error "bfloat16 constexpr support requires __SYCL_BITCAST_IS_CONSTEXPR"
#endif

using sycl::ext::oneapi::bfloat16;

// Check to and from float conversion (only from uses nvptx)
constexpr float CheckToFloat() {
  const bfloat16 bf16{1.f};
  return bf16;
}
static_assert(CheckToFloat() == 1.f,
              "bfloat16 to float conversion must be constexpr");

constexpr float CheckNegate() {
  bfloat16 bf16{1.f};
  return -bf16;
}
static_assert(CheckNegate() == -1.f, "bfloat16 unary minus must be constexpr");
