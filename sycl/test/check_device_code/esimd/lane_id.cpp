// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

// This test checks the codegen for the basic usage of __ESIMD_SIMT_BEGIN -
// __ESIMD_SIMT_END construct.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// Wrapper for designating a scalar region of code that will be
// vectorized by the backend compiler.
#define SIMT_BEGIN(N, lane)                                                    \
  [&]() SYCL_ESIMD_FUNCTION ESIMD_NOINLINE [[intel::sycl_esimd_vectorize(N)]] {                                     \
    int lane = __esimd_lane_id();
#define SIMT_END                                                               \
  }                                                                            \
  ();

// CHECK-LABEL: define dso_local spir_func void @_Z3fooi
//CHECK:  call spir_func void @_ZZ3fooiENKUlvE_clEv(
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<int, 16> foo(int x) {
  simd<int, 16> v = 0;
  SIMT_BEGIN(16, lane)
  //CHECK: define internal spir_func void @_ZZ3fooiENKUlvE_clEv({{.*}}) {{.*}} #[[ATTR:[0-9]+]]
  //CHECK: %{{[0-9a-zA-Z_.]+}} = tail call spir_func noundef i32 @_Z15__esimd_lane_idv()
  v.select<1, 1>(lane) = x++;
  SIMT_END
  return v;
}

//CHECK: attributes #[[ATTR]] = { {{.*}} "CMGenxSIMT"="16" {{.*}}}
