// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -emit-llvm -o %t.comp.ll %s
// RUN: sycl-post-link -ir-output-only -lower-esimd -S %t.comp.ll -o %t.out.ll
// RUN: FileCheck --input-file=%t.out.ll %s

// Performs a basic check that ESIMDOptimizeVecArgCallConv does the right job as
// a part of sycl-post-link.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

ESIMD_PRIVATE simd<float, 3 * 32 * 4> GRF;
#define V(x, w, i) (x).template select<w, 1>(i)

// clang-format off

// "Fall-through case", incoming optimizeable parameter is just returned

__attribute__((noinline))
SYCL_EXTERNAL simd<float, 16> callee__sret__param(simd<float, 16> x) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <16 x float> @_Z19callee__sret__param{{.*}}(<16 x float> %[[PARAM:.+]])
  return x;
}

// * Caller 1: simd object is read from array

__attribute__((noinline))
SYCL_EXTERNAL simd<float, 16> test__sret__fall_through__arr(simd<float, 16> *x, int i) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <16 x float> @_Z29test__sret__fall_through__arr{{.*}}(ptr addrspace(4) noundef %[[PARAM0:.+]], i32 noundef %{{.*}})
  return callee__sret__param(x[i]);
// CHECK: %{{.*}} = call spir_func <16 x float> @_Z19callee__sret__param{{.*}}(<16 x float> %{{.*}})
}

// * Caller 2 : simd object is read from a global

__attribute__((noinline))
SYCL_EXTERNAL simd<float, 16> test__sret__fall_through__glob() SYCL_ESIMD_FUNCTION {
  return callee__sret__param(V(GRF, 16, 0));
// CHECK: %{{.*}} = call spir_func <16 x float> @_Z19callee__sret__param{{.*}}(<16 x float> %{{.*}})
}
