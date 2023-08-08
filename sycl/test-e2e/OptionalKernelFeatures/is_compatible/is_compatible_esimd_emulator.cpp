// REQUIRES: esimd_emulator

// RUN: %clangxx -fsycl %S/Inputs/is_compatible_with_env.cpp %t_negative_case.out
// RUN: env ONEAPI_DEVICE_SELECTOR=ext_intel_esimd_emulator:gpu %{run} not %t_negative_case.out

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Just an example from
// https://github.com/intel/llvm/tree/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_esimd

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

int main() {
  sycl::device dev;
  if (sycl::is_compatible<class Test>(dev)) {
    float *A = malloc_shared<float>(Size, q);
    float *B = malloc_shared<float>(Size, q);
    float *C = malloc_shared<float>(Size, q);

    for (unsigned i = 0; i != Size; i++) {
      A[i] = B[i] = i;
    }

    q.submit([&](handler &cgh) {
       cgh.parallel_for<class Test>(Size / VL,
                                    [=](id<1> i) [[intel::sycl_explicit_simd]] {
                                      auto offset = i * VL;
                                      // pointer arithmetic, so offset is in
                                      // elements:
                                      simd<float, VL> va(A + offset);
                                      simd<float, VL> vb(B + offset);
                                      simd<float, VL> vc = va + vb;
                                      vc.copy_to(C + offset);
                                    });
     }).wait_and_throw();
    return 0;
  }
  return 1;
}
