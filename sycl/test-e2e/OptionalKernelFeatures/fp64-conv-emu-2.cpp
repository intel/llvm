// -fsycl-fp64-conv-emu is not a language feature, but an implementation
// feature that is only available for Intel GPUs. In this test we recreate a
// user-provided scenario where an application has two kinds of kernels: ones
// which require full fp64 support and ones which can be emulated.
//
// The test ensures that the application can be successfully AOT compiled and
// ran on different HW (w/ and w/o native fp64 support), i.e. it also serves
// as an integration test for two features: fp64 emulation and optional kernel
// features AOT.
//
// REQUIRES: ocloc
//
// We require a certain HW here, because we specifically want to exercise AOT
// compilation and not JIT fallback. However, to make this test run in more
// environments (and therefore cover more scenarios), the list of HW is bigger
// than just a single target.
//
// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12 || arch-intel_gpu_pvc || arch-intel_gpu_mtl_h || arch-intel_gpu_mtl_u
//
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: FP64 emulation is an Intel specific feature.

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10,intel_gpu_dg2_g11,intel_gpu_dg2_g12,intel_gpu_pvc,intel_gpu_mtl_h,intel_gpu_mtl_u -fsycl-fp64-conv-emu --no-offload-new-driver %O0 %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10,intel_gpu_dg2_g11,intel_gpu_dg2_g12,intel_gpu_pvc,intel_gpu_mtl_h,intel_gpu_mtl_u -fsycl-fp64-conv-emu --offload-new-driver %O0 %s -o %t.out
// RUN: %{run} %t.out

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10,intel_gpu_dg2_g11,intel_gpu_dg2_g12,intel_gpu_pvc,intel_gpu_mtl_h,intel_gpu_mtl_u -fsycl-fp64-conv-emu --offload-new-driver -g %O0 %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
using namespace sycl;

template <typename T> struct Increment {
  T operator()(T x) const { return x + 1; }
};

template <typename T> struct IntCastThenIncrement {
  int operator()(T x) const { return static_cast<int>(x) + 1; }
};

template <typename Op> int test(queue &q) {
  double res[] = {1.};
  {
    buffer<double, 1> buf(res, 1);
    q.submit([&](handler &cgh) {
       accessor acc(buf, cgh);
       cgh.single_task([=] { acc[0] = Op()(acc[0]); });
     }).wait();
  }
  double ref = 1.;
  ref = Op()(ref);
  if (res[0] != ref) {
    std::cout << typeid(Op).name() << " fail: got " << res[0] << ", expected "
              << ref << "\n";
    return 1;
  }
  return 0;
}

int main() {
  int nfail = 0;
  queue q;

  nfail += test<Increment<int>>(q);
  nfail += test<Increment<long>>(q);
  nfail += test<Increment<float>>(q);

  // This test is currently disabled because it requires the -ze-fp64-gen-emu
  // IGC option to run FP64 arithmetic operations. The -fsycl-fp64-conv-emu flag
  // only enables the -ze-fp64-gen-conv-emu IGC option, which provides partial
  // FP64 emulation limited to kernels with FP64 conversions but no FP64
  // computations.
  // TODO: Implement support for a new flag, -fsycl-fp64-gen-emu, which will
  // enable the use of the -ze-fp64-gen-emu IGC option. if
  // (q.get_device().has(aspect::fp64)) {
  //   nfail += test<Increment<double>>(q);
  // }

  nfail += test<IntCastThenIncrement<double>>(q);

  if (nfail == 0)
    std::cout << "success\n";
  return nfail;
}
