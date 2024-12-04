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
// UNSUPPORTED-REASON: FP64 emulation is an Intel specific feature.

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10,intel_gpu_dg2_g11,intel_gpu_dg2_g12,intel_gpu_pvc,intel_gpu_mtl_h,intel_gpu_mtl_u -fsycl-fp64-conv-emu -O0 %s -o %t.out
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

  if (q.get_device().has(aspect::fp64))
    nfail += test<Increment<double>>(q);

  nfail += test<IntCastThenIncrement<double>>(q);

  if (nfail == 0)
    std::cout << "success\n";
  return nfail;
}
