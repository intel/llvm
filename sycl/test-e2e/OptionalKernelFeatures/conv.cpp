// REQUIRES: ocloc, linux, arch-intel_gpu_dg2_g10

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10 -fsycl-fp64-conv-emu -O0 %s -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
using namespace sycl;

template <typename T> T op(T x) { return x + 1; }

template <typename T> int test(queue &q) {
  double res[] = {1.};
  {
    buffer<double, 1> buf(res, 1);
    q.submit([&](handler &cgh) {
       accessor acc(buf, cgh);
       cgh.single_task([=] { acc[0] = op<T>(acc[0]); });
     }).wait();
  }
  double ref = 1.;
  ref = op<T>(ref);
  if (res[0] != ref) {
    std::cout << typeid(T).name() << " fail: got " << res[0] << ", expected "
              << ref << "\n";
    return 1;
  }
  return 0;
}

int main() {
  int nfail = 0;
  queue q;

  nfail += test<int>(q);
  nfail += test<long>(q);
  nfail += test<float>(q);
  if (q.get_device().has(aspect::fp64))
    nfail += test<double>(q);

  if (nfail == 0)
    std::cout << "success\n";
  return nfail;
}