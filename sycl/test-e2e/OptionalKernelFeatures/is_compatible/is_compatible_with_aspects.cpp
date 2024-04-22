// requires: cpu, gpu, accelerator
// UNSUPPORTED: hip
// FIXME: enable the test back, see intel/llvm#8146
// RUN: %{build} -O0 -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

[[sycl::device_has(sycl::aspect::cpu)]] void foo(){};
[[sycl::device_has(sycl::aspect::gpu)]] void bar(){};
[[sycl::device_has(sycl::aspect::accelerator)]] void baz(){};

class KernelCPU;
class KernelGPU;
class KernelACC;
class GoodWGSize;
class WrongReqWGSize;

constexpr int SIZE = 2;

int main() {
  bool Compatible = true;
  bool Called = false;
  sycl::device Dev;
  sycl::queue Q(Dev);

  // Returns true for empty vector of kernels
  assert(sycl::is_compatible({}, Dev));

  if (sycl::is_compatible<KernelCPU>(Dev)) {
    Q.submit(
        [&](sycl::handler &h) { h.single_task<KernelCPU>([=]() { foo(); }); });
    Q.wait();
    Compatible &= Dev.is_cpu();
    Called = true;
  }
  if (sycl::is_compatible<KernelGPU>(Dev)) {
    Q.submit(
        [&](sycl::handler &h) { h.single_task<KernelGPU>([=]() { bar(); }); });
    Q.wait();
    Compatible &= Dev.is_gpu();
    Called = true;
  }
  if (sycl::is_compatible<KernelACC>(Dev)) {
    Q.submit(
        [&](sycl::handler &h) { h.single_task<KernelACC>([=]() { baz(); }); });
    Q.wait();
    Compatible &= Dev.is_accelerator();
    Called = true;
  }

  if (sycl::is_compatible<GoodWGSize>(Dev)) {
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class GoodWGSize>(
          sycl::range<2>(4, 2),
          [=](sycl::item<2> it) [[sycl::reqd_work_group_size(SIZE, SIZE)]] {});
    });
    Q.wait();
    Compatible &= (Dev.get_info<sycl::info::device::max_work_group_size>() >
                   (SIZE * SIZE));
    Called = true;
  }

  if (Dev.get_info<sycl::info::device::max_work_group_size>() > INT_MAX) {
    Compatible &= true;
  }
  if (sycl::is_compatible<WrongReqWGSize>(Dev)) {
    assert(false && "sycl::is_compatible<WrongReqWGSize> must be false");
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class WrongReqWGSize>(
          sycl::range<1>(2),
          [=](sycl::item<1> it) [[sycl::reqd_work_group_size(INT_MAX)]] {});
    });
  }
  if (sycl::is_compatible<class WrongReqSGSize>(Dev)) {
    assert(false && "sycl::is_compatible<WrongReqSGSize> must be false");
    Q.submit([&](sycl::handler &h) {
      h.parallel_for<class WrongReqSGSize>(
          sycl::range<1>(2),
          [=](sycl::item<1> it) [[sycl::reqd_sub_group_size(INT_MAX)]] {});
    });
  }

  return (Compatible && Called) ? 0 : 1;
}
