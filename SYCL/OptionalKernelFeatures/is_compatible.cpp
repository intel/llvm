// requires: cpu, gpu, accelerator
// RUN: %clangxx -fsycl -O0 %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

[[sycl::device_has(sycl::aspect::cpu)]] void foo(){};
[[sycl::device_has(sycl::aspect::gpu)]] void bar(){};
[[sycl::device_has(sycl::aspect::accelerator)]] void baz(){};

class KernelCPU;
class KernelGPU;
class KernelACC;

int main() {
  bool Compatible = true;
  bool Called = false;
  sycl::device Dev;
  sycl::queue Q(Dev);

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

  return (Compatible && Called) ? 0 : 1;
}
