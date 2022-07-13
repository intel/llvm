// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel -o %t.out %s \
// RUN: -fsycl-dead-args-optimization
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// XFAIL: hip_nvidia

#include <sycl/sycl.hpp>

class Kern1;
class Kern2;
class Kern3;

int main() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    auto KernelID1 = sycl::get_kernel_id<Kern1>();
    auto KernelID2 = sycl::get_kernel_id<Kern2>();
    auto KernelID3 = sycl::get_kernel_id<Kern3>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID1});
    auto Krn = KB.get_kernel(KernelID1);

    assert(!KB.has_kernel(KernelID2));
    assert(!KB.has_kernel(KernelID3));

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern1>(Krn, [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    auto KernelID1 = sycl::get_kernel_id<Kern1>();
    auto KernelID2 = sycl::get_kernel_id<Kern2>();
    auto KernelID3 = sycl::get_kernel_id<Kern3>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID2});
    auto Krn = KB.get_kernel(KernelID2);

    assert(!KB.has_kernel(KernelID1));
    assert(!KB.has_kernel(KernelID3));

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern2>(Krn, [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    auto KernelID1 = sycl::get_kernel_id<Kern1>();
    auto KernelID2 = sycl::get_kernel_id<Kern2>();
    auto KernelID3 = sycl::get_kernel_id<Kern3>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID3});
    auto Krn = KB.get_kernel(KernelID3);

    assert(!KB.has_kernel(KernelID1));
    assert(!KB.has_kernel(KernelID2));

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern3>(Krn, [=]() { Acc[0] = 3; });
    });
  }
  assert(Data == 3);

  return 0;
}
