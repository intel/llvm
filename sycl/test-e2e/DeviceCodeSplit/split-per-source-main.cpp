// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_source -I %S/Inputs -o %t.out %s %S/Inputs/split-per-source-second-file.cpp \
// RUN: -fsycl-dead-args-optimization
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// XFAIL: hip_nvidia

#include "Inputs/split-per-source.h"

int main() {
  sycl::queue Q;
  int Data = 0;
  {
    sycl::buffer<int, 1> Buf(&Data, sycl::range<1>(1));
    auto KernelID = sycl::get_kernel_id<File1Kern1>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID});
    auto Krn = KB.get_kernel(KernelID);

    assert(KB.has_kernel(KernelID));
    // TODO uncomment once the KernelInfo in multiple translation units
    // bug is fixed.
    // assert(!Prg.has_kernel<File2Kern1>());

    Q.submit([&](sycl::handler &Cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern1>(/*Krn,*/ [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    sycl::buffer<int, 1> Buf(&Data, sycl::range<1>(1));
    auto KernelID1 = sycl::get_kernel_id<File1Kern1>();
    auto KernelID2 = sycl::get_kernel_id<File1Kern2>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID1});
    auto Krn = KB.get_kernel(KernelID2);

    assert(KB.has_kernel(KernelID1));
    // TODO uncomment once the KernelInfo in multiple translation units
    // bug is fixed.
    // assert(!Prg.has_kernel<File2Kern1>());

    Q.submit([&](sycl::handler &Cgh) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File1Kern2>(/*Krn,*/ [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  runKernelsFromFile2();

  return 0;
}
