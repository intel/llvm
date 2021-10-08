#include "split-per-source.h"

void runKernelsFromFile2() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    auto KernelID1 = sycl::get_kernel_id<File2Kern1>();
    auto KernelID2 = sycl::get_kernel_id<File1Kern1>();
    auto KernelID3 = sycl::get_kernel_id<File1Kern2>();
    auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        Q.get_context(), {KernelID1});
    auto Krn = KB.get_kernel(KernelID1);

    assert(!KB.has_kernel(KernelID2));
    assert(!KB.has_kernel(KernelID3));

    Q.submit([&](cl::sycl::handler &Cgh) {
      Cgh.use_kernel_bundle(KB);
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File2Kern1>(Krn, [=]() { Acc[0] = 3; });
    });
  }
  assert(Data == 3);
}
