#include "split-per-source.h"

void runKernelsFromFile2() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::program Prg(Q.get_context());
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    Prg.build_with_kernel_type<File2Kern1>();
    cl::sycl::kernel Krn = Prg.get_kernel<File2Kern1>();

    assert(!Prg.has_kernel<File1Kern1>());
    assert(!Prg.has_kernel<File1Kern2>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<File2Kern1>(Krn, [=]() { Acc[0] = 3; });
    });
  }
  assert(Data == 3);
}
