// UNSUPPORTED: cuda || rocm
// CUDA does not support device code splitting.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_kernel -o %t.out %s
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

class Kern1;
class Kern2;
class Kern3;

int main() {
  cl::sycl::queue Q;
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern1>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern1>();

    assert(!Prg.has_kernel<Kern2>());
    assert(!Prg.has_kernel<Kern3>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern1>(Krn, [=]() { Acc[0] = 1; });
    });
  }
  assert(Data == 1);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern2>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern2>();

    assert(!Prg.has_kernel<Kern1>());
    assert(!Prg.has_kernel<Kern3>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern2>(Krn, [=]() { Acc[0] = 2; });
    });
  }
  assert(Data == 2);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    cl::sycl::program Prg(Q.get_context());
    Prg.build_with_kernel_type<Kern3>();
    cl::sycl::kernel Krn = Prg.get_kernel<Kern3>();

    assert(!Prg.has_kernel<Kern1>());
    assert(!Prg.has_kernel<Kern2>());

    Q.submit([&](cl::sycl::handler &Cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(Cgh);
      Cgh.single_task<Kern3>(Krn, [=]() { Acc[0] = 3; });
    });
  }
  assert(Data == 3);

  return 0;
}
