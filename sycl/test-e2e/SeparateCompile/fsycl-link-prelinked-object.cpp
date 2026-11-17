// Checks that a program runs correctly when an object created with
// -fsycl-link is linked with -fsycl together with the objects it was created
// from, into an executable and into a shared library (#23500).
//
// The kernel captures an accessor, so dead argument elimination removes some
// of its arguments. The object created with -fsycl-link must be passed to the
// host linker as is. If its device image went through the device link again,
// the information about the removed arguments would be lost, and setting the
// kernel arguments would fail with
// UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX.
//
// Only the old offloading model is affected.
//
// REQUIRES: target-spir, linux

// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver %fPIC -DKERNEL -c %s -o %t.dir/kernel.o
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver %fPIC -fsycl-link %t.dir/kernel.o -o %t.dir/device.o
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver -c %s -o %t.dir/main.o

/// Executable linked from the -fsycl-link object and the original object.
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver %t.dir/main.o %t.dir/device.o %t.dir/kernel.o -o %t.dir/app.out
// RUN: %{run} %t.dir/app.out

/// Shared library linked from the -fsycl-link object and the original object.
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver %shared_lib %t.dir/device.o %t.dir/kernel.o -o %t.dir/libkernel.so
// RUN: %clangxx -fsycl %{sycl_target_opts} --no-offload-new-driver %t.dir/main.o -L%t.dir -lkernel -Wl,-rpath,%t.dir -o %t.dir/app_shared.out
// RUN: %{run} %t.dir/app_shared.out

#ifdef KERNEL
#include <sycl/detail/core.hpp>

int launch() {
  int Result = 0;
  {
    sycl::buffer<int> Buffer(&Result, sycl::range<1>(1));
    sycl::queue Queue;
    Queue.submit([&](sycl::handler &CGH) {
      sycl::accessor Output(Buffer, CGH, sycl::write_only);
      CGH.single_task([=] { Output[0] = 42; });
    });
  }
  return Result;
}
#else
#include <iostream>

int launch();

int main() {
  int Result = launch();
  if (Result != 42) {
    std::cerr << "Expected 42, got " << Result << "\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
#endif
