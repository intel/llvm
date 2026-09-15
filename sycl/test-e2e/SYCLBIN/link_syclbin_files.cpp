// REQUIRES: opencl-aot, cpu, opencl-cpu-rt
// REQUIRES: aspect-usm_shared_allocations

// -- Test for linking SYCLBIN files in input and object state into a single
// -- SYCLBIN file in executable state with -fsycl-link.

// The device code in a SYCLBIN file is not tied to a device, so the
// architectures to compile it for have to be named explicitly and the link
// compiles the linked device code for them ahead of time. opencl-aot compiles
// for the isa of the host it runs on, so the architecture only labels the
// resulting device image and the link has to happen on the run system.

// RUN: %{run-aux} %clangxx --offload-new-driver -fsyclbin=input -fsycl-allow-device-image-dependencies %S/Inputs/importing_kernel.cpp -o %t.import.syclbin
// RUN: %{run-aux} %clangxx --offload-new-driver -fsyclbin=object -fsycl-allow-device-image-dependencies %S/Inputs/exporting_function.cpp -o %t.export.syclbin
// RUN: %{run-aux} %clangxx -fsycl-link --offload-arch=corei7 %t.import.syclbin %t.export.syclbin -o %t.linked.syclbin
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out %t.linked.syclbin

// The resulting SYCLBIN file is in executable state, so it cannot be linked
// again.
// RUN: not %clangxx -fsycl-link --offload-arch=corei7 %t.linked.syclbin %t.export.syclbin -o %t.fail.syclbin 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-EXECUTABLE-INPUT
// CHECK-EXECUTABLE-INPUT: error: SYCLBIN file '{{.*}}.syclbin' is in executable state; only SYCLBIN files in input or object state can be linked

// A SYCL_EXTERNAL function left undefined cannot be resolved once the output is
// in executable state, so it is diagnosed as an error.
// RUN: not %clangxx -fsycl-link --offload-arch=corei7 %t.import.syclbin -o %t.fail.syclbin 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-UNDEFINED
// CHECK-UNDEFINED: error: undefined SYCL_EXTERNAL function in the device code being linked:
// CHECK-UNDEFINED-NEXT: TestFunc(int*, int)
// CHECK-UNDEFINED-NEXT: provide the definition in one of the linked inputs

#define SYCLBIN_EXECUTABLE_STATE

#include "Inputs/common.hpp"

#include <iostream>

#include <sycl/usm.hpp>

static constexpr size_t NUM = 10;

int main(int argc, char *argv[]) {
  assert(argc == 2);

  sycl::queue Q;

  int Failed = CommonLoadCheck(Q.get_context(), argv[1]);

  auto KBExe = syclexp::get_kernel_bundle<sycl::bundle_state::executable>(
      Q.get_context(), std::string{argv[1]});

  assert(KBExe.ext_oneapi_has_kernel("TestKernel1"));
  sycl::kernel TestKernel1 = KBExe.ext_oneapi_get_kernel("TestKernel1");

  int *Ptr = sycl::malloc_shared<int>(NUM, Q);
  Q.fill(Ptr, int{0}, NUM).wait_and_throw();

  Q.submit([&](sycl::handler &CGH) {
     CGH.set_args(Ptr, int{NUM});
     CGH.single_task(TestKernel1);
   }).wait_and_throw();

  for (int I = 0; I < NUM; I++) {
    if (Ptr[I] != I) {
      std::cout << "Result: " << Ptr[I] << " expected " << I << "\n";
      ++Failed;
    }
  }

  sycl::free(Ptr, Q);
  return Failed;
}
