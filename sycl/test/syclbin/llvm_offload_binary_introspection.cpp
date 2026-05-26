// Verify that generic offload-aware tooling (`llvm-objdump --offloading`)
// can introspect a v2 SYCLBIN. This is a regression test for the v2
// design's promise that triple/arch are surfaced via OffloadBinary
// StringData so that tools that don't know the SYCLBIN format can still
// show useful information.
//
// RUN: %clangxx --offload-new-driver -fsyclbin=executable -fsycl-targets=spir64 -o %t.syclbin %s
// RUN: llvm-objdump --offloading %t.syclbin | FileCheck %s

#include <sycl/sycl.hpp>

extern "C" SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}

// Expect at least one IR entry that exposes a triple via OffloadBinary
// StringData. Specific entry index / OffloadKind formatting is
// llvm-objdump-version-dependent so we check only the substring that is
// stable.
//
// CHECK: triple {{.*}}spir64-unknown-unknown
