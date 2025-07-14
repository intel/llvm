// RUN: %clangxx -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/experimental/root_group.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
func(sycl::ext::oneapi::experimental::root_group<1> &rg) {
  // CHECK: call spir_func void @_Z22__spirv_ControlBarrier{{.*}}(i32 noundef 1, i32 noundef 1, i32 noundef 912)
  sycl::group_barrier(rg);
}
