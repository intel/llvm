// RUN: %clangxx -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

SYCL_EXTERNAL void
sub_group_prefetch(sycl::sub_group sg, void *voidPtr, int *intPtr) {
  auto prop = syclex::properties{syclex::prefetch_hint_L1};
  syclex::joint_prefetch(sg, voidPtr, 2);
  // CHECK: call {{.*}}__spirv_SubgroupBlockPrefetchINTEL
  syclex::joint_prefetch(sg, voidPtr, 2, prop);
  // CHECK: call {{.*}}__spirv_SubgroupBlockPrefetchINTEL

  syclex::joint_prefetch(sg, intPtr);
  // CHECK: call {{.*}}__spirv_SubgroupBlockPrefetchINTEL
  syclex::joint_prefetch(sg, intPtr, prop);
  // CHECK: call {{.*}}__spirv_SubgroupBlockPrefetchINTEL

  // Check for unoptimized sizes
  syclex::joint_prefetch(sg, intPtr, 3);
  // CHECK: call {{.*}}__spirv_ocl_prefetch
  syclex::joint_prefetch(sg, intPtr, 3, prop);
  // CHECK: call {{.*}}__spirv_ocl_prefetch
}
