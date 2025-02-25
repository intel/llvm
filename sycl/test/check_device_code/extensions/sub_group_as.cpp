// RUN: %clangxx -fsycl-device-only -O3 -S -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

constexpr int N = 64;

SYCL_EXTERNAL void test(sycl::accessor<int, 1, sycl::access::mode::read_write,
                                       sycl::access::target::device>
                            global,
                        sycl::local_accessor<int, 1> local,
                        sycl::nd_item<1> it) {
  int v[N] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
              32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  sycl::sub_group sg = it.get_sub_group();
  if (!it.get_local_id(0)) {
    int end = it.get_global_id(0) + it.get_local_range()[0];
    for (int i = it.get_global_id(0); i < end; i++) {
      local[i] = i;
    }
  }
  it.barrier();

  int i = (it.get_global_id(0) / sg.get_max_local_range()[0]) *
          sg.get_max_local_range()[0];

  auto x = sg.load(&global[i]);
  auto y = sg.load(&local[i]);
  auto z = sg.load(v + i);

  sg.store(&global[i], x + y + z);
}

// clang-format off
// CHECK:  call spir_func void {{.*}}spirv_ControlBarrierjjj

// load() for global address space
// CHECK: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK: {{.*}}SubgroupLocalInvocationId
// CHECK: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)


// load() for local address space
// CHECK: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK: {{.*}}SubgroupLocalInvocationId
// CHECK: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)

// load() for private address space
// CHECK: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK: {{.*}}SubgroupLocalInvocationId
// CHECK: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)

// store() for global address space
// NOTE: Call to __spirv_GenericCastToPtrExplicit_ToLocal is consolidated with an earlier call to it.
// CHECK: {{.*}}SubgroupLocalInvocationId
// CHECK: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK: call spir_func void {{.*}}spirv_SubgroupBlockWriteINTEL{{.*}}(ptr addrspace(1)
// clang-format off
