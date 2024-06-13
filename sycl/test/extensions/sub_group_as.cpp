// RUN: %clangxx -fsycl-device-only -O3 -S -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s --check-prefix CHECK-O3
// RUN: %clangxx -fsycl-device-only -O0 -S -emit-llvm -Xclang -no-enable-noundef-analysis %s -o - | FileCheck %s --check-prefix CHECK-O0
// Test compilation with -O3 when all methods are inlined in kernel function
// and -O0 when helper methods are preserved.
#include <cassert>
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
// CHECK-O3:  call spir_func void {{.*}}spirv_ControlBarrierjjj

// load() for global address space
// CHECK-O3: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O3: {{.*}}SubgroupLocalInvocationId
// CHECK-O3: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O3: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)
// CHECK-O3: call spir_func void {{.*}}assert


// load() for local address space
// CHECK-O3: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O3: {{.*}}SubgroupLocalInvocationId
// CHECK-O3: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O3: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)
// CHECK-O3: call spir_func void {{.*}}assert

// load() for private address space
// CHECK-O3: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O3: {{.*}}SubgroupLocalInvocationId
// CHECK-O3: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O3: call spir_func i32 {{.*}}spirv_SubgroupBlockRead{{.*}}(ptr addrspace(1)
// CHECK-O3: call spir_func void {{.*}}assert

// store() for global address space
// NOTE: Call to __spirv_GenericCastToPtrExplicit_ToLocal is consolidated with an earlier call to it.
// CHECK-O3: {{.*}}SubgroupLocalInvocationId
// CHECK-O3: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O3: call spir_func void {{.*}}spirv_SubgroupBlockWriteINTEL{{.*}}(ptr addrspace(1)
// CHECK-O3: call spir_func void {{.*}}assert

// load() accepting raw pointers method
// CHECK-O0: define{{.*}}spir_func i32 {{.*}}4sycl3_V19sub_group4load{{.*}}addrspace(4) %
// CHECK-O0: call spir_func ptr addrspace(3) {{.*}}SYCL_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O0: call spir_func i32 {{.*}}sycl3_V19sub_group4load{{.*}}ptr addrspace(3) %
// CHECK-O0: call spir_func ptr addrspace(1) {{.*}}SYCL_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O0: call spir_func i32 {{.*}}sycl3_V19sub_group4load{{.*}}ptr addrspace(1) %
// CHECK-O0: call spir_func void {{.*}}assert

// store() accepting raw pointers method
// CHECK-O0: define{{.*}}spir_func void {{.*}}4sycl3_V19sub_group5store{{.*}}ptr addrspace(4) %
// CHECK-O0: call spir_func ptr addrspace(3) {{.*}}SYCL_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O0: call spir_func void {{.*}}4sycl3_V19sub_group5store{{.*}}, ptr addrspace(3) %
// CHECK-O0: call spir_func ptr addrspace(1) {{.*}}SYCL_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// CHECK-O0: call spir_func void {{.*}}4sycl3_V19sub_group5store{{.*}}, ptr addrspace(1) %
// CHECK-O0: call spir_func void {{.*}}assert

// CHECK-O0: define {{.*}}spir_func ptr addrspace(3) {{.*}}SYCL_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4) %
// CHECK-O0: call spir_func ptr addrspace(3) {{.*}}spirv_GenericCastToPtrExplicit_ToLocal{{.*}}(ptr addrspace(4)
// CHECK-O0: define {{.*}}spir_func ptr addrspace(1) {{.*}}SYCL_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4) %
// CHECK-O0: call spir_func ptr addrspace(1) {{.*}}spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}(ptr addrspace(4)
// clang-format off
