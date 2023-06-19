// RUN: %clangxx -O2 -fsycl -fsycl-device-only -Xclang -no-opaque-pointers -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck --check-prefixes=CHECK,CHECK-TYPED %s -input-file=%t_esimd_0.ll
// RUN: %clangxx -O2 -fsycl -fsycl-device-only -Xclang -opaque-pointers -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t -o %t.table
// RUN: FileCheck --check-prefixes=CHECK,CHECK-OPAQUE %s -input-file=%t_esimd_0.ll
// Checks that we set 0 as VCSLMSize when slm_init is used with
// non-constant operand, like with specialization constant.

#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

constexpr sycl::specialization_id<uint64_t> Size(1024);

int main() {
  sycl::queue queue;
  {
    queue.submit([&](sycl::handler &cgh) {
      cgh.single_task<class Kernel3Name>(
          [=](sycl::kernel_handler kh) SYCL_ESIMD_KERNEL {
            slm_init(kh.get_specialization_constant<Size>());
          });
      // CHECK-TYPED: define weak_odr dso_local spir_kernel void @{{.*}}(i8 addrspace(1)* noundef align 1 "VCArgumentIOKind"="0" %{{.*}}) local_unnamed_addr #1
      // CHECK-OPAQUE: define weak_odr dso_local spir_kernel void @{{.*}}(ptr addrspace(1) noundef align 1 "VCArgumentIOKind"="0" %{{.*}}) local_unnamed_addr #1
    });
  }

  return 0;
}

// CHECK: attributes #1 = { {{.*}} "VCSLMSize"="0"
