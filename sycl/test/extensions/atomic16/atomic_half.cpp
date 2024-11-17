// RUN: %clangxx %s -fsycl-device-only -S -o - | FileCheck %s

// CHECK: call spir_func void @_Z19__spirv_AtomicStore{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half noundef 0xH3C00)
// CHECK: call spir_func noundef half @_Z18__spirv_AtomicLoad{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}})
// CHECK: call spir_func noundef half @_Z22__spirv_AtomicExchange{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half {{.*}})
// CHECK: call spir_func noundef half @_Z21__spirv_AtomicFAddEXT{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half {{.*}})
// CHECK: call spir_func noundef half @_Z21__spirv_AtomicFAddEXT{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half {{.*}})
// CHECK: call spir_func noundef half @_Z21__spirv_AtomicFMinEXT{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half {{.*}})
// CHECK: call spir_func noundef half @_Z21__spirv_AtomicFMaxEXT{{.*}}(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, half {{.*}})

#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      // TODO: also test short, unsigned short and bfloat16 when available
      sycl::half val_half = 1.0;
      auto ref_half =
          sycl::atomic_ref<sycl::half, sycl::memory_order_acq_rel,
                           sycl::memory_scope_device,
                           sycl::access::address_space::local_space>(val_half);

      ref_half.store(val_half);
      sycl::half load = ref_half.load();
      sycl::half exch = ref_half.exchange(val_half);
      sycl::half add = ref_half.fetch_add(val_half);
      sycl::half sub = ref_half.fetch_sub(val_half);
      sycl::half min = ref_half.fetch_min(val_half);
      sycl::half max = ref_half.fetch_max(val_half);
    });
  });
  return 0;
}
