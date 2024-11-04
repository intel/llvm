// REQUIRES: linux
// RUN: %clangxx -fsycl-device-only  -fsycl-targets=native_cpu -S -emit-llvm  -o %t_temp.ll %s
// RUN: %clangxx -mllvm -sycl-native-cpu-backend -S -emit-llvm -o - %t_temp.ll | FileCheck %s
#include <sycl/sycl.hpp>

constexpr sycl::memory_order order = sycl::memory_order::relaxed;
constexpr sycl::memory_scope scope = sycl::memory_scope::work_group;
constexpr sycl::access::address_space space =
    sycl::access::address_space::global_space;

class Test;
using namespace sycl;
int main() {
  queue q;
  const size_t N = 32;
  float sum = 0;
  std::vector<float> output(N);
  std::fill(output.begin(), output.end(), 0.f);
  {
    buffer<float> sum_buf(&sum, 1);
    q.submit([&](handler &cgh) {
       auto sum = sum_buf.template get_access<access::mode::read_write>(cgh);
       cgh.parallel_for<Test>(range<1>(N), [=](item<1> it) {
         int gid = it.get_id(0);
         auto atm = atomic_ref<float, order, scope, space>(sum[0]);
         atm.fetch_add(1.f, order);
         //CHECK-DAG: float @_Z21__spirv_AtomicFAddEXT{{.*}}(ptr {{.*}} %[[ARG0:.*]], i32 {{.*}}, i32 {{.*}}, float {{.*}} %[[ARG3:.*]])
         //CHECK:   %[[RES:.*]] = atomicrmw fadd ptr addrspace(1) %[[ARG0]], float %[[ARG3]] monotonic, align 4
         //CHECK:   ret float %[[RES]]
         atm.fetch_max(1.f, order);
         //CHECK-DAG: float @_Z21__spirv_AtomicFMaxEXT{{.*}}(ptr {{.*}} %[[ARG0:.*]], i32 {{.*}}, i32 {{.*}}, float {{.*}} %[[ARG3:.*]])
         //CHECK:   %[[RES:.*]] = atomicrmw fmax ptr addrspace(1) %[[ARG0]], float %[[ARG3]] monotonic, align 4
         //CHECK:   ret float %[[RES]]
         atm.fetch_min(1.f, order);
         //CHECK-DAG: float @_Z21__spirv_AtomicFMinEXT{{.*}}(ptr {{.*}} %[[ARG0:.*]], i32 {{.*}}, i32 {{.*}}, float {{.*}} %[[ARG3:.*]])
         //CHECK:   %[[RES:.*]] = atomicrmw fmin ptr addrspace(1) %[[ARG0]], float %[[ARG3]] monotonic, align 4
         //CHECK:   ret float %[[RES]]
       });
     }).wait_and_throw();
  }
}
