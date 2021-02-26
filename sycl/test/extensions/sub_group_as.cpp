// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -O3 -S -emit-llvm -x c++ %s -o - | FileCheck %s

#include <CL/sycl.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
  cl::sycl::queue queue;
  printf("Device Name = %s\n",
         queue.get_device().get_info<cl::sycl::info::device::name>().c_str());

  // Initialize some host memory
  constexpr int N = 64;
  int host_mem[N];
  for (int i = 0; i < N; ++i) {
    host_mem[i] = i * 100;
  }

  // Use the device to transform each value
  {
    cl::sycl::buffer<int, 1> buf(host_mem, N);
    queue.submit([&](cl::sycl::handler &cgh) {
      auto global =
          buf.get_access<cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          local(N, cgh);

      // Check that load/store functions for raw pointer was called
      //
      // CHECK: spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE4test"
      // CHECK-COUNT-3: call spir_func i32
      // {{.*}}loadIiEENSt9enable_ifIXsr3std7is_sameINS0_6detail9remove_ASIT_E4typeES7_EE5valueES7_E4typeEPS7_{{.*}}i32
      // addrspace(4)*
      // CHECK: call spir_func void {{.*}}storeIiEENSt9enable_ifIXsr3std7is_sameINS0_6detail9remove_ASIT_E4typeES7_EE5valueEvE4typeEPS7_RKS9_({{.*}} addrspace(4)*{{.*}},
      cgh.parallel_for<class test>(
          cl::sycl::nd_range<1>(N, 32), [=](cl::sycl::nd_item<1> it) {
            int v[N] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
            cl::sycl::ONEAPI::sub_group sg = it.get_sub_group();
            if (!it.get_local_id(0)) {
              int end = it.get_global_id(0) + it.get_local_range()[0];
              for (int i = it.get_global_id(0); i < end; i++) {
                local[i] = i;
              }
            }
            it.barrier();

            int i = (it.get_global_id(0) / sg.get_max_local_range()[0]) *
                    sg.get_max_local_range()[0];

            // CHECK: spir_func i32{{.*}}loadIiEENSt9enable_ifIXsr3std7is_sameINS0_6detail9remove_ASIT_E4typeES7_EE5valueES7_E4typeEPS7_
            // CHECK: call spir_func i8 addrspace(3)* @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK:  call spir_func i32 @_Z33__spirv_SubgroupLocalInvocationIdv
            // CHECK: call spir_func i8 addrspace(1)* @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK: call spir_func i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)*
            // CHECK: call spir_func i8* @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK: call spir_func void @__assert_fail
            // CHECK: call spir_func i8 addrspace(4)* @_Z24__spirv_PtrCastToGenericPKv(i8 addrspace(4)*
            // CHECK: call spir_func i32 @_Z30__spirv_SubgroupBlockReadINTELIjET_PU3AS1Kj(i32 addrspace(1)*
            // Global address space
            auto x = sg.load(&global[i]);

            // Local address space
            auto y = sg.load(&local[i]);

            auto z = sg.load(v + i);

            // CHECK: spir_func void {{.*}}storeIiEENSt9enable_ifIXsr3std7is_sameINS0_6detail9remove_ASIT_E4typeES7_EE5valueEvE4typeEPS7_RKS9_
            // CHECK: call spir_func i8 addrspace(3)* @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK: call spir_func i32 @_Z33__spirv_SubgroupLocalInvocationIdv
            // CHECK: call spir_func i8 addrspace(1)* @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELIjEvPU3AS1jT_(i32 addrspace(1)*
            // CHECK: call spir_func i8* @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePKvN5__spv12StorageClass4FlagE(i8 addrspace(4)*
            // CHECK: call spir_func void @__assert_fail
            // CHECK: call spir_func i8 addrspace(4)* @_Z24__spirv_PtrCastToGenericPKv(i8 addrspace(4)*
            // CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELIjEvPU3AS1jT_(i32 addrspace(1)*
            sg.store(&global[i], x + y + z);
          });
    });
  }

  return 0;
}
