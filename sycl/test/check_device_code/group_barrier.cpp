// RUN: %clangxx -fsycl-device-only -fsycl-unnamed-lambda -S -Xclang -emit-llvm %s -o - | FileCheck %s
#include <sycl/sycl.hpp>

const auto TestLambda = [](auto G) {
  sycl::group_barrier(G);
  sycl::group_barrier(G, sycl::memory_scope_work_item);
  sycl::group_barrier(G, sycl::memory_scope_sub_group);
  sycl::group_barrier(G, sycl::memory_scope_work_group);
  sycl::group_barrier(G, sycl::memory_scope_device);
  sycl::group_barrier(G, sycl::memory_scope_system);
};

int main() {
  sycl::queue Q;

  Q.submit([](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{1}, sycl::range{1}},
                     [](sycl::nd_item<1> item) {
                       auto G = item.get_group();
                       auto SG = item.get_sub_group();
                       TestLambda(G);
                       TestLambda(SG);
                     });
  });
  Q.submit([](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1}, sycl::range{1, 1}},
                     [](sycl::nd_item<2> item) {
                       auto G = item.get_group();
                       TestLambda(G);
                     });
  });
  Q.submit([](sycl::handler &CGH) {
    CGH.parallel_for(sycl::nd_range{sycl::range{1, 1, 1}, sycl::range{1, 1, 1}},
                     [](sycl::nd_item<3> item) {
                       auto G = item.get_group();
                       TestLambda(G);
                     });
  });
  return 0;
}
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 4, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 3, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 1, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 0, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 3, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 4, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 3, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 1, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 3, i32 0, i32 912) #2

// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 4, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 3, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 1, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 0, i32 912) #2

// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 4, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 3, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 2, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 1, i32 912) #2
// CHECK: tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 2, i32 0, i32 912) #2
