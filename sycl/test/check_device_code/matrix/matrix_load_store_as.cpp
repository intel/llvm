// RUN: %clangxx -fsycl-device-only -S -emit-llvm -o - %s | FileCheck %s

// Check that SROA and mem2reg won't leave alloca of matrix type in IR
// CHECK-NOT: alloca target("spirv.JointMatrixINTEL"

// check that correct address spaces are used to load from and store to
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

SYCL_EXTERNAL [[intel::reqd_sub_group_size(16)]] void matrix_store_as(
    multi_ptr<unsigned short, access::address_space::global_space> pA,
    multi_ptr<unsigned short, access::address_space::global_space> pB,
    multi_ptr<float, access::address_space::global_space> pC,
    local_accessor<unsigned short, 2> tileA, nd_item<2> it) {
  joint_matrix<sub_group, unsigned short, use::a, 8, 16, layout::row_major> tA;
  joint_matrix<sub_group, unsigned short, use::b, 16, 16,
               layout::ext_intel_packed>
      tB;
  joint_matrix<sub_group, float, use::accumulator, 8, 16> tC;

  sub_group sg = it.get_sub_group();
  vec<unsigned short, 8> slmvec = sg.load<8>(pA);
  sg.store<8>(tileA.template get_multi_ptr<sycl::access::decorated::yes>(),
              slmvec);
  it.barrier(access::fence_space::local_space);

  // A should load from local address space
  // CHECK: %{{.*}} = tail call spir_func noundef target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0) @_Z[[#]]__spirv_JointMatrixLoadINTEL{{.*}}(ptr addrspace(3) noundef %{{.*}}, i64 noundef 16, i32 noundef 0, i32 noundef 3, i32 noundef 0) #{{.*}}
  joint_matrix_load(
      sg, tA, tileA.template get_multi_ptr<sycl::access::decorated::yes>(), 16);
  // B should load from global address space
  // CHECK: %{{.*}} = tail call spir_func noundef target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1) @_Z[[#]]__spirv_JointMatrixLoadINTEL{{.*}}(ptr addrspace(1) noundef %{{.*}}, i64 noundef 32, i32 noundef 2, i32 noundef 3, i32 noundef 0) #{{.*}}
  joint_matrix_load(sg, tB, pB, 32);
  joint_matrix_mad(sg, tC, tA, tB, tC);
  // C should store to global address space
  // CHECK: tail call spir_func void @_Z[[#]]__spirv_JointMatrixStoreINTEL{{.*}}(ptr addrspace(1) noundef %{{.*}}, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef %{{.*}}, i64 noundef 16, i32 noundef 0, i32 noundef 3, i32 noundef 0) #{{.*}}
  joint_matrix_store(sg, tC, pC, 16, layout::row_major);
}
