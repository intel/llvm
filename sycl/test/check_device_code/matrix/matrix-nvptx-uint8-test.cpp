// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_72 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<uint8_t, 1> bufA(nullptr, range<1>(1));
  buffer<uint8_t, 1> bufB(nullptr, range<1>(1));
  buffer<int32_t, 1> bufC(nullptr, range<1>(1));
  buffer<int32_t, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class row_row_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 16, 16,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 16, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 16, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.u8.p0i32(i32* %call.ascast.i.i52.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.row.row.u8(i32 %11, i32 %12, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 16, 16,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 16, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 16, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.u8.p0i32(i32* %call.ascast.i.i52.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.col.col.u8(i32 %11, i32 %12, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class row_row_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 32, 8,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 32, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 8, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.u8.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.row.row.u8(i32 %11, i32 %12, i32 %13, i32 %14, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 32, 8,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 32, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 8, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.u8.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.col.col.u8(i32 %11, i32 %12, i32 %13, i32 %14, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class row_row_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 8, 32,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 8, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 32, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.u8.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.row.row.u8(i32 %10, i32 %13, i32 %14, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<int32_t, matrix_use::accumulator, 8, 32,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<uint8_t, matrix_use::a, 8, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<uint8_t, matrix_use::b, 16, 32, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.u8.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.u8.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.col.col.u8(i32 %10, i32 %13, i32 %14, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_14, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });
  });

  return 0;
};
