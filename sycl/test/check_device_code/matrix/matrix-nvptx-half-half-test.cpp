// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<half, 1> bufA(nullptr, range<1>(1));
  buffer<half, 1> bufB(nullptr, range<1>(1));
  buffer<half, 1> bufC(nullptr, range<1>(1));
  buffer<half, 1> bufD(nullptr, range<1>(1));

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

          joint_matrix<half, matrix_use::accumulator, 16, 16,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 16, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 16, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<half, matrix_use::accumulator, 16, 16,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 16, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 16, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class row_row_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<half, matrix_use::accumulator, 32, 8,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 32, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 8, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.mma.row.row.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<half, matrix_use::accumulator, 32, 8,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 32, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 8, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.mma.col.col.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class row_row_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<half, matrix_use::accumulator, 8, 32,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 8, 16, matrix_layout::row_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 32, matrix_layout::row_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.mma.row.row.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });

    cgh.parallel_for<class col_col_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<half, matrix_use::accumulator, 8, 32,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<half, matrix_use::a, 8, 16, matrix_layout::col_major>
              sub_a;

          joint_matrix<half, matrix_use::b, 16, 32, matrix_layout::col_major>
              sub_b;

          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f16.p0i32(i32* %call.ascast.i.i49.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.f16.p0i32(i32* %call.ascast.i.i41.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.f16.p0i32(i32* %call.ascast.i.i33.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.mma.col.col.f16.f16(<2 x half> %8, <2 x half> %9, <2 x half> %10, <2 x half> %11, <2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %21, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %2, <2 x half> %3, <2 x half> %4, <2 x half> %5) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.f16.p0i32(i32* %call.ascast.i.i.i, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);
        });
  });

  return 0;
};
