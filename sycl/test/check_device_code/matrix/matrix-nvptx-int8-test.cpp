// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_72 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s
// RUN: %clangxx -Xclang -opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_72 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<int8_t, 1> bufA(nullptr, range<1>(1));
  buffer<int8_t, 1> bufB(nullptr, range<1>(1));
  buffer<int32_t, 1> bufC(nullptr, range<1>(1));
  buffer<int32_t, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_row_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::row_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::row_major);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.row.row.s8(i32 %11, i32 %12, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::row_major);
        });

    cgh.parallel_for<class col_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::col_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::col_major);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.col.col.s8(i32 %11, i32 %12, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::col_major);
        });

    cgh.parallel_for<class row_row_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 32, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 8, layout::row_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::row_major);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.row.row.s8(i32 %11, i32 %12, i32 %13, i32 %14, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::row_major);
        });

    cgh.parallel_for<class col_col_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 32, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 8, layout::col_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::col_major);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.col.col.s8(i32 %11, i32 %12, i32 %13, i32 %14, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::col_major);
        });

    cgh.parallel_for<class row_row_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 8, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 32, layout::row_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::row_major);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.row.row.s8(i32 %10, i32 %13, i32 %14, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::row_major);
        });

    cgh.parallel_for<class col_col_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, int32_t, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, int8_t, use::a, 8, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, int8_t, use::b, 16, 32, layout::col_major>
              sub_b{};

          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accC, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.s32.p1(ptr addrspace(1) %_arg_accC, i32 16)
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride,
                            layout::col_major);
          // CHECK: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.s8.p0i32(i32* %call.ascast.i.i{{.*}}.i, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.s8.p0(ptr %call.ascast.i.i{{.*}}.i, i32 16)
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.col.col.s8(i32 %10, i32 %13, i32 %14, i32 %15, i32 %16, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.s32.p1i32(i32 addrspace(1)* %_arg_accD, i32 %18, i32 %19, i32 %20, i32 %21, i32 %22, i32 %23, i32 %24, i32 %25, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.s32.p1(ptr addrspace(1) %_arg_accD, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride,
                             layout::col_major);
        });
  });

  return 0;
};
