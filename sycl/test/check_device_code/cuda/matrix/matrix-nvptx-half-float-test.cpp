// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s
// RUN: %clangxx -Xclang -opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<half, 1> bufA(nullptr, range<1>(1));
  buffer<half, 1> bufB(nullptr, range<1>(1));
  buffer<float, 1> bufC(nullptr, range<1>(1));
  buffer<float, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<half, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<half, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accB(bufB, cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accC(bufC, cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accD(bufD, cgh);

    cgh.parallel_for<class row_row_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, half, use::a, 16, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 16, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, half, use::a, 16, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 16, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
        });

    cgh.parallel_for<class row_row_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, half, use::a, 32, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 8, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.row.row.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.row.row.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, half, use::a, 32, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 8, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.col.col.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: %30 = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.col.col.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
        });

    cgh.parallel_for<class row_row_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, half, use::a, 8, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 32, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.row.row.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.row.row.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, half, use::a, 8, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, half, use::b, 16, 32, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f32.p0f32(float* %0, i32 16
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f32.p0(ptr %0, i32 16
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
          // CHECK: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.f16.p0i32(i32* %11, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.f16.p0(ptr %10, i32 16
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.f16.p0i32(i32* %22, i32 16
          // CHECK-OPAQUE: tail call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.f16.p0(ptr %20, i32 16
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
              stride);
          // CHECK: = tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.col.col.f32.f32(<2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %20, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, <2 x half> %30, <2 x half> %31, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.col.col.f32.f32(<2 x half> %12, <2 x half> %13, <2 x half> %14, <2 x half> %15, <2 x half> %16, <2 x half> %17, <2 x half> %18, <2 x half> %19, <2 x half> %22, <2 x half> %23, <2 x half> %24, <2 x half> %25, <2 x half> %26, <2 x half> %27, <2 x half> %28, <2 x half> %29, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.f32.p0f32(float* %41, float %33, float %34, float %35, float %36, float %37, float %38, float %39, float %40, i32 16
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.f32.p0(ptr %39, float %31, float %32, float %33, float %34, float %35, float %36, float %37, float %38, i32 16
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::no>(),
              stride, layout::col_major);
        });
  });

  return 0;
};
