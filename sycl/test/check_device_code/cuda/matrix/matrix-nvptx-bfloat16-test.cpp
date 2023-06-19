// REQUIRES: cuda

// RUN: %clangxx -Xclang -no-opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s
// RUN: %clangxx -Xclang -opaque-pointers -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

constexpr int stride = 16;

int main() {

  buffer<bfloat16, 1> bufA(nullptr, range<1>(1));
  buffer<bfloat16, 1> bufB(nullptr, range<1>(1));
  buffer<float, 1> bufC(nullptr, range<1>(1));
  buffer<float, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {
    sycl::accessor<bfloat16, 1, sycl::access::mode::read_write,
                   sycl::target::device>
        accA(bufA, cgh);
    sycl::accessor<bfloat16, 1, sycl::access::mode::read_write,
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
          joint_matrix<sub_group, bfloat16, use::a, 16, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 16, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.row.row.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m16n16k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 16, 16> sub_c{};
          joint_matrix<sub_group, bfloat16, use::a, 16, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 16, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.mma.col.col.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
        });

    cgh.parallel_for<class row_row_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, bfloat16, use::a, 32, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 8, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.row.row.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.row.row.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m32n8k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 32, 8> sub_c{};
          joint_matrix<sub_group, bfloat16, use::a, 32, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 8, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.col.col.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m32n8k16.mma.col.col.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
        });

    cgh.parallel_for<class row_row_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, bfloat16, use::a, 8, 16, layout::row_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 32, layout::row_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.row.row.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.row.row.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::row_major);
        });

    cgh.parallel_for<class col_col_m8n32k16>(
        nd_range<2>({1, 32}, {1, 32}),
        [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();

          joint_matrix<sub_group, float, use::accumulator, 8, 32> sub_c{};
          joint_matrix<sub_group, bfloat16, use::a, 8, 16, layout::col_major>
              sub_a{};
          joint_matrix<sub_group, bfloat16, use::b, 16, 32, layout::col_major>
              sub_b{};

          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_c, accC.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
          // CHECK: tail call { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.bf16.p0i32(i32* %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.bf16.p0(ptr %{{.*}}, i32 16)
          joint_matrix_load(
              sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(),
              stride);
          // CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.col.col.bf16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float {{.*}}
          // CHECK-OPAQUE: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m8n32k16.mma.col.col.bf16(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}})
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          // CHECK: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.f32.p1f32(float addrspace(1)* %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}}, i32 16)
          // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.f32.p1(ptr addrspace(1) %{{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, float {{.*}}, i32 16)
          joint_matrix_store(
              sg, sub_c, accD.template get_multi_ptr<access::decorated::yes>(),
              stride, layout::col_major);
        });
  });

  return 0;
};
