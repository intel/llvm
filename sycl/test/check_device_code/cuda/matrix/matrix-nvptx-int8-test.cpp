// REQUIRES: cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_72 -S -Xclang -emit-llvm %s -o -| FileCheck %s --check-prefixes=CHECK-OPAQUE

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

// The following SYCL_EXTERNAL functions (e.g. row_row_m16n16k16) test perform
// matrix multiplication in various different ways. They were originally written
// in the following manner:
//
//  ...
//  q.submit([&] (handler &cgh) {
//      sycl::accessor<int8_t,  1, sycl::access::mode::read_write,
//      sycl::target::device> accA(bufA, cgh); sycl::accessor<int8_t,  1,
//      sycl::access::mode::read_write, sycl::target::device> accB(bufB, cgh);
//      sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
//      sycl::target::device> accC(bufC, cgh); sycl::accessor<int32_t, 1,
//      sycl::access::mode::read_write, sycl::target::device> accD(bufD, cgh);

//      cgh.parallel_for<class row_row_m16n16k16>(nd_range<2>({1, 32}, {1, 32}),
//          [=](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
//              row_row_m16n16k16(accA, accB, accC, accD, item);
//          });
//  });
//

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
row_row_m16n16k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::row_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::row_major);
  // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
col_col_m16n16k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accA,
                  sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accB,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accC,
                  sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                 sycl::target::device>
                      accD,
                  nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 16, 16> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 16, 16, layout::col_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 16, layout::col_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::col_major);
  // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32 } @llvm.nvvm.wmma.m16n16k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::col_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
row_row_m32n8k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accB,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accC,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accD,
                 nd_item<2> item) {

  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 32, 8> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 32, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 8, layout::row_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::row_major);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
col_col_m32n8k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accB,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accC,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accD,
                 nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 32, 8> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 32, 16, layout::col_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 8, layout::col_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.c.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::col_major);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m32n8k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m32n8k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m32n8k16.store.d.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::col_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
row_row_m8n32k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accB,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accC,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accD,
                 nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 8, 32> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 8, 16, layout::row_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 32, layout::row_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::row_major);
  // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.row.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.row.row.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.row.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::row_major);
}

SYCL_EXTERNAL [[sycl::reqd_work_group_size(1, 1, 32)]] void
col_col_m8n32k16(sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accA,
                 sycl::accessor<int8_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accB,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accC,
                 sycl::accessor<int32_t, 1, sycl::access::mode::read_write,
                                sycl::target::device>
                     accD,
                 nd_item<2> item) {
  sycl::sub_group sg = item.get_sub_group();

  joint_matrix<sub_group, int32_t, use::accumulator, 8, 32> sub_c{};
  joint_matrix<sub_group, int8_t, use::a, 8, 16, layout::col_major> sub_a{};
  joint_matrix<sub_group, int8_t, use::b, 16, 32, layout::col_major> sub_b{};

  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.c.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 16)
  joint_matrix_load(sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(),
                    stride, layout::col_major);
  // CHECK-OPAQUE: tail call i32 @llvm.nvvm.wmma.m8n32k16.load.a.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_a, accA.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.load.b.col.stride.s8.p0(ptr %{{.*}}, i32 16)
  joint_matrix_load(
      sg, sub_b, accB.template get_multi_ptr<access::decorated::yes>(), stride);
  // CHECK-OPAQUE: tail call { i32, i32, i32, i32, i32, i32, i32, i32 } @llvm.nvvm.wmma.m8n32k16.mma.col.col.s8(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  // CHECK-OPAQUE: tail call void @llvm.nvvm.wmma.m8n32k16.store.d.col.stride.s32.p1(ptr addrspace(1) %{{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 16)
  joint_matrix_store(sg, sub_c,
                     accD.template get_multi_ptr<access::decorated::yes>(),
                     stride, layout::col_major);
}
