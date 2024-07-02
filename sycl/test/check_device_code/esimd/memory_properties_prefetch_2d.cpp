// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD memory functions accepting compile time properties for prefetch
// and 2D APIs. NOTE: must be run in -O0, as optimizer optimizes away some of
// the code.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_prefetch(AccType &, float *, int byte_offset32, size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_2d(float *);

class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  LocalAccTypeInt local_acc_int;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_prefetch(acc, ptr, byte_offset32, byte_offset64);
    test_2d(ptr);
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar(AccType &acc, LocalAccType &local_acc, LocalAccTypeInt &local_acc_int,
         float *ptr, int byte_offset32, size_t byte_offset64) {
  EsimdFunctor esimdf{acc, local_acc,     local_acc_int,
                      ptr, byte_offset32, byte_offset64};
  kernel<class kernel_esimd>(esimdf);
}

// CHECK-LABEL: define {{.*}} @_Z13test_prefetch{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_prefetch(AccType &acc, float *ptrf,
                                                     int byte_offset32,
                                                     size_t byte_offset64) {
  properties props_cache_load{cache_hint_L1<cache_hint::cached>,
                              cache_hint_L2<cache_hint::uncached>};
  properties props_cache_load_align{cache_hint_L1<cache_hint::cached>,
                                    cache_hint_L2<cache_hint::uncached>,
                                    alignment<8>};

  uint8_t *ptrb = reinterpret_cast<uint8_t *>(ptrf);

  simd<uint32_t, 32> ioffset_n32(byte_offset32, 8);
  simd<uint64_t, 32> loffset_n32(byte_offset64, 16);
  auto ioffset_n32_view = ioffset_n32.select<32, 1>();
  auto loffset_n32_view = loffset_n32.select<32, 1>();

  simd<uint32_t, 16> ioffset_n16(byte_offset32, 8);
  simd<uint64_t, 16> loffset_n16(byte_offset64, 16);
  auto ioffset_n16_view = ioffset_n16.select<16, 1>();
  auto loffset_n16_view = loffset_n16.select<16, 1>();

  simd_mask<32> mask_n32 = 1;
  simd_mask<16> mask_n16 = 1;
  simd_mask<1> mask_n1 = 1;

  // Test USM prefetch using this plan:
  // 1) prefetch(usm, offsets): offsets is simd or simd_view
  // 2) prefetch(usm, offsets, mask): offsets is simd or simd_view
  // 3) prefetch(usm, offset): same as (1) above, but with offset as a scalar.
  // 4) prefetch(usm, offset): same as (1) and (2) above, but with VS > 1.

  // 1) prefetch(usm, offsets): offsets is simd or simd_view

  // CHECK-COUNT-10: call void @llvm.genx.lsc.prefetch.stateless.v32i1.v32i64(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, i32 0)
  prefetch(ptrf, ioffset_n32, props_cache_load);
  prefetch<float, 32>(ptrf, ioffset_n32_view, props_cache_load);
  prefetch<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), props_cache_load);
  prefetch(ptrf, ioffset_n32_view, props_cache_load);
  prefetch(ptrf, ioffset_n32_view.select<32, 1>(), props_cache_load);

  prefetch(ptrf, loffset_n32, props_cache_load);
  prefetch<float, 32>(ptrf, loffset_n32_view, props_cache_load);
  prefetch<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), props_cache_load);
  prefetch(ptrf, loffset_n32_view, props_cache_load);
  prefetch(ptrf, loffset_n32_view.select<32, 1>(), props_cache_load);

  // 2) prefetch(usm, offsets, mask): offsets is simd or simd_view
  // CHECK-COUNT-10: call void @llvm.genx.lsc.prefetch.stateless.v32i1.v32i64(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, i32 0)
  prefetch(ptrf, ioffset_n32, mask_n32, props_cache_load);
  prefetch<float, 32>(ptrf, ioffset_n32_view, mask_n32, props_cache_load);
  prefetch<float, 32>(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32,
                      props_cache_load);
  prefetch(ptrf, ioffset_n32_view, mask_n32, props_cache_load);
  prefetch(ptrf, ioffset_n32_view.select<32, 1>(), mask_n32, props_cache_load);

  prefetch(ptrf, loffset_n32, mask_n32, props_cache_load);
  prefetch<float, 32>(ptrf, loffset_n32_view, mask_n32, props_cache_load);
  prefetch<float, 32>(ptrf, loffset_n32_view.select<32, 1>(), mask_n32,
                      props_cache_load);
  prefetch(ptrf, loffset_n32_view, mask_n32, props_cache_load);
  prefetch(ptrf, loffset_n32_view.select<32, 1>(), mask_n32, props_cache_load);

  // 3) prefetch(usm, offset): offset is scalar
  // CHECK-COUNT-16: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  __ESIMD_NS::prefetch(ptrf, byte_offset32, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, byte_offset64, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, mask_n1, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, byte_offset32, mask_n1, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, byte_offset64, mask_n1, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, byte_offset32, mask_n1, props_cache_load);
  __ESIMD_NS::prefetch(ptrf, byte_offset64, mask_n1, props_cache_load);

  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset32, props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset64, props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, mask_n1, props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset32, mask_n1,
                                   props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset64, mask_n1,
                                   props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset32, mask_n1,
                                   props_cache_load_align);
  __ESIMD_NS::prefetch<uint8_t, 4>(ptrb, byte_offset64, mask_n1,
                                   props_cache_load_align);

  // 4) prefetch(usm, ...): same as (1), (2) above, but with VS > 1.
  // CHECK-COUNT-10: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0)
  prefetch<float, 32, 2>(ptrf, ioffset_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, ioffset_n16_view, props_cache_load);
  prefetch<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(),
                         props_cache_load);
  prefetch<2>(ptrf, ioffset_n16_view, props_cache_load);
  prefetch<2>(ptrf, ioffset_n16_view.select<16, 1>(), props_cache_load);

  prefetch<float, 32, 2>(ptrf, loffset_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, loffset_n16_view, props_cache_load);
  prefetch<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(),
                         props_cache_load);
  prefetch<2>(ptrf, loffset_n16_view, props_cache_load);
  prefetch<2>(ptrf, loffset_n16_view.select<16, 1>(), props_cache_load);

  // CHECK-COUNT-10: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0)
  prefetch<float, 32, 2>(ptrf, ioffset_n16, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, ioffset_n16_view, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
                         props_cache_load);
  prefetch<2>(ptrf, ioffset_n16_view, mask_n16, props_cache_load);
  prefetch<2>(ptrf, ioffset_n16_view.select<16, 1>(), mask_n16,
              props_cache_load);

  prefetch<float, 32, 2>(ptrf, loffset_n16, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, loffset_n16_view, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
                         props_cache_load);
  prefetch<2>(ptrf, loffset_n16_view, mask_n16, props_cache_load);
  prefetch<2>(ptrf, loffset_n16_view.select<16, 1>(), mask_n16,
              props_cache_load);

  // CHECK-COUNT-2: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 7, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  __ESIMD_NS::prefetch<float, 32>(ptrf, 0, props_cache_load);
  __ESIMD_NS::prefetch<float, 32>(ptrf, 0, 1, props_cache_load);

  // Test Acc prefetch using this plan:
  // 1) prefetch(acc, offsets): offsets is simd or simd_view
  // 2) prefetch(acc, offsets, mask): offsets is simd or simd_view
  // 3) prefetch(acc, offset): same as (1) above, but with offset as a scalar.
  // 4) prefetch(acc, offset): same as (1) and (2) above, but with VS > 1.

  // 1) prefetch(acc, offsets): offsets is simd or simd_view
  // CHECK-STATEFUL-COUNT-3: call void @llvm.genx.lsc.prefetch.bti.v32i1.v32i32(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-3: call void @llvm.genx.lsc.prefetch.stateless.v32i1.v32i64(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, i32 0)
  prefetch<float>(acc, ioffset_n32, props_cache_load);
  prefetch<float, 32>(acc, ioffset_n32_view, props_cache_load);
  prefetch<float, 32>(acc, ioffset_n32_view.select<32, 1>(), props_cache_load);

  // 2) prefetch(acc, offsets, mask): offsets is simd or simd_view
  // CHECK-STATEFUL-COUNT-3: call void @llvm.genx.lsc.prefetch.bti.v32i1.v32i32(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-3: call void @llvm.genx.lsc.prefetch.stateless.v32i1.v32i64(<32 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <32 x i64> {{[^)]+}}, i32 0)
  prefetch<float>(acc, ioffset_n32, mask_n32, props_cache_load);
  prefetch<float, 32>(acc, ioffset_n32_view, mask_n32, props_cache_load);
  prefetch<float, 32>(acc, ioffset_n32_view.select<32, 1>(), mask_n32,
                      props_cache_load);

  // 3) prefetch(acc, offset): offset is scalar
  // CHECK-STATEFUL-COUNT-10: call void @llvm.genx.lsc.prefetch.bti.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-10: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 1, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  prefetch<float>(acc, byte_offset32, props_cache_load);
  prefetch<float>(acc, props_cache_load);
  prefetch<float>(acc, mask_n1, props_cache_load);
  prefetch<float>(acc, byte_offset32, mask_n1, props_cache_load);
  prefetch<float>(acc, byte_offset32, mask_n1, props_cache_load);

  prefetch<uint8_t, 4>(acc, byte_offset32, props_cache_load_align);
  prefetch<uint8_t, 4>(acc, props_cache_load_align);
  prefetch<uint8_t, 4>(acc, mask_n1, props_cache_load_align);
  prefetch<uint8_t, 4>(acc, byte_offset32, mask_n1, props_cache_load_align);
  prefetch<uint8_t, 4>(acc, byte_offset32, mask_n1, props_cache_load_align);

  // 4) prefetch(usm, ...): same as (1), (2) above, but with VS > 1.
  // CHECK-STATEFUL-COUNT-3: call void @llvm.genx.lsc.prefetch.bti.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-3: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0)
  prefetch<float, 32, 2>(acc, ioffset_n16, props_cache_load);
  prefetch<float, 32, 2>(acc, ioffset_n16_view, props_cache_load);
  prefetch<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(),
                         props_cache_load);

  // CHECK-STATEFUL-COUNT-3: call void @llvm.genx.lsc.prefetch.bti.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-3: call void @llvm.genx.lsc.prefetch.stateless.v16i1.v16i64(<16 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 2, i8 1, i8 0, <16 x i64> {{[^)]+}}, i32 0)
  prefetch<float, 32, 2>(acc, ioffset_n16, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(acc, ioffset_n16_view, mask_n16, props_cache_load);
  prefetch<float, 32, 2>(acc, ioffset_n16_view.select<16, 1>(), mask_n16,
                         props_cache_load);

  // CHECK-STATEFUL-COUNT-2: call void @llvm.genx.lsc.prefetch.bti.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 7, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS-COUNT-2: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 7, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  prefetch<float, 32>(acc, 0, props_cache_load);
  prefetch<float, 32>(acc, 0, 1, props_cache_load);
}

// CHECK-LABEL: define {{.*}} @_Z7test_2d{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_2d(float *ptr) {
  properties props_cache_load{cache_hint_L1<cache_hint::streaming>,
                              cache_hint_L2<cache_hint::uncached>};
  simd<float, 16> Vals;
  auto Vals_view = Vals.select<16, 1>();

  constexpr int BlockWidth = 16;
  constexpr int BlockHeight = 1;
  constexpr int NBlocks = 1;
  constexpr bool Transposed = false;
  constexpr bool Transformed = false;

  unsigned SurfaceWidth;
  unsigned SurfaceHeight;
  unsigned SurfacePitch;
  int X;
  int Y;
  // Test USM 2d API using this plan:
  // 1) prefetch_2d(): combinations of explicit and default template parameters
  // 2) load_2d(): combinations of explicit and default template parameters
  // 3) same as (2) but without compile time properties
  // 4) store_2d(): combinations of explicit and default template parameters
  // 5) same as (4) but without compile time properties

  // CHECK-COUNT-3: call void @llvm.genx.lsc.prefetch2d.stateless.v1i1.i64(<1 x i1> {{[^)]+}}, i8 5, i8 1, i8 3, i8 1, i8 1, i16 16, i16 1, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}})
  prefetch_2d<float, BlockWidth, BlockHeight, NBlocks>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, props_cache_load);
  prefetch_2d<float, BlockWidth, BlockHeight>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, props_cache_load);
  prefetch_2d<float, BlockWidth>(ptr, SurfaceWidth, SurfaceHeight, SurfacePitch,
                                 X, Y, props_cache_load);

  // CHECK-COUNT-5: call <16 x float> @llvm.genx.lsc.load2d.stateless.v16f32.v1i1.i64(<1 x i1> {{[^)]+}}, i8 5, i8 1, i8 3, i8 1, i8 1, i16 16, i16 1, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}})
  Vals =
      load_2d<float, BlockWidth, BlockHeight, NBlocks, Transposed, Transformed>(
          ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
          props_cache_load);
  Vals = load_2d<float, BlockWidth, BlockHeight, NBlocks, Transposed>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, props_cache_load);
  Vals = load_2d<float, BlockWidth, BlockHeight, NBlocks>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, props_cache_load);
  Vals = load_2d<float, BlockWidth, BlockHeight>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, props_cache_load);
  Vals = load_2d<float, BlockWidth>(ptr, SurfaceWidth, SurfaceHeight,
                                    SurfacePitch, X, Y, props_cache_load);

  // CHECK-COUNT-5: call <16 x float> @llvm.genx.lsc.load2d.stateless.v16f32.v1i1.i64(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 3, i8 1, i8 1, i16 16, i16 1, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}})
  Vals =
      load_2d<float, BlockWidth, BlockHeight, NBlocks, Transposed, Transformed>(
          ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
  Vals = load_2d<float, BlockWidth, BlockHeight, NBlocks, Transposed>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
  Vals = load_2d<float, BlockWidth, BlockHeight, NBlocks>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
  Vals = load_2d<float, BlockWidth, BlockHeight>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
  Vals = load_2d<float, BlockWidth>(ptr, SurfaceWidth, SurfaceHeight,
                                    SurfacePitch, X, Y);

  // CHECK-COUNT-4: call void @llvm.genx.lsc.store2d.stateless.v1i1.i64.v16f32(<1 x i1> {{[^)]+}}, i8 5, i8 1, i8 3, i8 1, i8 1, i16 16, i16 1, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, <16 x float> {{[^)]+}})
  store_2d<float, BlockWidth, BlockHeight>(ptr, SurfaceWidth, SurfaceHeight,
                                           SurfacePitch, X, Y, Vals,
                                           props_cache_load);
  store_2d<float, BlockWidth>(ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X,
                              Y, Vals, props_cache_load);
  store_2d<float, BlockWidth, BlockHeight, 16>(ptr, SurfaceWidth, SurfaceHeight,
                                               SurfacePitch, X, Y, Vals_view,
                                               props_cache_load);
  store_2d<float, BlockWidth, BlockHeight, 16>(
      ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Vals_view.select<16, 1>(), props_cache_load);

  // CHECK-COUNT-4: call void @llvm.genx.lsc.store2d.stateless.v1i1.i64.v16f32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 3, i8 1, i8 1, i16 16, i16 1, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, <16 x float> {{[^)]+}})
  store_2d<float, BlockWidth, BlockHeight>(ptr, SurfaceWidth, SurfaceHeight,
                                           SurfacePitch, X, Y, Vals);
  store_2d<float, BlockWidth>(ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X,
                              Y, Vals);
  store_2d<float, BlockWidth, BlockHeight, 16>(ptr, SurfaceWidth, SurfaceHeight,
                                               SurfacePitch, X, Y, Vals_view);
  store_2d<float, BlockWidth, BlockHeight, 16>(ptr, SurfaceWidth, SurfaceHeight,
                                               SurfacePitch, X, Y,
                                               Vals_view.select<16, 1>());
}