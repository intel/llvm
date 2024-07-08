// RUN: %clangxx -O0 -fsycl -fno-sycl-esimd-force-stateless-mem -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-esimd-force-stateless-mem -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD intrinsic translation.
// NOTE: must be run in -O0, as optimizer optimizes away some of the code.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void foo(AccType &);

class EsimdFunctor {
public:
  AccType acc;
  void operator()() __attribute__((sycl_explicit_simd)) { foo(acc); }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar(AccType &acc) {
  EsimdFunctor esimdf{acc};
  kernel<class kernel_esimd>(esimdf);
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void foo(AccType &acc) {
  constexpr int VL = 4;
  int *ptr = 0;
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  simd<int, VL> data1 = 1;
  lsc_block_store<int, VL>(ptr, data1);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  simd<int, VL> data2 = lsc_block_load<int, VL>(ptr);

  //CHECK: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  lsc_prefetch<int, VL, lsc_data_size::default_size, cache_hint::uncached,
               cache_hint::cached>(ptr);

  simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
  simd_mask<VL> mask;

  // CHECK: call void @llvm.genx.lsc.store.stateless.v4i1.v4i64.v4i32(<4 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  lsc_scatter<int>(ptr, offsets, data1);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  simd<int, VL> data3 = lsc_gather<int>(ptr, offsets);

  // CHECK: call void @llvm.genx.lsc.prefetch.stateless.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0)
  lsc_prefetch<int, 1, lsc_data_size::default_size, cache_hint::uncached,
               cache_hint::cached>(ptr, offsets);

  uint32_t surf_offset = 1 * VL * sizeof(int);

  // CHECK-STATEFUL:  call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  lsc_block_store<int, VL>(acc, surf_offset, data1);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.load.bti.v4i32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  simd<int, VL> data4 = lsc_block_load<int, VL>(acc, surf_offset);

  // CHECK-STATEFUL:  call void @llvm.genx.lsc.prefetch.bti.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call void @llvm.genx.lsc.prefetch.stateless.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  lsc_prefetch<int, 4, lsc_data_size::default_size, cache_hint::uncached,
               cache_hint::cached>(acc, surf_offset);

  // CHECK-STATEFUL:  call void @llvm.genx.lsc.store.bti.v4i1.v4i32.v4i32(<4 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call void @llvm.genx.lsc.store.stateless.v4i1.v4i64.v4i32(<4 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  lsc_scatter<int>(acc, offsets, data1);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  simd<int, VL> data5 = lsc_gather<int>(acc, offsets);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  data5 = lsc_gather<int>(acc, offsets, mask);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  data5 = lsc_gather<int>(acc, offsets, mask, data5);

  // CHECK-STATEFUL:  call void @llvm.genx.lsc.prefetch.bti.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, i32 {{[^)]+}})
  // CHECK-STATELESS: call void @llvm.genx.lsc.prefetch.stateless.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 0, i8 1, i8 2, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, i32 0)
  lsc_prefetch<int, 1, lsc_data_size::default_size, cache_hint::uncached,
               cache_hint::cached>(acc, offsets);

  // CHECK: call void @llvm.genx.lsc.store.slm.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  lsc_slm_block_store<int, VL>(surf_offset, data1);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  simd<int, VL> data6 = lsc_slm_gather<int>(offsets);

  auto add = simd<int, VL>(5);
  auto compare = simd<int, VL>(VL, 1);
  auto swap = compare * 2;
  auto pred = simd_mask<VL>(1);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_flat_atomic_0 =
      lsc_atomic_update<atomic_op::inc, int>(ptr, offsets, pred);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_flat_atomic_1 =
      lsc_atomic_update<atomic_op::add, int>(ptr, offsets, add, pred);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
  auto res_flat_atomic_2 = lsc_atomic_update<atomic_op::cmpxchg, int>(
      ptr, offsets, compare, swap, pred);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_slm_atomic_0 =
      lsc_slm_atomic_update<atomic_op::inc, int>(offsets, pred);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_slm_atomic_1 =
      lsc_slm_atomic_update<atomic_op::add, int>(offsets, add, pred);

  // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
  auto res_slm_atomic_2 = lsc_slm_atomic_update<atomic_op::cmpxchg, int>(
      offsets, compare, swap, pred);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_surf_atomic_0 =
      lsc_atomic_update<atomic_op::inc, int>(acc, offsets, pred);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
  auto res_surf_atomic_1 =
      lsc_atomic_update<atomic_op::add, int>(acc, offsets, add, pred);

  // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> undef)
  // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
  auto res_surf_atomic_2 = lsc_atomic_update<atomic_op::cmpxchg, int>(
      acc, offsets, compare, swap, pred);

  constexpr unsigned Width = 4;
  constexpr unsigned Height = 4;
  constexpr unsigned NumBlocks = 2;
  unsigned data_height, data_width, data_pitch, x, y;

  // CHECK: call <32 x i32> @llvm.genx.lsc.load2d.stateless.v32i32.v1i1.i64(<1 x i1> {{[^)]+}}, i8 1, i8 1, i8 3, i8 1, i8 2, i16 4, i16 4, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}})
  simd<int, Width * Height * NumBlocks> data7 =
      lsc_load_2d<int, Width, Height, NumBlocks, false, false,
                  cache_hint::uncached, cache_hint::uncached>(
          ptr, data_width, data_height, data_pitch, x, y);

  simd<int, Width * Height * 1> data8 = 7;
  // CHECK: call void @llvm.genx.lsc.store2d.stateless.v1i1.i64.v16i32(<1 x i1> {{[^)]+}}, i8 1, i8 1, i8 3, i8 1, i8 1, i16 4, i16 4, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, <16 x i32> {{[^)]+}})
  lsc_store_2d<int, Width, Height, cache_hint::uncached, cache_hint::uncached>(
      ptr, data_width, data_height, data_pitch, x, y, data8);

  // CHECK: call void @llvm.genx.lsc.prefetch2d.stateless.v1i1.i64(<1 x i1> {{[^)]+}}, i8 1, i8 2, i8 3, i8 1, i8 2, i16 4, i16 4, i8 0, i64 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}}, i32 {{[^)]+}})
  lsc_prefetch_2d<int, Width, Height, NumBlocks, cache_hint::uncached,
                  cache_hint::cached>(ptr, data_width, data_height, data_pitch,
                                      x, y);

  lsc_fence<lsc_memory_kind::shared_local, lsc_fence_op::none, lsc_scope::group,
            16>();
  // CHECK: call void @llvm.genx.lsc.fence.v16i1(<16 x i1> {{[^)]+}}, i8 3, i8 0, i8 0)
}
