// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx -O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -D__ESIMD_GATHER_SCATTER_LLVM_IR -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD memory functions accepting compile time properties for
// atomic_update APIs. NOTE: must be run in -O0, as optimizer optimizes away
// some of the code.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_atomic_update(AccType &, LocalAccTypeInt &, float *, int byte_offset32,
                   size_t byte_offset64);

class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  LocalAccTypeInt local_acc_int;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_atomic_update(acc, local_acc_int, ptr, byte_offset32, byte_offset64);
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

// CHECK-LABEL: define {{.*}} @_Z18test_atomic_update{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_atomic_update(AccType &acc, LocalAccTypeInt local_acc, float *ptrf,
                   int byte_offset32, size_t byte_offset64) {
  constexpr int VL = 4;
  int *ptr = 0;
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

  simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
  auto offsets_view = offsets.select<VL, 1>();

  auto add = simd<int, VL>(5);
  auto add_view = add.select<VL, 1>();
  auto compare = simd<int, VL>(VL, 1);
  auto compare_view = compare.select<VL, 1>();
  auto swap = compare * 2;
  auto swap_view = swap.select<VL, 1>();
  auto pred = simd_mask<VL>(1);

  properties props_a{cache_hint_L1<cache_hint::uncached>,
                     cache_hint_L2<cache_hint::write_back>};

  properties props_b{cache_hint_L1<cache_hint::uncached>,
                     cache_hint_L2<cache_hint::uncached>};

  // Test atomic update with no operands.
  {
    // USM
    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_0 =
        atomic_update<atomic_op::inc, int>(ptr, offsets, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_1 =
        atomic_update<atomic_op::inc, int>(ptr, offsets, pred, props_b);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_2 =
        atomic_update<atomic_op::inc, int>(ptr, offsets, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_3 =
        atomic_update<atomic_op::inc, int>(ptr, offsets_view, pred, props_a);
    res_atomic_3 =
        atomic_update<atomic_op::inc>(ptr, offsets_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_4 =
        atomic_update<atomic_op::inc, int, VL>(ptr, offsets_view, props_a);
    res_atomic_4 = atomic_update<atomic_op::inc>(ptr, offsets_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_5 = atomic_update<atomic_op::inc, int, VL>(
        ptr, offsets_view.select<VL, 1>(), props_a);
    res_atomic_5 = atomic_update<atomic_op::inc>(
        ptr, offsets_view.select<VL, 1>(), props_a);

    // atomic_upate without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_6 =
        atomic_update<atomic_op::inc, int, VL>(ptr, offsets, pred);

    // atomic_upate without cache hints and mask:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_7 = atomic_update<atomic_op::inc, int, VL>(ptr, offsets);

    // Try the atomic_update without cache hints, but with non-standard
    // vector length to check that LSC atomic is generated.
    // CHECK: call <5 x i32> @llvm.genx.lsc.xatomic.stateless.v5i32.v5i1.v5i64(<5 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <5 x i64> {{[^)]+}}, <5 x i32> undef, <5 x i32> undef, i32 0, <5 x i32> undef)
    {
      constexpr int VL = 5;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      auto pred = simd_mask<VL>(1);
      auto atomic_res =
          atomic_update<atomic_op::inc, int, VL>(ptr, offsets, pred);
    }

    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> undef, <8 x i32> undef, i32 0, <8 x i32> undef)
    {
      int16_t *ptr = 0;
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto atomic_res =
          atomic_update<atomic_op::inc, int16_t, VL>(ptr, offsets);
    }

    // Accessor

    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_acc_0 =
        atomic_update<atomic_op::inc, int>(acc, offsets, pred, props_a);

    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_acc_1 =
        atomic_update<atomic_op::inc, int>(acc, offsets, pred, props_b);

    // CHECK-STATEFUL:  call <4 x i64> @llvm.genx.lsc.xatomic.bti.v4i64.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 4, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i64> undef, <4 x i64> undef, i32 {{[^)]+}}, <4 x i64> undef)
    // CHECK-STATELESS: call <4 x i64> @llvm.genx.lsc.xatomic.stateless.v4i64.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 4, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i64> undef, <4 x i64> undef, i32 0, <4 x i64> undef)
    auto res_atomic_acc_2 =
        atomic_update<atomic_op::inc, int64_t>(acc, offsets, props_a);

    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_acc_3 =
        atomic_update<atomic_op::inc, int>(acc, offsets_view, pred, props_a);

    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_acc_4 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets_view, props_a);

    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_acc_5 = atomic_update<atomic_op::inc, int, VL>(
        acc, offsets_view.select<VL, 1>(), props_a);

    // atomic_upate without cache hints:
    // CHECK-STATEFUL: call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_6 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets, pred);

    // atomic_upate without cache hints and mask:
    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_7 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets);

    // Try the atomic_update without cache hints, but with non-standard
    // vector length to check that LSC atomic is generated.
    // CHECK-STATEFUL:  call <5 x i32> @llvm.genx.lsc.xatomic.bti.v5i32.v5i1.v5i32(<5 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <5 x i32> {{[^)]+}}, <5 x i32> undef, <5 x i32> undef, i32 {{[^)]+}}, <5 x i32> undef)
    // CHECK-STATELESS: call <5 x i32> @llvm.genx.lsc.xatomic.stateless.v5i32.v5i1.v5i64(<5 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <5 x i64> {{[^)]+}}, <5 x i32> undef, <5 x i32> undef, i32 0, <5 x i32> undef)
    {
      constexpr int VL = 5;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      auto pred = simd_mask<VL>(1);
      auto atomic_res_acc =
          atomic_update<atomic_op::inc, int, VL>(acc, offsets, pred);
    }
    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK-STATEFUL:  call <8 x i32> @llvm.genx.lsc.xatomic.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> undef, <8 x i32> undef, i32 {{[^)]+}}, <8 x i32> undef)
    // CHECK-STATELESS: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> undef, <8 x i32> undef, i32 0, <8 x i32> undef)
    {
      using AccType =
          sycl::accessor<int16_t, 1, sycl::access::mode::read_write>;
      AccType *acc = nullptr;
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto atomic_res =
          atomic_update<atomic_op::inc, int16_t, VL>(*acc, offsets);
    }
  }

  // Test atomic update with one operand.
  {
    // USM

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_0 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_1 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_2 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets, add_view, pred, props_a);
    res_atomic_2 =
        atomic_update<atomic_op::add>(ptr, offsets, add_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_3 =
        atomic_update<atomic_op::add, int, VL>(ptr, offsets, add_view, props_a);
    res_atomic_3 =
        atomic_update<atomic_op::add>(ptr, offsets, add_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    res_atomic_3 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets, add_view.select<VL, 1>(), props_a);
    res_atomic_3 = atomic_update<atomic_op::add>(
        ptr, offsets, add_view.select<VL, 1>(), props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_4 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add, pred, props_a);
    res_atomic_4 =
        atomic_update<atomic_op::add>(ptr, offsets_view, add, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_5 =
        atomic_update<atomic_op::add, int, VL>(ptr, offsets_view, add, props_a);
    res_atomic_5 =
        atomic_update<atomic_op::add>(ptr, offsets_view, add, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    res_atomic_5 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view.select<VL, 1>(), add, props_a);
    res_atomic_5 = atomic_update<atomic_op::add>(
        ptr, offsets_view.select<VL, 1>(), add, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_6 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add_view, pred, props_a);
    res_atomic_6 = atomic_update<atomic_op::add>(ptr, offsets_view, add_view,
                                                 pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_7 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add_view, props_a);
    res_atomic_7 =
        atomic_update<atomic_op::add>(ptr, offsets_view, add_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    res_atomic_7 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), props_a);
    res_atomic_7 = atomic_update<atomic_op::add>(
        ptr, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), props_a);

    // atomic_update without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.add.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_8 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, pred);

    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32>{{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    {
      int16_t *ptr = 0;
      constexpr int VL = 4;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto add = simd<int16_t, VL>(5);
      auto atomic_res =
          atomic_update<atomic_op::add, int16_t, VL>(ptr, offsets, add);
    }

    // Accessors

    // CHECK-STATEFUL-COUNT-26:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS-COUNT-26: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_9 =
        atomic_update<atomic_op::add, int>(acc, offsets, add, pred, props_a);

    auto res_atomic_10 =
        atomic_update<atomic_op::add, int>(acc, offsets, add, props_a);

    auto res_atomic_11 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets, add_view, pred, props_a);

    res_atomic_11 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets, add_view.select<VL, 1>(), pred, props_a);

    auto res_atomic_12 =
        atomic_update<atomic_op::add, int, VL>(acc, offsets, add_view, props_a);

    res_atomic_12 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets, add_view.select<VL, 1>(), props_a);

    auto res_atomic_13 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add, pred, props_a);

    res_atomic_13 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view.select<VL, 1>(), add, pred, props_a);

    auto res_atomic_14 =
        atomic_update<atomic_op::add, int, VL>(acc, offsets_view, add, props_a);
    res_atomic_14 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view.select<VL, 1>(), add, props_a);

    auto res_atomic_15 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add_view, pred, props_a);

    res_atomic_15 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), pred,
        props_a);

    auto res_atomic_16 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add_view, props_a);

    res_atomic_16 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), props_a);

    res_atomic_11 =
        atomic_update<atomic_op::add>(acc, offsets, add_view, pred, props_a);

    res_atomic_11 = atomic_update<atomic_op::add>(
        acc, offsets, add_view.select<VL, 1>(), pred, props_a);

    res_atomic_12 =
        atomic_update<atomic_op::add>(acc, offsets, add_view, props_a);

    res_atomic_12 = atomic_update<atomic_op::add>(
        acc, offsets, add_view.select<VL, 1>(), props_a);

    res_atomic_13 =
        atomic_update<atomic_op::add>(acc, offsets_view, add, pred, props_a);

    res_atomic_13 = atomic_update<atomic_op::add>(
        acc, offsets_view.select<VL, 1>(), add, pred, props_a);

    res_atomic_14 =
        atomic_update<atomic_op::add>(acc, offsets_view, add, props_a);
    res_atomic_14 = atomic_update<atomic_op::add>(
        acc, offsets_view.select<VL, 1>(), add, props_a);

    res_atomic_15 = atomic_update<atomic_op::add>(acc, offsets_view, add_view,
                                                  pred, props_a);

    res_atomic_15 =
        atomic_update<atomic_op::add>(acc, offsets_view.select<VL, 1>(),
                                      add_view.select<VL, 1>(), pred, props_a);

    res_atomic_16 =
        atomic_update<atomic_op::add>(acc, offsets_view, add_view, props_a);

    res_atomic_16 = atomic_update<atomic_op::add>(
        acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), props_a);

    // atomic_update without cache hints:
    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.dword.atomic.sub.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.svm.atomic.sub.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_17 =
        atomic_update<atomic_op::sub, int>(acc, offsets, add, pred);

    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK-STATEFUL: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    {
      using AccType =
          sycl::accessor<int16_t, 1, sycl::access::mode::read_write>;
      AccType *acc = nullptr;
      constexpr int VL = 4;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto add = simd<int16_t, VL>(5);
      auto atomic_res =
          atomic_update<atomic_op::add, int16_t, VL>(*acc, offsets, add);
    }
  }

  // Test atomic update with two operands.
  {
    // USM

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_1 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets, swap, compare, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_2 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets, swap, compare, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_3 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view, pred, props_a);
    res_atomic_3 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap, compare_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    res_atomic_3 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view.select<VL, 1>(), pred, props_a);
    res_atomic_3 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap, compare_view.select<VL, 1>(), pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_4 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view, props_a);
    res_atomic_4 = atomic_update<atomic_op::cmpxchg>(ptr, offsets, swap,
                                                     compare_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_5 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare, pred, props_a);
    res_atomic_5 = atomic_update<atomic_op::cmpxchg>(ptr, offsets, swap_view,
                                                     compare, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_6 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare, props_a);
    res_atomic_6 = atomic_update<atomic_op::cmpxchg>(ptr, offsets, swap_view,
                                                     compare, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_7 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare_view, pred, props_a);
    res_atomic_7 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap_view, compare_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_8 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare_view, props_a);
    res_atomic_8 = atomic_update<atomic_op::cmpxchg>(ptr, offsets, swap_view,
                                                     compare_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_9 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view, swap, compare, pred, props_a);
    res_atomic_9 = atomic_update<atomic_op::cmpxchg>(ptr, offsets_view, swap,
                                                     compare, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_10 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view, swap, compare, props_a);
    res_atomic_10 = atomic_update<atomic_op::cmpxchg>(ptr, offsets_view, swap,
                                                      compare, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_11 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap, compare_view, pred, props_a);
    res_atomic_11 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view, swap, compare_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_12 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap, compare_view, props_a);
    res_atomic_12 = atomic_update<atomic_op::cmpxchg>(ptr, offsets_view, swap,
                                                      compare_view, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_13 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare, pred, props_a);
    res_atomic_13 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view, swap_view, compare, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_14 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare, props_a);
    res_atomic_14 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view, swap_view, compare, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_15 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare_view, pred, props_a);
    res_atomic_15 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view, swap_view, compare_view, pred, props_a);

    // CHECK-COUNT-2: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_16 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare_view, props_a);
    res_atomic_16 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view, swap_view, compare_view, props_a);

    // CHECK-COUNT-26: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    res_atomic_4 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view.select<VL, 1>(), props_a);

    res_atomic_5 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view.select<VL, 1>(), compare, pred, props_a);

    res_atomic_6 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view.select<VL, 1>(), compare, props_a);

    res_atomic_7 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_8 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        props_a);

    res_atomic_9 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view.select<VL, 1>(), swap, compare, pred, props_a);

    res_atomic_10 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view.select<VL, 1>(), swap, compare, props_a);

    res_atomic_11 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_12 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        props_a);

    res_atomic_13 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        pred, props_a);

    res_atomic_14 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        props_a);

    res_atomic_15 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_16 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), props_a);

    res_atomic_4 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap, compare_view.select<VL, 1>(), props_a);

    res_atomic_5 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap_view.select<VL, 1>(), compare, pred, props_a);

    res_atomic_6 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap_view.select<VL, 1>(), compare, props_a);

    res_atomic_7 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_8 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        props_a);

    res_atomic_9 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap, compare, pred, props_a);

    res_atomic_10 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap, compare, props_a);

    res_atomic_11 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_12 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        props_a);

    res_atomic_13 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        pred, props_a);

    res_atomic_14 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        props_a);

    res_atomic_15 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_16 = atomic_update<atomic_op::cmpxchg>(
        ptr, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), props_a);

    {
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      simd<float, VL> swap = simd<float, VL>(1) * sizeof(int);
      auto compare = swap * 2;
      auto pred = simd_mask<VL>(1);
      // Do not pass the properties.
      // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 23, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}}, i32 0, <8 x i32> undef)
      auto atomic_res0 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          ptrf, offsets, swap, compare, pred);
      // Now with cache hints.
      // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 23, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}}, i32 0, <8 x i32> undef)
      auto atomic_res1 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          ptrf, offsets, swap, compare, pred, props_a);
    }

    // atomic_update without cache hints
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.cmpxchg.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_100 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare, pred);

    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    {
      int16_t *ptr = 0;
      constexpr int VL = 4;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      simd<int16_t, VL> swap = simd<int16_t, VL>(1) * sizeof(int);
      auto compare = swap * 2;
      auto atomic_res = atomic_update<atomic_op::cmpxchg, int16_t, VL>(
          ptr, offsets, swap, compare);
    }

    // Accessors

    // CHECK-STATEFUL-COUNT-58:  call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS-COUNT-58: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_17 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap, compare, pred, props_a);

    auto res_atomic_18 =
        atomic_update<atomic_op::cmpxchg>(acc, offsets, swap, compare, props_a);

    auto res_atomic_19 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare_view, pred, props_a);

    auto res_atomic_20 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare_view, props_a);

    auto res_atomic_21 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view, compare, pred, props_a);

    auto res_atomic_22 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view, compare, props_a);

    auto res_atomic_23 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view, compare_view, pred, props_a);

    auto res_atomic_24 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view, compare_view, props_a);

    auto res_atomic_25 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap, compare, pred, props_a);

    auto res_atomic_26 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap, compare, props_a);

    auto res_atomic_27 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap, compare_view, pred, props_a);

    auto res_atomic_28 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap, compare_view, props_a);

    auto res_atomic_29 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap_view, compare, pred, props_a);

    auto res_atomic_30 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap_view, compare, props_a);

    auto res_atomic_31 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap_view, compare_view, pred, props_a);

    auto res_atomic_32 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view, swap_view, compare_view, props_a);

    res_atomic_19 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_20 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare_view.select<VL, 1>(), props_a);

    res_atomic_21 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view.select<VL, 1>(), compare, pred, props_a);

    res_atomic_22 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view.select<VL, 1>(), compare, props_a);

    res_atomic_23 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_24 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        props_a);

    res_atomic_25 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap, compare, pred, props_a);

    res_atomic_26 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap, compare, props_a);

    res_atomic_27 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_28 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        props_a);

    res_atomic_29 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        pred, props_a);

    res_atomic_30 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        props_a);

    res_atomic_31 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_32 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), props_a);

    res_atomic_19 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap, compare_view, pred, props_a);

    res_atomic_20 = atomic_update<atomic_op::cmpxchg>(acc, offsets, swap,
                                                      compare_view, props_a);

    res_atomic_21 = atomic_update<atomic_op::cmpxchg>(acc, offsets, swap_view,
                                                      compare, pred, props_a);

    res_atomic_22 = atomic_update<atomic_op::cmpxchg>(acc, offsets, swap_view,
                                                      compare, props_a);

    res_atomic_23 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap_view, compare_view, pred, props_a);

    res_atomic_24 = atomic_update<atomic_op::cmpxchg>(acc, offsets, swap_view,
                                                      compare_view, props_a);

    res_atomic_25 = atomic_update<atomic_op::cmpxchg>(acc, offsets_view, swap,
                                                      compare, pred, props_a);

    res_atomic_26 = atomic_update<atomic_op::cmpxchg>(acc, offsets_view, swap,
                                                      compare, props_a);

    res_atomic_27 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view, swap, compare_view, pred, props_a);

    res_atomic_28 = atomic_update<atomic_op::cmpxchg>(acc, offsets_view, swap,
                                                      compare_view, props_a);

    res_atomic_29 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view, swap_view, compare, pred, props_a);

    res_atomic_30 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view, swap_view, compare, props_a);

    res_atomic_31 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view, swap_view, compare_view, pred, props_a);

    res_atomic_32 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view, swap_view, compare_view, props_a);

    res_atomic_19 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap, compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_20 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap, compare_view.select<VL, 1>(), props_a);

    res_atomic_21 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap_view.select<VL, 1>(), compare, pred, props_a);

    res_atomic_22 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap_view.select<VL, 1>(), compare, props_a);

    res_atomic_23 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_24 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(),
        props_a);

    res_atomic_25 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap, compare, pred, props_a);

    res_atomic_26 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap, compare, props_a);

    res_atomic_27 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        pred, props_a);

    res_atomic_28 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(),
        props_a);

    res_atomic_29 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        pred, props_a);

    res_atomic_30 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare,
        props_a);

    res_atomic_31 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred, props_a);

    res_atomic_32 = atomic_update<atomic_op::cmpxchg>(
        acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), props_a);

    {
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      simd<float, VL> swap = simd<float, VL>(1) * sizeof(int);
      auto compare = swap * 2;
      auto pred = simd_mask<VL>(1);
      // Do not pass the properties.
      // CHECK-STATEFUL:  call <8 x i32> @llvm.genx.lsc.xatomic.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 23, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}} <8 x i32> {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> undef)
      // CHECK-STATELESS: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 23, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}}, i32 0, <8 x i32> undef)
      auto atomic_res0 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          acc, offsets, swap, compare, pred);
      // Now with cache hints.
      // CHECK-STATEFUL:  call <8 x i32> @llvm.genx.lsc.xatomic.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 23, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}} <8 x i32> {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> undef)
      // CHECK-STATELESS: call <8 x i32> @llvm.genx.lsc.xatomic.stateless.v8i32.v8i1.v8i64(<8 x i1> {{[^)]+}}, i8 23, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i64> {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}}, i32 0, <8 x i32> undef)
      auto atomic_res1 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          acc, offsets, swap, compare, pred, props_a);
    }

    // atomic_update without cache hints
    // CHECK-STATEFUL:  call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.svm.atomic.cmpxchg.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_33 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare, pred);

    // Try with int16_t to check that LSC atomic is generated
    // The result is later casted to int16, not captured here.
    // CHECK-STATEFUL: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> undef)
    // CHECK-STATELESS: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    {
      using AccType =
          sycl::accessor<int16_t, 1, sycl::access::mode::read_write>;
      AccType *acc = nullptr;
      constexpr int VL = 4;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      simd<int16_t, VL> swap = simd<int16_t, VL>(1) * sizeof(int);
      auto compare = swap * 2;
      auto atomic_res = atomic_update<atomic_op::cmpxchg, int16_t, VL>(
          *acc, offsets, compare, swap);
    }
  }

  // Test slm_atomic_update without operands.
  {
    // CHECK-COUNT-6: call <4 x i32> @llvm.genx.dword.atomic.dec.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    {
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::dec, int>(offsets, pred);
      auto res_slm_atomic_1 = slm_atomic_update<atomic_op::dec, int>(offsets);
      auto res_slm_atomic_2 =
          slm_atomic_update<atomic_op::dec, int, VL>(offsets_view, pred);
      auto res_slm_atomic_3 =
          slm_atomic_update<atomic_op::dec, int, VL>(offsets_view);
      auto res_slm_atomic_4 = slm_atomic_update<atomic_op::dec, int, VL>(
          offsets_view.select<VL, 1>(), pred);
      auto res_slm_atomic_5 = slm_atomic_update<atomic_op::dec, int, VL>(
          offsets_view.select<VL, 1>());
    }

    // Expect DWORD for load.
    // CHECK: call <4 x i32> @llvm.genx.dword.atomic.or.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_slm_atomic_4 =
        slm_atomic_update<atomic_op::load, int>(offsets, pred);

    // Expect LSC for short.
    {
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto pred = simd_mask<VL>(1);

      // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.slm.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 10, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i32> undef, i32 0, <8 x i32> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::load, int16_t>(offsets, pred);
    }
  }

  // Test slm_atomic_update with one operand.
  {
    // CHECK-COUNT-26: call <4 x i32> @llvm.genx.dword.atomic.add.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    {
      auto res_slm_atomic_1 =
          slm_atomic_update<atomic_op::add>(offsets, add, pred);
      auto res_slm_atomic_2 = slm_atomic_update<atomic_op::add>(offsets, add);
      auto res_slm_atomic_3 =
          slm_atomic_update<atomic_op::add, int, VL>(offsets, add_view, pred);
      auto res_slm_atomic_4 =
          slm_atomic_update<atomic_op::add, int, VL>(offsets, add_view);
      auto res_slm_atomic_5 =
          slm_atomic_update<atomic_op::add, int, VL>(offsets_view, add, pred);
      auto res_slm_atomic_6 =
          slm_atomic_update<atomic_op::add, int, VL>(offsets_view, add);
      auto res_slm_atomic_7 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets_view, add_view, pred);
      auto res_slm_atomic_8 =
          slm_atomic_update<atomic_op::add, int, VL>(offsets_view, add_view);
      res_slm_atomic_3 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets, add_view.select<VL, 1>(), pred);
      res_slm_atomic_4 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets, add_view.select<VL, 1>());
      res_slm_atomic_5 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets_view.select<VL, 1>(), add, pred);
      res_slm_atomic_6 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets_view.select<VL, 1>(), add);
      res_slm_atomic_7 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), pred);
      res_slm_atomic_8 = slm_atomic_update<atomic_op::add, int, VL>(
          offsets_view.select<VL, 1>(), add_view.select<VL, 1>());
      res_slm_atomic_3 =
          slm_atomic_update<atomic_op::add>(offsets, add_view, pred);
      res_slm_atomic_4 = slm_atomic_update<atomic_op::add>(offsets, add_view);
      res_slm_atomic_5 =
          slm_atomic_update<atomic_op::add>(offsets_view, add, pred);
      res_slm_atomic_6 = slm_atomic_update<atomic_op::add>(offsets_view, add);
      res_slm_atomic_7 =
          slm_atomic_update<atomic_op::add>(offsets_view, add_view, pred);
      res_slm_atomic_8 =
          slm_atomic_update<atomic_op::add>(offsets_view, add_view);
      res_slm_atomic_3 = slm_atomic_update<atomic_op::add>(
          offsets, add_view.select<VL, 1>(), pred);
      res_slm_atomic_4 =
          slm_atomic_update<atomic_op::add>(offsets, add_view.select<VL, 1>());
      res_slm_atomic_5 = slm_atomic_update<atomic_op::add>(
          offsets_view.select<VL, 1>(), add, pred);
      res_slm_atomic_6 =
          slm_atomic_update<atomic_op::add>(offsets_view.select<VL, 1>(), add);
      res_slm_atomic_7 = slm_atomic_update<atomic_op::add>(
          offsets_view.select<VL, 1>(), add_view.select<VL, 1>(), pred);
      res_slm_atomic_8 = slm_atomic_update<atomic_op::add>(
          offsets_view.select<VL, 1>(), add_view.select<VL, 1>());
    }

    // Expect LSC for short.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      simd<int16_t, VL> add = simd<int16_t, VL>(1) * sizeof(int);

      // CHECK: call <16 x i32> @llvm.genx.lsc.xatomic.slm.v16i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x i32> {{[^)]+}}, <16 x i32> undef, i32 0, <16 x i32> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::add, int16_t>(offsets, add);
    }
    // Expect LSC for fmin.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(float);
      auto pred = simd_mask<VL>(1);
      simd<float, VL> min = simd<float, VL>(1) * sizeof(int);

      // CHECK: call <16 x float> @llvm.genx.lsc.xatomic.slm.v16f32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 21, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x float> {{[^)]+}}, <16 x float> undef, i32 0, <16 x float> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::fmin, float>(offsets, min, pred);
    }
    // Expect LSC for half.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(sycl::half);
      auto pred = simd_mask<VL>(1);
      simd<sycl::half, VL> min = simd<sycl::half, VL>(1) * sizeof(int);

      // CHECK: call <16 x i32> @llvm.genx.lsc.xatomic.slm.v16i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 21, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x i32> {{[^)]+}}, <16 x i32> undef, i32 0, <16 x i32> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::fmin, sycl::half>(offsets, min, pred);
    }
  }

  // Test slm_atomic_update with two operands.
  {
    // CHECK-COUNT-58: call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_1 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets, swap, compare, pred);
    auto res_atomic_2 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets, swap, compare);

    auto res_atomic_3 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap, compare_view, pred);
    auto res_atomic_4 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap, compare_view);

    auto res_atomic_5 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view, compare, pred);
    auto res_atomic_6 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view, compare);

    auto res_atomic_7 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view, compare_view, pred);
    auto res_atomic_8 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view, compare_view);

    auto res_atomic_9 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap, compare, pred);
    auto res_atomic_10 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap, compare);

    auto res_atomic_11 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap, compare_view, pred);
    auto res_atomic_12 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap, compare_view);

    auto res_atomic_13 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap_view, compare, pred);
    auto res_atomic_14 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap_view, compare);

    auto res_atomic_15 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap_view, compare_view, pred);
    auto res_atomic_16 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view, swap_view, compare_view);
    res_atomic_3 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap, compare_view.select<VL, 1>(), pred);
    res_atomic_4 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap, compare_view.select<VL, 1>());

    res_atomic_5 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view.select<VL, 1>(), compare, pred);
    res_atomic_6 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view.select<VL, 1>(), compare);

    res_atomic_7 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(), pred);
    res_atomic_8 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>());

    res_atomic_9 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap, compare, pred);
    res_atomic_10 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap, compare);

    res_atomic_11 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(), pred);
    res_atomic_12 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>());

    res_atomic_13 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare, pred);
    res_atomic_14 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare);

    res_atomic_15 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_atomic_16 = slm_atomic_update<atomic_op::cmpxchg, int, VL>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());

    res_atomic_3 = slm_atomic_update<atomic_op::cmpxchg>(offsets, swap,
                                                         compare_view, pred);
    res_atomic_4 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets, swap, compare_view);

    res_atomic_5 = slm_atomic_update<atomic_op::cmpxchg>(offsets, swap_view,
                                                         compare, pred);
    res_atomic_6 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets, swap_view, compare);

    res_atomic_7 = slm_atomic_update<atomic_op::cmpxchg>(offsets, swap_view,
                                                         compare_view, pred);
    res_atomic_8 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets, swap_view, compare_view);

    res_atomic_9 = slm_atomic_update<atomic_op::cmpxchg>(offsets_view, swap,
                                                         compare, pred);
    res_atomic_10 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets_view, swap, compare);

    res_atomic_11 = slm_atomic_update<atomic_op::cmpxchg>(offsets_view, swap,
                                                          compare_view, pred);
    res_atomic_12 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets_view, swap, compare_view);

    res_atomic_13 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view, swap_view, compare, pred);
    res_atomic_14 =
        slm_atomic_update<atomic_op::cmpxchg>(offsets_view, swap_view, compare);

    res_atomic_15 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view, swap_view, compare_view, pred);
    res_atomic_16 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view, swap_view, compare_view);
    res_atomic_3 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap, compare_view.select<VL, 1>(), pred);
    res_atomic_4 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap, compare_view.select<VL, 1>());

    res_atomic_5 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap_view.select<VL, 1>(), compare, pred);
    res_atomic_6 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap_view.select<VL, 1>(), compare);

    res_atomic_7 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>(), pred);
    res_atomic_8 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets, swap_view.select<VL, 1>(), compare_view.select<VL, 1>());

    res_atomic_9 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap, compare, pred);
    res_atomic_10 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap, compare);

    res_atomic_11 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>(), pred);
    res_atomic_12 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap, compare_view.select<VL, 1>());

    res_atomic_13 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare, pred);
    res_atomic_14 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(), compare);

    res_atomic_15 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_atomic_16 = slm_atomic_update<atomic_op::cmpxchg>(
        offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());

    // Expect LSC for short.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto compare = simd<int16_t, VL>(VL, 1);
      auto swap = compare * 2;

      // CHECK: call <16 x i32> @llvm.genx.lsc.xatomic.slm.v16i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x i32> {{[^)]+}}, <16 x i32> {{[^)]+}}, i32 0, <16 x i32> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::cmpxchg, int16_t, VL>(offsets, swap,
                                                             compare);
    }

    // Expect LSC for int64_t.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int64_t);
      auto compare = simd<int64_t, VL>(VL, 1);
      auto swap = compare * 2;
      auto pred = simd_mask<VL>(1);

      // CHECK: call <16 x i64> @llvm.genx.lsc.xatomic.slm.v16i64.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 4, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x i64> {{[^)]+}}, <16 x i64> {{[^)]+}}, i32 0, <16 x i64> undef)
      auto res_slm_atomic_0 = slm_atomic_update<atomic_op::cmpxchg, int64_t>(
          offsets, swap, compare, pred);
    }

    // Expect LSC for FP types.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int64_t);
      auto compare = simd<float, VL>(VL, 1);
      auto swap = compare * 2;
      auto pred = simd_mask<VL>(1);

      // CHECK: call <16 x float> @llvm.genx.lsc.xatomic.slm.v16f32.v16i1.v16i32(<16 x i1> {{[^)]+}} i8 23, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x float> {{[^)]+}}, <16 x float> {{[^)]+}}, i32 0, <16 x float> undef)
      auto res_slm_atomic_0 = slm_atomic_update<atomic_op::fcmpxchg, float>(
          offsets, swap, compare, pred);
    }
  }

  // Test with local accessor.
  // Zero operand atomic.
  // CHECK-COUNT-6: call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
  {
    auto res_slm_atomic_1 =
        atomic_update<atomic_op::inc, int>(local_acc, offsets, pred);
    auto res_slm_atomic_2 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets);
    auto res_slm_atomic_3 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets_view, pred);
    auto res_slm_atomic_4 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets_view);
    auto res_slm_atomic_5 = atomic_update<atomic_op::inc, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), pred);
    auto res_slm_atomic_6 = atomic_update<atomic_op::inc, int, VL>(
        local_acc, offsets_view.select<VL, 1>());

    // Expect LSC for short.
    {
      using LocalAccType = sycl::local_accessor<int16_t, 1>;
      LocalAccType *local_acc = nullptr;
      // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
      auto res_slm_atomic_1 =
          atomic_update<atomic_op::inc, int16_t>(*local_acc, offsets);
    }
  }
  // One operand atomic.
  {
    // CHECK-COUNT-26: call <4 x i32> @llvm.genx.dword.atomic.add.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_slm_atomic_1 =
        atomic_update<atomic_op::add>(local_acc, offsets, add, pred);
    auto res_slm_atomic_2 =
        atomic_update<atomic_op::add>(local_acc, offsets, add);
    auto res_slm_atomic_3 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets, add_view, pred);
    auto res_slm_atomic_4 =
        atomic_update<atomic_op::add, int, VL>(local_acc, offsets, add_view);
    auto res_slm_atomic_5 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view, add, pred);
    auto res_slm_atomic_6 =
        atomic_update<atomic_op::add, int, VL>(local_acc, offsets_view, add);
    auto res_slm_atomic_7 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view, add_view, pred);
    auto res_slm_atomic_8 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view, add_view);
    res_slm_atomic_3 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets, add_view.select<VL, 1>(), pred);
    res_slm_atomic_4 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets, add_view.select<VL, 1>());
    res_slm_atomic_5 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), add, pred);
    res_slm_atomic_6 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), add);
    res_slm_atomic_7 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>(),
        pred);
    res_slm_atomic_8 = atomic_update<atomic_op::add, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>());

    res_slm_atomic_3 =
        atomic_update<atomic_op::add>(local_acc, offsets, add_view, pred);
    res_slm_atomic_4 =
        atomic_update<atomic_op::add>(local_acc, offsets, add_view);
    res_slm_atomic_5 =
        atomic_update<atomic_op::add>(local_acc, offsets_view, add, pred);
    res_slm_atomic_6 =
        atomic_update<atomic_op::add>(local_acc, offsets_view, add);
    res_slm_atomic_7 =
        atomic_update<atomic_op::add>(local_acc, offsets_view, add_view, pred);
    res_slm_atomic_8 =
        atomic_update<atomic_op::add>(local_acc, offsets_view, add_view);
    res_slm_atomic_3 = atomic_update<atomic_op::add>(
        local_acc, offsets, add_view.select<VL, 1>(), pred);
    res_slm_atomic_4 = atomic_update<atomic_op::add>(local_acc, offsets,
                                                     add_view.select<VL, 1>());
    res_slm_atomic_5 = atomic_update<atomic_op::add>(
        local_acc, offsets_view.select<VL, 1>(), add, pred);
    res_slm_atomic_6 = atomic_update<atomic_op::add>(
        local_acc, offsets_view.select<VL, 1>(), add);
    res_slm_atomic_7 =
        atomic_update<atomic_op::add>(local_acc, offsets_view.select<VL, 1>(),
                                      add_view.select<VL, 1>(), pred);
    res_slm_atomic_8 = atomic_update<atomic_op::add>(
        local_acc, offsets_view.select<VL, 1>(), add_view.select<VL, 1>());

    // Expect LSC for short.
    {
      using LocalAccType = sycl::local_accessor<int16_t, 1>;
      LocalAccType *local_acc = nullptr;
      simd<int16_t, VL> add = simd<int16_t, VL>(1) * sizeof(int);
      // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
      auto res_slm_atomic_1 =
          atomic_update<atomic_op::add, int16_t>(*local_acc, offsets, add);
    }
  }
  // Two operand atomic.
  {
    // CHECK-COUNT-58: call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_slm_atomic_1 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap, compare, pred);
    auto res_slm_atomic_2 =
        atomic_update<atomic_op::cmpxchg>(local_acc, offsets, swap, compare);
    auto res_slm_atomic_3 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap, compare_view, pred);
    auto res_slm_atomic_4 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap, compare_view);
    auto res_slm_atomic_5 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view, compare, pred);
    auto res_slm_atomic_6 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view, compare);
    auto res_slm_atomic_7 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view, compare_view, pred);
    auto res_slm_atomic_8 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view, compare_view);
    auto res_slm_atomic_9 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap, compare, pred);
    auto res_slm_atomic_10 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap, compare);
    auto res_slm_atomic_11 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap, compare_view, pred);
    auto res_slm_atomic_12 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap, compare_view);
    auto res_slm_atomic_13 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap_view, compare, pred);
    auto res_slm_atomic_14 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap_view, compare);
    auto res_slm_atomic_15 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap_view, compare_view, pred);
    auto res_slm_atomic_16 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view, swap_view, compare_view);
    res_slm_atomic_3 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap, compare_view.select<VL, 1>(), pred);
    res_slm_atomic_4 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap, compare_view.select<VL, 1>());
    res_slm_atomic_5 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view.select<VL, 1>(), compare, pred);
    res_slm_atomic_6 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view.select<VL, 1>(), compare);
    res_slm_atomic_7 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_8 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets, swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());
    res_slm_atomic_9 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap, compare, pred);
    res_slm_atomic_10 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap, compare);
    res_slm_atomic_11 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap,
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_12 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap,
        compare_view.select<VL, 1>());
    res_slm_atomic_13 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare, pred);
    res_slm_atomic_14 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare);
    res_slm_atomic_15 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_16 = atomic_update<atomic_op::cmpxchg, int, VL>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());

    res_slm_atomic_3 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap, compare_view, pred);
    res_slm_atomic_4 = atomic_update<atomic_op::cmpxchg>(local_acc, offsets,
                                                         swap, compare_view);
    res_slm_atomic_5 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view, compare, pred);
    res_slm_atomic_6 = atomic_update<atomic_op::cmpxchg>(local_acc, offsets,
                                                         swap_view, compare);
    res_slm_atomic_7 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view, compare_view, pred);
    res_slm_atomic_8 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view, compare_view);
    res_slm_atomic_9 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap, compare, pred);
    res_slm_atomic_10 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap, compare);
    res_slm_atomic_11 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap, compare_view, pred);
    res_slm_atomic_12 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap, compare_view);
    res_slm_atomic_13 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap_view, compare, pred);
    res_slm_atomic_14 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap_view, compare);
    res_slm_atomic_15 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap_view, compare_view, pred);
    res_slm_atomic_16 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view, swap_view, compare_view);
    res_slm_atomic_3 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap, compare_view.select<VL, 1>(), pred);
    res_slm_atomic_4 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap, compare_view.select<VL, 1>());
    res_slm_atomic_5 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view.select<VL, 1>(), compare, pred);
    res_slm_atomic_6 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view.select<VL, 1>(), compare);
    res_slm_atomic_7 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_8 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets, swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());
    res_slm_atomic_9 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap, compare, pred);
    res_slm_atomic_10 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap, compare);
    res_slm_atomic_11 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap,
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_12 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap,
        compare_view.select<VL, 1>());
    res_slm_atomic_13 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare, pred);
    res_slm_atomic_14 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare);
    res_slm_atomic_15 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>(), pred);
    res_slm_atomic_16 = atomic_update<atomic_op::cmpxchg>(
        local_acc, offsets_view.select<VL, 1>(), swap_view.select<VL, 1>(),
        compare_view.select<VL, 1>());

    // Expect LSC for short.
    {
      using LocalAccType = sycl::local_accessor<int16_t, 1>;
      LocalAccType *local_acc = nullptr;
      auto compare = simd<int16_t, VL>(VL, 1);
      auto swap = compare * 2;
      // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.slm.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
      auto res_slm_atomic_1 = atomic_update<atomic_op::cmpxchg, int16_t, VL>(
          *local_acc, offsets, swap, compare);
    }
  }
}