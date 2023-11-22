// RUN: %clangxx -O0 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// Checks ESIMD memory functions accepting compile time properties.
// NOTE: must be run in -O0, as optimizer optimizes away some of the code.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;
using LocalAccType = sycl::local_accessor<double, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_block_load(AccType &,
                                                       LocalAccType &, float *,
                                                       int byte_offset32,
                                                       size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_atomic_update(AccType &, float *, int byte_offset32, size_t byte_offset64);

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_block_store(AccType &, float *, int byte_offset32, size_t byte_offset64);

class EsimdFunctor {
public:
  AccType acc;
  LocalAccType local_acc;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    test_block_load(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_atomic_update(acc, ptr, byte_offset32, byte_offset64);
    test_block_store(acc, ptr, byte_offset32, byte_offset64);
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar(AccType &acc, LocalAccType &local_acc, float *ptr, int byte_offset32,
         size_t byte_offset64) {
  EsimdFunctor esimdf{acc, local_acc, ptr, byte_offset32, byte_offset64};
  kernel<class kernel_esimd>(esimdf);
}

// CHECK-LABEL: define {{.*}} @_Z15test_block_load{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_block_load(AccType &acc, LocalAccType &local_acc, float *ptrf,
                int byte_offset32, size_t byte_offset64) {
  properties props_a{cache_hint_L1<cache_hint::streaming>,
                     cache_hint_L2<cache_hint::cached>, alignment<16>};
  static_assert(props_a.has_property<cache_hint_L1_key>(), "Missing L1 hint");
  static_assert(props_a.has_property<cache_hint_L2_key>(), "Missing L2 hint");
  static_assert(props_a.has_property<alignment_key>(), "Missing alignment");

  properties props_b{cache_hint_L1<cache_hint::cached>,
                     cache_hint_L2<cache_hint::uncached>, alignment<8>};
  static_assert(props_b.has_property<cache_hint_L1_key>(), "Missing L1 hint");
  static_assert(props_b.has_property<cache_hint_L2_key>(), "Missing L2 hint");
  static_assert(props_b.has_property<alignment_key>(), "Missing alignment");

  properties props_c{alignment<4>};
  properties props_c16{alignment<16>};
  static_assert(props_c.has_property<alignment_key>(), "Missing alignment");

  properties store_props_a{cache_hint_L1<cache_hint::uncached>,
                           cache_hint_L2<cache_hint::uncached>, alignment<16>};

  properties store_props_b{alignment<16>};

  properties store_props_c{cache_hint_L1<cache_hint::write_back>,
                           cache_hint_L2<cache_hint::write_back>,
                           alignment<32>};

  properties store_props_d{alignment<8>};

  constexpr int N = 4;
  simd<float, N> pass_thru = 1;
  simd<int, N> pass_thrui = 1;
  const int *ptri = reinterpret_cast<const int *>(ptrf);
  const int8_t *ptrb = reinterpret_cast<const int8_t *>(ptrf);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d1 = block_load<float, N>(ptrf, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d2 = block_load<int, N>(ptri, byte_offset32, props_a);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d3 = block_load<float, N>(ptrf, byte_offset64, props_b);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  simd_mask<1> mask = 1;
  auto d4 = block_load<float, N>(ptrf, mask, props_a);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.merge.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x float> {{[^)]+}})
  auto d5 = block_load<float, N>(ptrf, mask, pass_thru, props_b);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d6 = block_load<float, N>(ptrf, byte_offset32, mask, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d7 = block_load<int, N>(ptri, byte_offset64, mask, props_b);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  auto d8 = block_load<int, N>(ptri, byte_offset32, mask, pass_thrui, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  auto d9 = block_load<int, N>(ptri, byte_offset64, mask, pass_thru, props_b);

  // Now try block_load without cache hints and using the mask to verify
  // svm/legacy code-gen. Also, intentially use vector lengths that are
  // not power-of-two because only svm/legacy block_load supports
  // non-power-of-two vector lengths now.

  // CHECK: load <5 x float>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x1 = block_load<float, 5>(ptrf, props_c);

  // CHECK: load <6 x i32>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x2 = block_load<int, 6>(ptri, byte_offset32, props_c);

  // CHECK: load <7 x float>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x3 = block_load<float, 7>(ptrf, byte_offset64, props_c);

  // Verify ACCESSOR-based block_load.

  // CHECK: call <4 x float> @llvm.genx.lsc.load.bti.v4f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  auto a1 = block_load<float, N>(acc, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.bti.v4i32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  auto a2 = block_load<int, N>(acc, byte_offset32, props_a);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.bti.v4f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}})
  auto a3 = block_load<float, N>(acc, byte_offset64, props_b);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.merge.bti.v4f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x float> {{[^)]+}})
  auto a4 = block_load<float, N>(acc, mask, props_a);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.merge.bti.v4f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x float> {{[^)]+}})
  auto a5 = block_load<float, N>(acc, mask, pass_thru, props_b);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.merge.bti.v4f32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x float> {{[^)]+}})
  auto a6 = block_load<float, N>(acc, byte_offset32, mask, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  auto a7 = block_load<int, N>(acc, byte_offset64, mask, props_b);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  auto a8 = block_load<int, N>(acc, byte_offset32, mask, pass_thru, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.bti.v4i32.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  auto a9 = block_load<int, N>(acc, byte_offset64, mask, pass_thrui, props_b);

  // Now try block_load without cache hints and using the mask to verify
  // svm/legacy code-gen. Also, intentially use vector lengths that are
  // not power-of-two because only svm/legacy block_load supports
  // non-power-of-two vector lengths now.

  // CHECK: call <4 x float> @llvm.genx.oword.ld.v4f32(i32 0, i32 {{[^)]+}}, i32 {{[^)]+}})
  auto z1 = block_load<float, 4>(acc, props_c);

  // CHECK: call <8 x i32> @llvm.genx.oword.ld.unaligned.v8i32(i32 0, i32 {{[^)]+}}, i32 {{[^)]+}})
  auto z2 = block_load<int, 8>(acc, byte_offset32, props_c);

  // CHECK: call <16 x float> @llvm.genx.oword.ld.v16f32(i32 0, i32 {{[^)]+}}, i32 {{[^)]+}})
  auto z3 = block_load<float, 16>(acc, byte_offset64, props_c16);

  // Now try SLM block_load() with and without cache hints that are ignored.

  // CHECK: load <11 x double>, ptr addrspace(3) {{[^)]+}}, align 16
  auto slm_bl1 = slm_block_load<double, 11>(byte_offset32, props_a);

  // CHECK: call <8 x double> @llvm.genx.lsc.load.slm.v8f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 5, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0)
  auto slm_bl2 = slm_block_load<double, 8>(byte_offset32, mask, props_c16);

  simd<double, 8> pass_thrud = 2.0;
  // CHECK: call <8 x double> @llvm.genx.lsc.load.merge.slm.v8f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 5, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0, <8 x double> {{[^)]+}})
  auto slm_bl3 =
      slm_block_load<double, 8>(byte_offset32, mask, pass_thrud, props_c16);

  // Now try block_load() accepting local accessor.

  // CHECK: load <2 x double>, ptr addrspace(3) {{[^)]+}}, align 16
  auto lacc_bl1 = block_load<double, 2>(local_acc, props_a);

  // CHECK: load <5 x double>, ptr addrspace(3) {{[^)]+}}, align 8
  auto lacc_bl2 = block_load<double, 5>(local_acc, byte_offset32, props_b);

  // CHECK: call <8 x double> @llvm.genx.lsc.load.slm.v8f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 5, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0)
  auto lacc_bl3 = block_load<double, 8>(local_acc, mask, props_a);

  // CHECK: call <16 x double> @llvm.genx.lsc.load.merge.slm.v16f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 6, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0, <16 x double> {{[^)]+}})
  simd<double, 16> pass_thrud16 = 2.0;
  auto lacc_bl4 =
      block_load<double, 16>(local_acc, mask, pass_thrud16, props_b);

  // CHECK: call <32 x double> @llvm.genx.lsc.load.slm.v32f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 7, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0)
  auto lacc_bl5 =
      block_load<double, 32>(local_acc, byte_offset32, mask, props_a);

  // CHECK: call <4 x double> @llvm.genx.lsc.load.merge.slm.v4f64.v1i1.v1i32(<1 x i1> {{[^)]+}}, i8 0, i8 0, i8 0, i16 1, i32 0, i8 4, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, i32 0, <4 x double> {{[^)]+}})
  simd<double, 4> pass_thrud4 = 2.0;
  auto lacc_bl6 = block_load<double, 4>(local_acc, byte_offset32, mask,
                                        pass_thrud4, props_a);

  // Check the default/assumed alignment when the alignment property is
  // not specified explicitly.
  // TODO: Extend this kind of tests:
  //   {usm, acc, local_acc, slm} x {byte, word, dword, qword}.

  // CHECK: load <16 x i8>, ptr addrspace(4) {{[^)]+}}, align 4
  auto align_check1 = block_load<int8_t, 16>(ptrb);
}

// CHECK-LABEL: define {{.*}} @_Z18test_atomic_update{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_atomic_update(AccType &acc, float *ptrf, int byte_offset32,
                   size_t byte_offset64) {
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

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_3 =
        atomic_update<atomic_op::inc, int>(ptr, offsets_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}} i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_4 =
        atomic_update<atomic_op::inc, int, VL>(ptr, offsets_view, props_a);

    // atomic_upate without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_5 =
        atomic_update<atomic_op::inc, int, VL>(ptr, offsets, pred);

    // atomic_upate without cache hints and mask:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.inc.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_6 = atomic_update<atomic_op::inc, int, VL>(ptr, offsets);

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

    // Accessor

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_0 =
        atomic_update<atomic_op::inc, int>(acc, offsets, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 1, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_1 =
        atomic_update<atomic_op::inc, int>(acc, offsets, pred, props_b);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_2 =
        atomic_update<atomic_op::inc, int>(acc, offsets, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_3 =
        atomic_update<atomic_op::inc, int>(acc, offsets_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 8, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> undef, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_4 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets_view, props_a);

    // atomic_upate without cache hints:
    // CHECK:  call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_5 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets, pred);

    // atomic_upate without cache hints and mask:
    // CHECK: call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_acc_6 =
        atomic_update<atomic_op::inc, int, VL>(acc, offsets);

    // Try the atomic_update without cache hints, but with non-standard
    // vector length to check that LSC atomic is generated.
    // CHECK: call <5 x i32> @llvm.genx.lsc.xatomic.bti.v5i32.v5i1.v5i32(<5 x i1> {{[^)]+}}, i8 8, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <5 x i32> {{[^)]+}}, <5 x i32> undef, <5 x i32> undef, i32 {{[^)]+}}, <5 x i32> undef)
    {
      constexpr int VL = 5;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      auto pred = simd_mask<VL>(1);
      auto atomic_res_acc =
          atomic_update<atomic_op::inc, int, VL>(acc, offsets, pred);
    }
  }

  // Test atomic update with one operand.
  {
    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_0 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_1 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_2 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets, add_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_3 =
        atomic_update<atomic_op::add, int, VL>(ptr, offsets, add_view, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_4 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_5 =
        atomic_update<atomic_op::add, int, VL>(ptr, offsets_view, add, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_6 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 0, <4 x i32> undef)
    auto res_atomic_7 = atomic_update<atomic_op::add, int, VL>(
        ptr, offsets_view, add_view, props_a);

    // atomic_upate without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.add.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_8 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, pred);
  }

  // Test atomic update with one operand.
  {
    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_1 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets, swap, compare, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_2 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets, swap, compare, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_3 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_4 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap, compare_view, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_5 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_6 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_7 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_8 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets, swap_view, compare_view, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_9 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view, swap, compare, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_10 = atomic_update<atomic_op::cmpxchg, int>(
        ptr, offsets_view, swap, compare, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_11 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap, compare_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_12 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap, compare_view, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_13 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_14 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_15 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare_view, pred, props_a);

    // CHECK: call <4 x i32> @llvm.genx.lsc.xatomic.stateless.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0, <4 x i32> undef)
    auto res_atomic_16 = atomic_update<atomic_op::cmpxchg, int, VL>(
        ptr, offsets_view, swap_view, compare_view, props_a);

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
  }
}

// CHECK-LABEL: define {{.*}} @_Z16test_block_store{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_block_store(AccType &acc,
                                                        float *ptrf,
                                                        int byte_offset32,
                                                        size_t byte_offset64) {
  // Test USM block store
  constexpr int N = 4;
  properties store_props_a{cache_hint_L1<cache_hint::uncached>,
                           cache_hint_L2<cache_hint::uncached>, alignment<16>};

  properties store_props_b{alignment<16>};

  properties store_props_c{cache_hint_L1<cache_hint::write_back>,
                           cache_hint_L2<cache_hint::write_back>,
                           alignment<32>};

  properties store_props_d{alignment<8>};
  simd<float, N> vals = 1;
  simd<int, N> valsi = 1;
  int *ptri = reinterpret_cast<int *>(ptrf);
  simd_mask<1> mask = 1;

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, vals, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  block_store(ptri, byte_offset32, valsi, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, byte_offset64, vals, store_props_c);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, vals, mask, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> %{{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  block_store(ptri, byte_offset64, valsi, mask, store_props_c);

  // Test SVM/legacy USM block store

  // CHECK: store <4 x float> {{[^)]+}}, ptr addrspace(4) {{[^)]+}}, align 16
  block_store(ptrf, vals, store_props_b);

  // CHECK: store <4 x float> {{[^)]+}}, ptr addrspace(4) {{[^)]+}}, align 8
  block_store(ptrf, vals, store_props_d);

  // CHECK: store <4 x float> {{[^)]+}}, ptr addrspace(4) {{[^)]+}}, align 16
  block_store(ptrf, byte_offset32, vals, store_props_b);

  // Test accessor block store

  // CHECK: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 {{[^)]+}})
  block_store(acc, vals, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}})
  block_store(acc, byte_offset32, valsi, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 {{[^)]+}})
  block_store(acc, byte_offset64, vals, store_props_c);

  // CHECK: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 {{[^)]+}})
  block_store(acc, vals, mask, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.bti.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}})
  block_store(acc, byte_offset64, valsi, mask, store_props_c);

  // Test accessor SVM/legacy block store

  // CHECK: call void @llvm.genx.oword.st.v4f32(i32 {{[^)]+}}, i32 {{[^)]+}}, <4 x float> {{[^)]+}})
  block_store(acc, vals, store_props_b);

  // CHECK: call void @llvm.genx.oword.st.v4i32(i32 {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}})
  block_store(acc, byte_offset32, valsi, store_props_b);
}
