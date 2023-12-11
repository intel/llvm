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
using LocalAccTypeInt = sycl::local_accessor<int, 1>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void test_block_load(AccType &,
                                                       LocalAccType &, float *,
                                                       int byte_offset32,
                                                       size_t byte_offset64);
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_atomic_update(AccType &, LocalAccTypeInt &, float *, int byte_offset32,
                   size_t byte_offset64);

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_block_store(AccType &, LocalAccType &local_acc, float *, int byte_offset32,
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
    test_block_load(acc, local_acc, ptr, byte_offset32, byte_offset64);
    test_atomic_update(acc, local_acc_int, ptr, byte_offset32, byte_offset64);
    test_block_store(acc, local_acc, ptr, byte_offset32, byte_offset64);
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
    // USM

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

    // atomic_update without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.svm.atomic.add.v4i32.v4i1.v4i64(<4 x i1> {{[^)]+}}, <4 x i64> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_8 =
        atomic_update<atomic_op::add, int>(ptr, offsets, add, pred);

    // Accessors

    // CHECK-COUNT-8: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 12, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef, i32 {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_9 =
        atomic_update<atomic_op::add, int>(acc, offsets, add, pred, props_a);

    auto res_atomic_10 =
        atomic_update<atomic_op::add, int>(acc, offsets, add, props_a);

    auto res_atomic_11 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets, add_view, pred, props_a);

    auto res_atomic_12 =
        atomic_update<atomic_op::add, int, VL>(acc, offsets, add_view, props_a);

    auto res_atomic_13 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add, pred, props_a);

    auto res_atomic_14 =
        atomic_update<atomic_op::add, int, VL>(acc, offsets_view, add, props_a);

    auto res_atomic_15 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add_view, pred, props_a);

    auto res_atomic_16 = atomic_update<atomic_op::add, int, VL>(
        acc, offsets_view, add_view, props_a);

    // atomic_update without cache hints:
    // CHECK: call <4 x i32> @llvm.genx.dword.atomic.add.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_17 =
        atomic_update<atomic_op::add, int>(acc, offsets, add, pred);
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

    // Accessors

    //CHECK-COUNT-16: call <4 x i32> @llvm.genx.lsc.xatomic.bti.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i8 18, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> undef)
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

    {
      constexpr int VL = 8;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int);
      simd<float, VL> swap = simd<float, VL>(1) * sizeof(int);
      auto compare = swap * 2;
      auto pred = simd_mask<VL>(1);
      // Do not pass the properties.
      // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 23, i8 0, i8 0, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}} <8 x i32> {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> undef)
      auto atomic_res0 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          acc, offsets, swap, compare, pred);
      // Now with cache hints.
      // CHECK: call <8 x i32> @llvm.genx.lsc.xatomic.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^)]+}}, i8 23, i8 1, i8 3, i16 1, i32 0, i8 3, i8 1, i8 1, i8 0, <8 x i32> {{[^)]+}}, <8 x i32> {{[^)]+}} <8 x i32> {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> undef)
      auto atomic_res1 = atomic_update<atomic_op::fcmpxchg, float, VL>(
          acc, offsets, swap, compare, pred, props_a);
    }

    // atomic_update without cache hints
    // CHECK: call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    auto res_atomic_33 = atomic_update<atomic_op::cmpxchg, int, VL>(
        acc, offsets, swap, compare, pred);
  }

  // Test slm_atomic_update without operands.
  {
    // CHECK-COUNT-4: call <4 x i32> @llvm.genx.dword.atomic.dec.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
    {
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::dec, int>(offsets, pred);
      auto res_slm_atomic_1 = slm_atomic_update<atomic_op::dec, int>(offsets);
      auto res_slm_atomic_2 =
          slm_atomic_update<atomic_op::dec, int, VL>(offsets_view, pred);
      auto res_slm_atomic_3 =
          slm_atomic_update<atomic_op::dec, int, VL>(offsets_view);
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
    // CHECK-COUNT-8: call <4 x i32> @llvm.genx.dword.atomic.add.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
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
    }

    // Expect LSC for short.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(int16_t);
      auto pred = simd_mask<VL>(1);
      simd<int16_t, VL> add = simd<int16_t, VL>(1) * sizeof(int);

      // CHECK: call <16 x i32> @llvm.genx.lsc.xatomic.slm.v16i32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i8 12, i8 0, i8 0, i16 1, i32 0, i8 6, i8 1, i8 1, i8 0, <16 x i32> {{[^)]+}}, <16 x i32> {{[^)]+}}, <16 x i32> undef, i32 0, <16 x i32> undef)
      auto res_slm_atomic_0 =
          slm_atomic_update<atomic_op::add, int16_t>(offsets, add, pred);
    }
    // Expect DWORD for fmin.
    {
      constexpr int VL = 16;
      simd<uint32_t, VL> offsets = simd<uint32_t, VL>(1) * sizeof(float);
      auto pred = simd_mask<VL>(1);
      simd<float, VL> min = simd<float, VL>(1) * sizeof(int);

      // CHECK: call <16 x float> @llvm.genx.dword.atomic.fmin.v16f32.v16i1.v16i32(<16 x i1> {{[^)]+}}, i32 {{[^)]+}}, <16 x i32> {{[^)]+}}, <16 x float> {{[^)]+}}, <16 x float> undef)
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
    // CHECK-COUNT-16: call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
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
  }

  // Test with local accessor.
  // Zero operand atomic.
  // CHECK-COUNT-4: call <4 x i32> @llvm.genx.dword.atomic.inc.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
  {
    auto res_slm_atomic_1 =
        atomic_update<atomic_op::inc, int>(local_acc, offsets, pred);
    auto res_slm_atomic_2 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets);
    auto res_slm_atomic_3 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets_view, pred);
    auto res_slm_atomic_4 =
        atomic_update<atomic_op::inc, int, VL>(local_acc, offsets_view);
  }
  // One operand atomic.
  {
    // CHECK-COUNT-8: call <4 x i32> @llvm.genx.dword.atomic.add.v4i32.v4i1.v4i32(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
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
  }
  // Two operand atomic.
  {
    // CHECK-COUNT-16: call <4 x i32> @llvm.genx.dword.atomic.cmpxchg.v4i32.v4i1(<4 x i1> {{[^)]+}}, i32 {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, <4 x i32> undef)
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
  }
}

// CHECK-LABEL: define {{.*}} @_Z16test_block_store{{.*}}
SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
test_block_store(AccType &acc, LocalAccType &local_acc, float *ptrf,
                 int byte_offset32, size_t byte_offset64) {
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
  auto view = vals.select<N, 1>();
  auto viewi = valsi.select<N, 1>();
  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, vals, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  block_store(ptri, byte_offset32, valsi, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, byte_offset64, vals, store_props_c);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 1, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store(ptrf, vals, mask, store_props_a);

  // CHECK: call void @llvm.genx.lsc.store.stateless.v1i1.v1i64.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 3, i8 3, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
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

  // Now try SLM block_store() with and without cache hints that are ignored.

  // CHECK-COUNT-2: store <4 x float> {{[^)]+}}, ptr addrspace(3) {{[^)]+}}, align 16
  slm_block_store<float, N>(byte_offset32, vals, store_props_b);
  slm_block_store<float, N>(byte_offset32, view, store_props_b);

  // CHECK-COUNT-2: store <4 x float> {{[^)]+}}, ptr addrspace(3) {{[^)]+}}, align 16
  slm_block_store<float, N>(byte_offset32, vals, store_props_a);
  slm_block_store<float, N>(byte_offset32, view, store_props_a);

  // Now try SLM block_store() with a predicate.

  // CHECK-COUNT-2: call void @llvm.genx.lsc.store.slm.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  slm_block_store<int, N>(byte_offset32, valsi, mask, store_props_b);
  slm_block_store<int, N>(byte_offset32, viewi, mask, store_props_b);

  // Now try block_store() accepting local accessor.

  // CHECK-COUNT-2: store <4 x float> {{[^)]+}}, ptr addrspace(3) {{[^)]+}}, align 8
  block_store<float, N>(local_acc, vals, store_props_d);
  block_store<float, N>(local_acc, view, store_props_d);

  // CHECK-COUNT-2: store <4 x i32> {{[^)]+}}, ptr addrspace(3) {{[^)]+}}, align 8
  block_store<int, N>(local_acc, byte_offset32, valsi, store_props_d);
  block_store<int, N>(local_acc, byte_offset32, viewi, store_props_d);

  // CHECK-COUNT-2: call void @llvm.genx.lsc.store.slm.v1i1.v1i32.v4f32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x float> {{[^)]+}}, i32 0)
  block_store<float, N>(local_acc, vals, mask, store_props_a);
  block_store<float, N>(local_acc, view, mask, store_props_a);

  // CHECK-COUNT-2: call void @llvm.genx.lsc.store.slm.v1i1.v1i32.v4i32(<1 x i1> {{[^)]+}}, i8 4, i8 0, i8 0, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i32> {{[^)]+}}, <4 x i32> {{[^)]+}}, i32 0)
  block_store<int, N>(local_acc, byte_offset32, valsi, mask, store_props_c);
  block_store<int, N>(local_acc, byte_offset32, viewi, mask, store_props_c);
}
