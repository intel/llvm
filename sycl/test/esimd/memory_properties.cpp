// RUN: %clangxx -O0 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// Checks ESIMD memory functions accepting compile time properties.
// NOTE: must be run in -O0, as optimizer optimizes away some of the code.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

using AccType = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write>;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
foo(AccType &, float *, int byte_offset32, size_t byte_offset64);

class EsimdFunctor {
public:
  AccType acc;
  float *ptr;
  int byte_offset32;
  size_t byte_offset64;
  void operator()() __attribute__((sycl_explicit_simd)) {
    foo(acc, ptr, byte_offset32, byte_offset64);
  }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar(AccType &acc, float *ptr, int byte_offset32, size_t byte_offset64) {
  EsimdFunctor esimdf{acc, ptr, byte_offset32, byte_offset64};
  kernel<class kernel_esimd>(esimdf);
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL void
foo(AccType &acc, float *ptrf, int byte_offset32, size_t byte_offset64) {
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
  static_assert(props_c.has_property<alignment_key>(), "Missing alignment");

  constexpr int N = 4;
  simd<int, N> old_values = 1;
  const int *ptri = reinterpret_cast<const int *>(ptrf);

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
  auto d5 = block_load<float, N>(ptrf, mask, old_values, props_b);

  // CHECK: call <4 x float> @llvm.genx.lsc.load.stateless.v4f32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d6 = block_load<float, N>(ptrf, byte_offset32, mask, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0)
  auto d7 = block_load<int, N>(ptri, byte_offset64, mask, props_b);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 5, i8 2, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  auto d8 = block_load<int, N>(ptri, byte_offset32, mask, old_values, props_a);

  // CHECK: call <4 x i32> @llvm.genx.lsc.load.merge.stateless.v4i32.v1i1.v1i64(<1 x i1> {{[^)]+}}, i8 0, i8 2, i8 1, i16 1, i32 0, i8 3, i8 4, i8 2, i8 0, <1 x i64> {{[^)]+}}, i32 0, <4 x i32> {{[^)]+}})
  auto d9 = block_load<int, N>(ptri, byte_offset64, mask, old_values, props_b);

  // Now repeat all the calls used above but in without cache hints to
  // verify svm/legacy code-gen. Also, intentially use vector lengths that are
  // not power-of-two because only svm/legacy block_load supports
  // non-power-of-two vector lengths now.

  // CHECK: load <5 x float>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x1 = block_load<float, 5>(ptrf, props_c);

  // CHECK: load <6 x i32>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x2 = block_load<int, 6>(ptri, byte_offset32, props_c);

  // CHECK: load <7 x float>, ptr addrspace(4) {{[^)]+}}, align 4
  auto x3 = block_load<float, 7>(ptrf, byte_offset64, props_c);
}
