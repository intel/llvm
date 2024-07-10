// RUN: %clangxx %clang_O0 -fsycl -fsycl-device-only -fno-sycl-esimd-force-stateless-mem -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=false -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATEFUL

// RUN: %clangxx %clang_O0 -fsycl -fsycl-device-only -fsycl-esimd-force-stateless-mem -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -lower-esimd-force-stateless-mem=true -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes=CHECK,CHECK-STATELESS

// Checks ESIMD intrinsic translation with opaque pointers.
// NOTE: must be run with %clang_O0, as optimizer optimizes away some of the
// code

#include <sycl/detail/image_ocl_types.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::intel::esimd;

ESIMD_PRIVATE
detail::vector_type_t<int, 32> vc;
ESIMD_PRIVATE ESIMD_REGISTER(192) simd<int, 16> vg;

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> foo();

class EsimdFunctor {
public:
  void operator()() __attribute__((sycl_explicit_simd)) { foo(); }
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  EsimdFunctor esimdf;
  kernel<class kernel_esimd>(esimdf);
}

// TODO
// 1. __esimd* intrinsic translation tests from
//   llvm\test\SYCLLowerIR\esimd_lower_intrins.ll should be refactored and
//   moved here, as the form below is much easier to maintain with the same
//   level of testing strength
// 2. Test cases above should be refactored not to use user-level APIs like
//   gather and use __esimd* calls instead.
template <class T, int N> using vec = typename simd<T, N>::raw_vector_type;

template <int N> using mask = typename simd_mask<N>::raw_vector_type;

SYCL_EXTERNAL void use(const vec<float, 8> &x) SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL void use(const vec<int, 8> &x) SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL void use(const vec<unsigned char, 8> &x) SYCL_ESIMD_FUNCTION;

SYCL_EXTERNAL vec<float, 8> get8f() SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL vec<int, 8> get8i() SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL vec<uint64_t, 8> get8ui64() SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL vec<unsigned short, 8> get8ui16() SYCL_ESIMD_FUNCTION;
SYCL_EXTERNAL vec<unsigned char, 8> get8ui8() SYCL_ESIMD_FUNCTION;

SYCL_EXTERNAL void
test_mem_intrins(int *addr, const vec<float, 8> &xf,
                 const vec<float, 8> &xi) SYCL_ESIMD_FUNCTION {
  {
    constexpr SurfaceIndex si = 0;
    vec<float, 8> x = __esimd_oword_ld_unaligned<float, 8>(si, 0);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.oword.ld.unaligned.v8f32(i32 0, i32 0, i32 0)
    use(x);
  }
  {
    constexpr SurfaceIndex si = 0;
    vec<float, 8> x = __esimd_oword_ld<float, 8>(si, 0);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.oword.ld.v8f32(i32 0, i32 0, i32 0)
    use(x);
  }
  {
    constexpr SurfaceIndex si = 0;
    __esimd_oword_st<float, 8>(si, 0, get8f());
    // CHECK-LABEL: call void @llvm.genx.oword.st.v8f32(i32 0, i32 0, <8 x float> %{{[a-zA-Z0-9.]+}})
  }
  {
    using VecT = typename simd<int, 8>::raw_vector_type;
    VecT *vec_addr = reinterpret_cast<VecT *>(addr);
    vec<int, 8> x = __esimd_svm_block_ld<int, 8, 4>(vec_addr);
    // CHECK-LABEL: load <8 x i32>, ptr addrspace(4) %{{[a-zA-Z0-9.]+}}, align 4
    use(x);
  }
  {
    using VecT = typename simd<int, 8>::raw_vector_type;
    VecT *vec_addr = reinterpret_cast<VecT *>(addr);
    vec<int, 8> x = __esimd_svm_block_ld<int, 8, 32>(vec_addr);
    // CHECK-LABEL: load <8 x i32>, ptr addrspace(4) %{{[a-zA-Z0-9.]+}}, align 32
    use(x);
  }
  {
    using VecT = typename simd<int, 8>::raw_vector_type;
    VecT *vec_addr = reinterpret_cast<VecT *>(addr);
    __esimd_svm_block_st<int, 8, 32>(vec_addr, get8i());
    // CHECK-LABEL: store <8 x i32> %{{[a-zA-Z0-9.]+}}, ptr addrspace(4) %{{[a-zA-Z0-9.]+}}, align 32
  }
  {
    uint32_t offset = 128;
    vec<int, 8> x = __esimd_slm_block_ld<int, 8, 32>(offset);
    // CHECK: %[[VAR_OFF1:[0-9a-zA-Z_.]+]] = inttoptr i32 %{{[a-zA-Z0-9.]+}} to ptr addrspace(3)
    // CHECK-NEXT: load <8 x i32>, ptr addrspace(3) %[[VAR_OFF1]], align 32
    use(x);
  }
  {
    uint32_t offset = 256;
    __esimd_slm_block_st<int, 8, 4>(offset, get8i());
    // CHECK: %[[VAR_OFF2:[0-9a-zA-Z_.]+]] = inttoptr i32 %{{[a-zA-Z0-9.]+}} to ptr addrspace(3)
    // CHECK-NEXT: store <8 x i32> %{{[a-zA-Z0-9.]+}}, ptr addrspace(3) %[[VAR_OFF2]], align 4
  }
  {
    auto x = __esimd_svm_gather<unsigned char, 8>(get8ui64(), get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x i8> @llvm.genx.svm.gather.v8i8.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, i32 0, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x i8> undef)
    use(x);
  }
  {
    __esimd_svm_scatter<unsigned char, 8>(get8ui64(), get8ui8(), get8ui16());
    // CHECK-LABEL: call void @llvm.genx.svm.scatter.v8i1.v8i64.v8i8(<8 x i1> %{{[a-zA-Z0-9.]+}}, i32 0, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x i8> %{{[a-zA-Z0-9.]+}})
  }
  {
    auto x =
        __esimd_svm_atomic0<atomic_op::inc, int, 8>(get8ui64(), get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x i32> @llvm.genx.svm.atomic.inc.v8i32.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x i32> undef)
    use(x);
  }
  {
    vec<float, 8> src0 = get8f();
    auto x = __esimd_svm_atomic1<atomic_op::fmin, float, 8>(get8ui64(), src0,
                                                            get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.svm.atomic.fmin.v8f32.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x float> %{{[a-zA-Z0-9.]+}}, <8 x float> undef)
    use(x);
  }
  {
    vec<float, 8> src0 = get8f();
    vec<float, 8> src1 = get8f();
    auto x = __esimd_svm_atomic2<atomic_op::fcmpwr, float, 8>(get8ui64(), src0,
                                                              src1, get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.svm.atomic.fcmpwr.v8f32.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x float> %{{[a-zA-Z0-9.]+}}, <8 x float> %{{[a-zA-Z0-9.]+}}, <8 x float> undef)
    use(x);
  }
  {
    vec<int, 8> src0 = get8i();
    auto x = __esimd_svm_atomic1<atomic_op::smin, int, 8>(get8ui64(), src0,
                                                          get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x i32> @llvm.genx.svm.atomic.imin.v8i32.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x i32> %{{[a-zA-Z0-9.]+}}, <8 x i32> undef)
    use(x);
  }
  {
    vec<int, 8> src0 = get8i();
    auto x = __esimd_svm_atomic1<atomic_op::smax, int, 8>(get8ui64(), src0,
                                                          get8ui16());
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x i32> @llvm.genx.svm.atomic.imax.v8i32.v8i1.v8i64(<8 x i1> %{{[a-zA-Z0-9.]+}}, <8 x i64> %{{[a-zA-Z0-9.]+}}, <8 x i32> %{{[a-zA-Z0-9.]+}}, <8 x i32> undef)
    use(x);
  }
  {
    constexpr SurfaceIndex si = 0;
    vec<float, 8> x =
        __esimd_media_ld<float, 2, 4, 0, SurfaceIndex, 0, 4>(si, 0, 0);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.media.ld.v8f32(i32 0, i32 0, i32 0, i32 4, i32 0, i32 0)
    use(x);
  }
  {
    constexpr SurfaceIndex si = 0;
    vec<float, 8> x = get8f();
    __esimd_media_st<float, 2, 4, 0, SurfaceIndex, 0, 4>(si, 0, 0, x);
    // CHECK-LABEL: call void @llvm.genx.media.st.v8f32(i32 0, i32 0, i32 0, i32 4, i32 0, i32 0, <8 x float> %{{[a-zA-Z0-9.]+}})
  }
}

SYCL_EXTERNAL void test_math_intrins() SYCL_ESIMD_FUNCTION {
  {
    vec<float, 8> x0 = get8f();
    vec<float, 8> x1 = get8f();
    auto y = __esimd_ieee_div<float, 8>(x0, x1);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.ieee.div.v8f32(<8 x float> %{{[a-zA-Z0-9.]+}}, <8 x float> %{{[a-zA-Z0-9.]+}})
    use(y);
  }
  {
    vec<float, 8> x = get8f();
    auto y = __esimd_ieee_sqrt<float, 8>(x);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x float> @llvm.genx.ieee.sqrt.v8f32(<8 x float> %{{[a-zA-Z0-9.]+}})
    use(y);
  }
  {
    vec<int, 8> x0 = get8i();
    vec<int, 8> x1 = get8i();
    vec<int, 8> x2 = get8i();
    auto res = __esimd_bfn<0xff, int, 8>(x0, x1, x2);
    // CHECK-LABEL: %{{[a-zA-Z0-9.]+}} = call <8 x i32> @llvm.genx.bfn.v8i32.v8i32(<8 x i32> %{{[a-zA-Z0-9.]+}}, <8 x i32> %{{[a-zA-Z0-9.]+}}, <8 x i32> %{{[a-zA-Z0-9.]+}}, i8 -1)
    use(res);
  }
}

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> foo() {
  // CHECK-LABEL: @_Z3foov
  constexpr int VL = 32;
  uint32_t *ptr = 0;
  using VecT = typename simd<uint32_t, VL>::raw_vector_type;
  VecT *vec_ptr = 0;

  int x = 0, y = 0, z = 0;

  simd<uint32_t, VL> v1(0, x + z);
  simd<uint64_t, VL> offsets(0, y);
  simd<uintptr_t, VL> v_addr(reinterpret_cast<uintptr_t>(ptr));
  simd_mask<VL> pred;
  v_addr += offsets;

  __esimd_svm_atomic0<atomic_op::inc, uint32_t, VL>(v_addr.data(), pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.inc.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  __esimd_svm_atomic1<atomic_op::add, uint32_t, VL>(v_addr.data(), v1.data(),
                                                    pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.add.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  __esimd_svm_atomic2<atomic_op::cmpxchg, uint32_t, VL>(
      v_addr.data(), v1.data(), v1.data(), pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.cmpxchg.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  simd<uint32_t, VL> v00 = __esimd_svm_block_ld<uint32_t, VL, 4>(vec_ptr);
  // CHECK: %[[VAR1:[0-9a-zA-Z_.]+]] = load <32 x i32>, ptr addrspace(4) %{{[a-zA-Z0-9.]+}}, align 4
  __esimd_svm_block_st<uint32_t, VL, 128>(vec_ptr, v00.data());
  // CHECK-NEXT: store <32 x i32> %[[VAR1]], ptr addrspace(4) %{{[a-zA-Z0-9.]+}}, align 128

  simd<uint32_t, VL> v01 =
      __esimd_svm_gather<uint32_t, VL>(v_addr.data(), pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.gather.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  __esimd_svm_scatter<uint32_t, VL>(v_addr.data(), v01.data(), pred.data());
  // CHECK: call void @llvm.genx.svm.scatter.v32i1.v32i64.v32i32(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})

  simd<short, 16> mina(0, 1);
  simd<short, 16> minc(5);
  minc = __esimd_smin<short, 16>(mina.data(), minc.data());
  // CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x i16> @llvm.genx.smin.v16i16.v16i16(<16 x i16> %{{[0-9a-zA-Z_.]+}}, <16 x i16> %{{[0-9a-zA-Z_.]+}})

  simd<float, 1> diva(2.f);
  simd<float, 1> divb(1.f);
  diva = __esimd_ieee_div<float, 1>(diva.data(), divb.data());
  // CHECK:  %{{[0-9a-zA-Z_.]+}} = call <1 x float> @llvm.genx.ieee.div.v1f32(<1 x float>  %{{[0-9a-zA-Z_.]+}}, <1 x float>  %{{[0-9a-zA-Z_.]+}})

  simd<float, 16> a(0.1f);
  simd<float, 8> b = __esimd_rdregion<float, 16, 8, 0, 8, 1>(a.data(), 0);
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x float> @llvm.genx.rdregionf.v8f32.v16f32.i16(<16 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0)

  simd<float, 16> c(0.0f);

  using PH = sycl::access::placeholder;

  sycl::accessor<sycl::vec<sycl::opencl::cl_int, 4>, 2,
                 sycl::access::mode::read, sycl::access::target::image,
                 PH::false_t>
      pA;
  sycl::accessor<sycl::vec<sycl::opencl::cl_int, 4>, 2,
                 sycl::access::mode::write, sycl::access::target::image,
                 PH::false_t>
      pB;

  auto d = __esimd_wrregion<float, 16 /*ret size*/, 8 /*write size*/,
                            0 /*vstride*/, 8 /*row width*/, 1 /*hstride*/>(
      c.data() /*dst*/, b.data() /*src*/, 0 /*offset*/);
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.wrregionf.v16f32.v8f32.i16.v8i1(<16 x float> %{{[0-9a-zA-Z_.]+}}, <8 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)

  simd<int, 32> va;
  va = media_block_load<int, 4, 8>(pA, x, y);
  // CHECK: %[[SI0_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 0) %{{[0-9a-zA-Z_.]+}})
  // CHECK: store i32 %[[SI0_VAL]], ptr addrspace(4) %[[SI0_ADDR:[0-9a-zA-Z_.]+]]
  // CHECK: %[[SI0:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI0_ADDR]]
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.media.ld.v32i32(i32 0, i32 %[[SI0]], i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})

  simd<int, 32> vb = va + 1;
  media_block_store<int, 4, 8>(pB, x, y, vb);
  // CHECK: %[[SI2_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 1) %{{[0-9a-zA-Z_.]+}})
  // CHECK: store i32 %[[SI2_VAL]], ptr addrspace(4) %[[SI2_ADDR:[0-9a-zA-Z_.]+]]
  // CHECK: %[[SI2:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI2_ADDR]]
  // CHECK: call void @llvm.genx.media.st.v32i32(i32 0, i32 %[[SI2]], i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})

  auto ee = __esimd_vload<int, 16>((detail::vector_type_t<int, 16> *)(&vg));
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.vload.v16i32.p0(ptr {{.*}})
  __esimd_vstore<int, 32>(&vc, va.data());
  // CHECK: store <32 x i32>  %{{[0-9a-zA-Z_.]+}}, ptr addrspace(4) {{.*}}

  {
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::device>
        acc;
    simd<uint32_t, 8> offsets = 1;
    simd_mask<8> pred({1, 0, 1, 0, 1, 0, 1, 0});

    // 4-byte element gather
    simd<int, 8> v = gather<int, 8>(acc, offsets, 100);
    // CHECK-STATEFUL: %[[SI3_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(ptr addrspace(1) noundef %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATEFUL: store i32 %[[SI3_VAL]], ptr addrspace(4) %[[SI3_ADDR:[0-9a-zA-Z_.]+]]
    // CHECK-STATEFUL: %[[SI3:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI3_ADDR]]
    // CHECK-STATEFUL: call <8 x i32> @llvm.genx.gather.masked.scaled2.v8i32.v8i32.v8i1(i32 2, i16 0, i32 %[[SI3]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i1> %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATELESS: call <8 x i32> @llvm.genx.svm.gather.v8i32.v8i1.v8i64(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <8 x i64> %{{[0-9a-zA-Z_.]+}}, <8 x i32> undef)

    // 4-byte element scatter
    scatter<int, 8>(acc, offsets, v, 100, pred);
    // CHECK-STATEFUL: %[[SI4_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(ptr addrspace(1) noundef %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATEFUL: store i32 %[[SI4_VAL]], ptr addrspace(4) %[[SI4_ADDR:[0-9a-zA-Z_.]+]]
    // CHECK-STATEFUL: %[[SI4:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI4_ADDR]]
    // CHECK-STATEFUL: call void @llvm.genx.scatter.scaled.v8i1.v8i32.v8i32(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 2, i16 0, i32 %[[SI4]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATELESS: call void @llvm.genx.svm.scatter.v8i1.v8i64.v8i32(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <8 x i64> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})

    // 1-byte element gather: same code with and without mask
    simd<unsigned char, 8> v1 = gather<unsigned char, 8>(acc, offsets, 100);
    // CHECK-STATEFUL: %[[SI5_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(ptr addrspace(1) noundef %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATEFUL: store i32 %[[SI5_VAL]], ptr addrspace(4) %[[SI5_ADDR:[0-9a-zA-Z_.]+]]
    // CHECK-STATEFUL: %[[SI5:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI5_ADDR]]
    // CHECK-STATEFUL: call <8 x i32> @llvm.genx.gather.masked.scaled2.v8i32.v8i32.v8i1(i32 0, i16 0, i32 %[[SI5]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i1> %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATELESS: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <8 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i8> undef)

    // 1-byte element gather using the mask
    v1 = gather<unsigned char, 8>(acc, offsets, 100, pred);
    // CHECK-STATEFUL: call <8 x i32> @llvm.genx.gather.masked.scaled2.v8i32.v8i32.v8i1(i32 0, i16 0, i32 {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i1> {{[^)]+}})
    // CHECK-STATELESS: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{[^)]+}}, i32 0, <8 x i64> {{[^)]+}}, <32 x i8> undef)

    // 1-byte element gather using the mask - the mask is signed, which may
    // expose different issues/conflicts in gather API.
    simd<int32_t, 8> ioffsets = 1;
    v1 = gather<unsigned char, 8>(acc, ioffsets, 0, pred);
    // CHECK-STATEFUL: call <8 x i32> @llvm.genx.gather.masked.scaled2.v8i32.v8i32.v8i1(i32 0, i16 0, i32 {{[^)]+}}, i32 {{[^)]+}}, <8 x i32> {{[^)]+}}, <8 x i1> {{[^)]+}})
    // CHECK-STATELESS: call <32 x i8> @llvm.genx.svm.gather.v32i8.v8i1.v8i64(<8 x i1> {{[^)]+}}, i32 0, <8 x i64> {{[^)]+}}, <32 x i8> undef)

    // 1-byte element scatter
    scatter<unsigned char, 8>(acc, offsets, v1, 100, pred);
    // CHECK-STATEFUL: %[[SI6_VAL:[0-9a-zA-Z_.]+]] = call spir_func noundef i32 @_Z21__spirv_ConvertPtrToU{{.*}}(ptr addrspace(1) noundef %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATEFUL: store i32 %[[SI6_VAL]], ptr addrspace(4) %[[SI6_ADDR:[0-9a-zA-Z_.]+]]
    // CHECK-STATEFUL: %[[SI6:[0-9a-zA-Z_.]+]] = load i32, ptr addrspace(4) %[[SI6_ADDR]]
    // CHECK-STATEFUL: call void @llvm.genx.scatter.scaled.v8i1.v8i32.v8i32(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, i16 0, i32 %[[SI6]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})
    // CHECK-STATELESS: call void @llvm.genx.svm.scatter.v8i1.v8i64.v32i8(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <8 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i8> %{{[0-9a-zA-Z_.]+}})
  }
  __esimd_fence(fence_mask::global_coherent_fence);
  // CHECK: call void @llvm.genx.fence(i8 1)
  __esimd_fence(fence_mask::l2_flush_instructions);
  // CHECK: call void @llvm.genx.fence(i8 2)
  __esimd_fence(fence_mask::l2_flush_texture_data);
  // CHECK: call void @llvm.genx.fence(i8 4)
  __esimd_fence(fence_mask::l2_flush_constant_data);
  // CHECK: call void @llvm.genx.fence(i8 8)
  __esimd_fence(fence_mask::l2_flush_rw_data);
  // CHECK: call void @llvm.genx.fence(i8 16)
  __esimd_fence(fence_mask::local_barrier);
  // CHECK: call void @llvm.genx.fence(i8 32)
  __esimd_fence(fence_mask::l1_flush_ro_data);
  // CHECK: call void @llvm.genx.fence(i8 64)

  return d;
}
