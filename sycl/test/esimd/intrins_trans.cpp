// RUN: %clangxx -O0 -fsycl -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// Checks ESIMD intrinsic translation.
// NOTE: must be run in -O0, as optimizer optimizes away some of the code

#include <CL/sycl.hpp>
#include <CL/sycl/detail/image_ocl_types.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

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

SYCL_ESIMD_FUNCTION SYCL_EXTERNAL simd<float, 16> foo() {
  // CHECK-LABEL: @_Z3foov
  constexpr int VL = 32;
  uint32_t *ptr = 0;

  int x = 0, y = 0, z = 0;

  simd<uint32_t, VL> v1(0, x + z);
  simd<uint64_t, VL> offsets(0, y);
  simd<uintptr_t, VL> v_addr(reinterpret_cast<uintptr_t>(ptr));
  simd<ushort, VL> pred;
  v_addr += offsets;

  __esimd_flat_atomic0<EsimdAtomicOpType::ATOMIC_INC, uint32_t, VL>(
      v_addr.data(), pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.inc.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  __esimd_flat_atomic1<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, VL>(
      v_addr.data(), v1, pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.add.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  __esimd_flat_atomic2<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, VL>(
      v_addr.data(), v1, v1, pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.cmpxchg.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  simd<uint32_t, VL> v00 =
      __esimd_flat_block_read_unaligned<uint32_t, VL>(addr);
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.block.ld.unaligned.v32i32.i64(i64 %{{[0-9a-zA-Z_.]+}})
  __esimd_flat_block_write<uint32_t, VL>(addr, v00.data());
  // CHECK: call void @llvm.genx.svm.block.st.i64.v32i32(i64 %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})

  simd<uint32_t, VL> v01 =
      __esimd_flat_read<uint32_t, VL>(v_addr.data(), 0, pred.data());
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.gather.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)

  __esimd_flat_write<uint32_t, VL>(v_addr.data(), v01.data(), 0, pred.data());
  // CHECK: call void @llvm.genx.svm.scatter.v32i1.v32i64.v32i32(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})

  simd<short, 16> mina(0, 1);
  simd<short, 16> minc(5);
  minc = __esimd_smin<short, 16>(mina.data(), minc.data());
  // CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x i16> @llvm.genx.smin.v16i16.v16i16(<16 x i16> %{{[0-9a-zA-Z_.]+}}, <16 x i16> %{{[0-9a-zA-Z_.]+}})

  simd<float, 1> diva(2.f);
  simd<float, 1> divb(1.f);
  diva = __esimd_div_ieee<1>(diva.data(), divb.data());
  // CHECK:  %{{[0-9a-zA-Z_.]+}} = call <1 x float> @llvm.genx.ieee.div.v1f32(<1 x float>  %{{[0-9a-zA-Z_.]+}}, <1 x float>  %{{[0-9a-zA-Z_.]+}})

  simd<float, 16> a(0.1f);
  simd<float, 8> b = __esimd_rdregion<float, 16, 8, 0, 8, 1>(a.data(), 0);
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x float> @llvm.genx.rdregionf.v8f32.v16f32.i16(<16 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0)

  simd<float, 16> c(0.0f);

  using PH = cl::sycl::access::placeholder;

  cl::sycl::accessor<cl::sycl::cl_int4, 2, cl::sycl::access::mode::read,
                     cl::sycl::access::target::image, PH::false_t>
      pA;
  cl::sycl::accessor<cl::sycl::cl_int4, 2, cl::sycl::access::mode::write,
                     cl::sycl::access::target::image, PH::false_t>
      pB;

  auto d = __esimd_wrregion<float, 16 /*ret size*/, 8 /*write size*/,
                            0 /*vstride*/, 8 /*row width*/, 1 /*hstride*/>(
      c.data() /*dst*/, b.data() /*src*/, 0 /*offset*/);
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.wrregionf.v16f32.v8f32.i16.v8i1(<16 x float> %{{[0-9a-zA-Z_.]+}}, <8 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)

  simd<int, 32> va;
  va = media_block_load<int, 4, 8>(pA, x, y);
  // CHECK: %[[SI0:[0-9a-zA-Z_.]+]] = ptrtoint %opencl.image2d_ro_t addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.media.ld.v32i32(i32 0, i32 %[[SI0]], i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})

  simd<int, 32> vb = va + 1;
  media_block_store<int, 4, 8>(pB, x, y, vb);
  // CHECK: %[[SI2:[0-9a-zA-Z_.]+]] = ptrtoint %opencl.image2d_wo_t addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
  // CHECK: call void @llvm.genx.media.st.v32i32(i32 0, i32 %[[SI2]], i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})

  auto ee = __esimd_vload<int, 16>((detail::vector_type_t<int, 16> *)(&vg));
  // CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.vload.v16i32.p0v16i32(<16 x i32>* {{.*}})
  __esimd_vstore<int, 32>(&vc, va.data());
  // CHECK: store <32 x i32>  %{{[0-9a-zA-Z_.]+}}, <32 x i32> addrspace(4)* {{.*}}

  {
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::global_buffer>
        acc;
    simd<uint32_t, 8> offsets = 1;
    simd<uint16_t, 8> pred{1, 0, 1, 0, 1, 0, 1, 0};

    // 4-byte element gather
    simd<int, 8> v = gather<int, 8>(acc, offsets, 100);
    // CHECK: %[[SI3:[0-9a-zA-Z_.]+]] = ptrtoint i32 addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
    // CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x i32> @llvm.genx.gather.scaled2.v8i32.v8i32(i32 2, i16 0, i32 %[[SI3]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})

    // 4-byte element scatter
    scatter<int, 8>(acc, v, offsets, 100, pred);
    // CHECK: %[[SI4:[0-9a-zA-Z_.]+]] = ptrtoint i32 addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
    // CHECK: call void @llvm.genx.scatter.scaled.v8i1.v8i32.v8i32(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 2, i16 0, i32 %[[SI4]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})

    // 1-byte element gather
    simd<unsigned char, 8> v1 = gather<unsigned char, 8>(acc, offsets, 100);
    // CHECK: %[[SI5:[0-9a-zA-Z_.]+]] = ptrtoint i32 addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
    // CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x i32> @llvm.genx.gather.scaled2.v8i32.v8i32(i32 0, i16 0, i32 %[[SI5]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})

    // 1-byte element scatter
    scatter<unsigned char, 8>(acc, v1, offsets, 100, pred);
    // CHECK: %[[SI6:[0-9a-zA-Z_.]+]] = ptrtoint i32 addrspace(1)* %{{[0-9a-zA-Z_.]+}} to i32
    // CHECK: call void @llvm.genx.scatter.scaled.v8i1.v8i32.v8i32(<8 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, i16 0, i32 %[[SI6]], i32 %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}})
  }
  return d;
}
