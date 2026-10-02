// RUN: %clangxx -O0 -fsycl -c -fsycl-device-only -Xclang -emit-llvm %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -force-disable-esimd-opt -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_0.esimd.ll

// Checks that the typed-surface (image) RGBA gather/scatter/prefetch APIs are
// lowered to the expected GenX intrinsics. Distinct channel masks and cache
// hints are used so that the operand order of each intrinsic is verified.

#include <sycl/accessor_image.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
namespace iexp = sycl::ext::intel::experimental::esimd;

int main() {
  queue Q;
  uint32_t Data[64 * 4] = {};
  image<2> Img(Data, image_channel_order::rgba,
               image_channel_type::unsigned_int32, range<2>{8, 8});
  Q.submit([&](handler &CGH) {
    auto In = Img.get_access<uint4, access::mode::read>(CGH);
    auto Out = Img.get_access<uint4, access::mode::write>(CGH);
    CGH.single_task([=]() SYCL_ESIMD_KERNEL {
      simd<uint32_t, 16> U16(0, 1), V16 = 0;
      simd<uint32_t, 8> U8(0, 1), V8 = 0;
      simd<uint32_t, 32> U32(0, 1), V32 = 0;

      // CHECK: call <64 x i32> @llvm.genx.gather4.typed.v64i32.v16i1.v16i32(i32 15, <16 x i1> {{[^,]+}}, i32 {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <64 x i32> undef)
      simd<uint32_t, 64> Abgr = gather_rgba_typed<uint32_t, 16>(In, U16, V16);
      // CHECK: call <8 x float> @llvm.genx.gather4.typed.v8f32.v8i1.v8i32(i32 1, <8 x i1> {{[^,]+}}, i32 {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x float> undef)
      simd<float, 8> R =
          gather_rgba_typed<float, 8, rgba_channel_mask::R>(In, U8);
      // CHECK: call void @llvm.genx.scatter4.typed.v16i1.v16i32.v32i32(i32 3, <16 x i1> {{[^,]+}}, i32 {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <32 x i32> {{[^)]+}})
      scatter_rgba_typed<uint32_t, 16, rgba_channel_mask::GR>(
          Out, U16, V16, 0, Abgr.select<32, 1>(0));

      // CHECK: call <32 x i32> @llvm.genx.lsc.load.merge.quad.typed.bti.v32i32.v16i1.v16i32(<16 x i1> {{[^,]+}}, i8 2, i8 1, i8 9, i32 {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <16 x i32> {{[^,]+}}, <32 x i32> {{[^)]+}})
      simd<uint32_t, 32> Ar =
          iexp::lsc_gather_rgba_typed<uint32_t, 16, rgba_channel_mask::AR,
                                      cache_hint::cached, cache_hint::uncached>(
              In, U16, V16);
      // CHECK: call <8 x i32> @llvm.genx.lsc.load.merge.quad.typed.bti.v8i32.v8i1.v8i32(<8 x i1> {{[^,]+}}, i8 1, i8 2, i8 4, i32 {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^)]+}})
      simd<uint32_t, 8> B =
          iexp::lsc_gather_rgba_typed<uint32_t, 8, rgba_channel_mask::B,
                                      cache_hint::uncached, cache_hint::cached>(
              In, U8, V8, 0, 0, U8 < 4, simd<uint32_t, 8>(7));
      // CHECK: call void @llvm.genx.lsc.store.quad.typed.bti.v32i1.v32i32.v96i32(<32 x i1> {{[^,]+}}, i8 5, i8 3, i8 7, i32 {{[^,]+}}, <32 x i32> {{[^,]+}}, <32 x i32> {{[^,]+}}, <32 x i32> {{[^,]+}}, <32 x i32> {{[^,]+}}, <96 x i32> {{[^)]+}})
      iexp::lsc_scatter_rgba_typed<uint32_t, 32, rgba_channel_mask::BGR,
                                   cache_hint::streaming,
                                   cache_hint::write_back>(
          Out, U32, V32, 0, 0, simd<uint32_t, 96>(Ar[0]));
      // CHECK: call void @llvm.genx.lsc.prefetch.quad.typed.bti.v8i1.v8i32(<8 x i1> {{[^,]+}}, i8 5, i8 1, i8 3, i32 {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^,]+}}, <8 x i32> {{[^)]+}})
      iexp::lsc_prefetch_rgba_typed<uint32_t, 8, rgba_channel_mask::GR,
                                    cache_hint::streaming,
                                    cache_hint::uncached>(In, U8, V8);
      (void)R;
      (void)B;
    });
  });
  return 0;
}
