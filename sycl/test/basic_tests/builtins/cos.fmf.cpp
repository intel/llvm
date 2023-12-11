// RUN: %clangxx -fpreview-breaking-changes -fsycl -O2 -ffast-math -S -emit-llvm -o - %s | FileCheck %s
#include <sycl/sycl.hpp>

using float_v1 = sycl::vec<float, 1>;
using float_v2 = sycl::vec<float, 2>;
using float_v3 = sycl::vec<float, 3>;
using float_v4 = sycl::vec<float, 4>;
using double_v1 = sycl::vec<double, 1>;
using double_v2 = sycl::vec<double, 2>;
using double_v4 = sycl::vec<double, 4>;
using half_v1 = sycl::vec<sycl::half, 1>;
using half_v2 = sycl::vec<sycl::half, 2>;
using half_v4 = sycl::vec<sycl::half, 4>;

#define TEST_ALL_TYPES(TEST_MACRO)                                             \
  TEST_MACRO(float)                                                            \
  TEST_MACRO(double)                                                           \
  TEST_MACRO(sycl::half)                                                       \
  TEST_MACRO(float_v1)                                                         \
  TEST_MACRO(float_v2)                                                         \
  TEST_MACRO(float_v3)                                                         \
  TEST_MACRO(float_v4)

#define TEST(FUNC, MACRO)                                                      \
  template <typename T> decltype(auto) test_##FUNC(T x) {                      \
    return sycl::FUNC(x);                                                      \
  }                                                                            \
  TEST_ALL_TYPES(MACRO)                                                        \
  decltype(auto) SYCL_EXTERNAL test_##FUNC##_swizzle(float_v4 x) {             \
    return sycl::FUNC(x.swizzle<1, 0>());                                      \
  }

#define TEST_COS(TYPE)                                                         \
  template decltype(auto) SYCL_EXTERNAL test_cos<TYPE>(TYPE);
TEST(cos, TEST_COS)

// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define weak_odr dso_local spir_func noundef nofpclass(nan inf) float @_Z8test_cosIfEDcT_(float noundef nofpclass(nan inf) %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) float @_Z22__spirv_ocl_native_cosf(float noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef nofpclass(nan inf) double @_Z8test_cosIdEDcT_(double noundef nofpclass(nan inf) %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) double @_Z15__spirv_ocl_cosd(double noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z8test_cosIN4sycl3_V16detail9half_impl4halfEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) half @_Z15__spirv_ocl_cosDF16_(half noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z8test_cosIN4sycl3_V13vecIfLi1EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) float @_Z22__spirv_ocl_native_cosf(float noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z8test_cosIN4sycl3_V13vecIfLi2EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z22__spirv_ocl_native_cosDv2_f(<2 x float> noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z8test_cosIN4sycl3_V13vecIfLi3EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) <3 x float> @_Z22__spirv_ocl_native_cosDv3_f(<3 x float> noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x float> %{{.*}}, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z8test_cosIN4sycl3_V13vecIfLi4EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) <4 x float> @_Z22__spirv_ocl_native_cosDv4_f(<4 x float> noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_cos_swizzleN4sycl3_V13vecIfLi4EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call fast spir_func noundef nofpclass(nan inf) <2 x float> @_Z22__spirv_ocl_native_cosDv2_f(<2 x float> noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define weak_odr dso_local noundef nofpclass(nan inf) float @_Z8test_cosIfEDcT_(float noundef nofpclass(nan inf) %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast noundef nofpclass(nan inf) float @_ZN4sycl3_V110__cos_implEf(float noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef nofpclass(nan inf) double @_Z8test_cosIdEDcT_(double noundef nofpclass(nan inf) %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast noundef nofpclass(nan inf) double @_ZN4sycl3_V110__cos_implEd(double noundef nofpclass(nan inf) %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z8test_cosIN4sycl3_V16detail9half_impl4halfEEDcT_(i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V110__cos_implENS0_6detail9half_impl4halfE(i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local float @_Z8test_cosIN4sycl3_V13vecIfLi1EEEEDcT_(float %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast float @_ZN4sycl3_V110__cos_implENS0_3vecIfLi1EEE(float %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local <2 x float> @_Z8test_cosIN4sycl3_V13vecIfLi2EEEEDcT_(<2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call fast <2 x float> @_ZN4sycl3_V110__cos_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z8test_cosIN4sycl3_V13vecIfLi3EEEEDcT_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V110__cos_implENS0_3vecIfLi3EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z8test_cosIN4sycl3_V13vecIfLi4EEEEDcT_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V110__cos_implENS0_3vecIfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z16test_cos_swizzleN4sycl3_V13vecIfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = tail call fast <2 x float> @_ZN4sycl3_V110__cos_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
