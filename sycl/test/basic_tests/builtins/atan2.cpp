// RUN: %clangxx -fpreview-breaking-changes -fsycl -O2 -S -emit-llvm -o - %s | FileCheck %s
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
  TEST_MACRO(float_v4)                                                         \
  TEST_MACRO(double_v1)                                                        \
  TEST_MACRO(double_v2)                                                        \
  TEST_MACRO(double_v4)                                                        \
  TEST_MACRO(half_v1)                                                          \
  TEST_MACRO(half_v2)                                                          \
  TEST_MACRO(half_v4)

template <typename T> decltype(auto) test_atan2(T x, T y) {
  return sycl::atan2(x, y);
}

#define TEST_ATAN2(TYPE)                                                       \
  template decltype(auto) SYCL_EXTERNAL test_atan2<TYPE>(TYPE, TYPE);
TEST_ALL_TYPES(TEST_ATAN2)

decltype(auto) SYCL_EXTERNAL test_swizzle(float_v4 x, float_v4) {
  return sycl::atan2(x.swizzle<1, 0>(), x.swizzle<2, 3>());
}
decltype(auto) SYCL_EXTERNAL test_vec_swizzle(sycl::vec<float, 2> x,
                                              sycl::vec<float, 4> y) {
  return sycl::atan2(x, y.swizzle<2, 3>());
}
decltype(auto) SYCL_EXTERNAL test_swizzle_vec(sycl::vec<float, 4> x,
                                              sycl::vec<float, 2> y) {
  return sycl::atan2(x.swizzle<2, 3>(), y);
}
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define weak_odr dso_local spir_func noundef float @_Z10test_atan2IfEDcT_S0_(float noundef %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z17__spirv_ocl_atan2ff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef double @_Z10test_atan2IdEDcT_S0_(double noundef %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z17__spirv_ocl_atan2dd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V16detail9half_impl4halfEEDcT_S5_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z17__spirv_ocl_atan2DF16_DF16_(half noundef %{{.*}}, half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIfLi1EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z17__spirv_ocl_atan2ff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIfLi2EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z17__spirv_ocl_atan2Dv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIfLi3EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x float> @_Z17__spirv_ocl_atan2Dv3_fS_(<3 x float> noundef %{{.*}}, <3 x float> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x float> %{{.*}}, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIfLi4EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z17__spirv_ocl_atan2Dv4_fS_(<4 x float> noundef %{{.*}}, <4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIdLi1EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z17__spirv_ocl_atan2dd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   store double %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIdLi2EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x double> @_Z17__spirv_ocl_atan2Dv2_dS_(<2 x double> noundef %{{.*}}, <2 x double> noundef %{{.*}})
// CHECK-NEXT:   store <2 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecIdLi4EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x double> @_Z17__spirv_ocl_atan2Dv4_dS_(<4 x double> noundef %{{.*}}, <4 x double> noundef %{{.*}})
// CHECK-NEXT:   store <4 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z17__spirv_ocl_atan2DF16_DF16_(half noundef %{{.*}}, half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x half>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <2 x half>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x half> @_Z17__spirv_ocl_atan2Dv2_DF16_S_(<2 x half> noundef %{{.*}}, <2 x half> noundef %{{.*}})
// CHECK-NEXT:   store <2 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x half>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x half>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x half> @_Z17__spirv_ocl_atan2Dv4_DF16_S_(<4 x half> noundef %{{.*}}, <4 x half> noundef %{{.*}})
// CHECK-NEXT:   store <4 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z12test_swizzleN4sycl3_V13vecIfLi4EEES2_(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readnone byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z17__spirv_ocl_atan2Dv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_vec_swizzleN4sycl3_V13vecIfLi2EEENS1_IfLi4EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z17__spirv_ocl_atan2Dv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_swizzle_vecN4sycl3_V13vecIfLi4EEENS1_IfLi2EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z17__spirv_ocl_atan2Dv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define weak_odr dso_local noundef float @_Z10test_atan2IfEDcT_S0_(float noundef %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V112__atan2_implEff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef double @_Z10test_atan2IdEDcT_S0_(double noundef %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef double @_ZN4sycl3_V112__atan2_implEdd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z10test_atan2IN4sycl3_V16detail9half_impl4halfEEDcT_S5_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V112__atan2_implENS0_6detail9half_impl4halfES3_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local float @_Z10test_atan2IN4sycl3_V13vecIfLi1EEEEDcT_S4_(float %{{.*}}, float %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call float @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi1EEES2_(float %{{.*}}, float %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local <2 x float> @_Z10test_atan2IN4sycl3_V13vecIfLi2EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z10test_atan2IN4sycl3_V13vecIfLi3EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi3EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z10test_atan2IN4sycl3_V13vecIfLi4EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local double @_Z10test_atan2IN4sycl3_V13vecIdLi1EEEEDcT_S4_(double %{{.*}}, double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call double @_ZN4sycl3_V112__atan2_implENS0_3vecIdLi1EEES2_(double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { double, double } @_Z10test_atan2IN4sycl3_V13vecIdLi2EEEEDcT_S4_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { double, double } @_ZN4sycl3_V112__atan2_implENS0_3vecIdLi2EEES2_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret { double, double } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z10test_atan2IN4sycl3_V13vecIdLi4EEEEDcT_S4_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V112__atan2_implENS0_3vecIdLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_S7_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V112__atan2_implENS0_3vecINS0_6detail9half_impl4halfELi1EEES5_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_S7_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V112__atan2_implENS0_3vecINS0_6detail9half_impl4halfELi2EEES5_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z10test_atan2IN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_S7_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V112__atan2_implENS0_3vecINS0_6detail9half_impl4halfELi4EEES5_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z12test_swizzleN4sycl3_V13vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z16test_vec_swizzleN4sycl3_V13vecIfLi2EEENS1_IfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z16test_swizzle_vecN4sycl3_V13vecIfLi4EEENS1_IfLi2EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__atan2_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
