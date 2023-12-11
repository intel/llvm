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

template <typename T> decltype(auto) test_acos(T x) { return sycl::acos(x); }

#define TEST_ACOS(TYPE)                                                        \
  template decltype(auto) SYCL_EXTERNAL test_acos<TYPE>(TYPE);
TEST_ALL_TYPES(TEST_ACOS)

decltype(auto) SYCL_EXTERNAL test_swizzle(float_v4 x) {
  return sycl::acos(x.swizzle<1, 0>());
}

// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define weak_odr dso_local spir_func noundef float @_Z9test_acosIfEDcT_(float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z16__spirv_ocl_acosf(float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef double @_Z9test_acosIdEDcT_(double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z16__spirv_ocl_acosd(double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V16detail9half_impl4halfEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z16__spirv_ocl_acosDF16_(half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIfLi1EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z16__spirv_ocl_acosf(float noundef %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIfLi2EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_acosDv2_f(<2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIfLi3EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x float> @_Z16__spirv_ocl_acosDv3_f(<3 x float> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x float> %{{.*}}, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIfLi4EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z16__spirv_ocl_acosDv4_f(<4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIdLi1EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z16__spirv_ocl_acosd(double noundef %{{.*}})
// CHECK-NEXT:   store double %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIdLi2EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x double> @_Z16__spirv_ocl_acosDv2_d(<2 x double> noundef %{{.*}})
// CHECK-NEXT:   store <2 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecIdLi4EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x double> @_Z16__spirv_ocl_acosDv4_d(<4 x double> noundef %{{.*}})
// CHECK-NEXT:   store <4 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z16__spirv_ocl_acosDF16_(half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x half>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x half> @_Z16__spirv_ocl_acosDv2_DF16_(<2 x half> noundef %{{.*}})
// CHECK-NEXT:   store <2 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x half>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x half> @_Z16__spirv_ocl_acosDv4_DF16_(<4 x half> noundef %{{.*}})
// CHECK-NEXT:   store <4 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z12test_swizzleN4sycl3_V13vecIfLi4EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_acosDv2_f(<2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define weak_odr dso_local noundef float @_Z9test_acosIfEDcT_(float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V111__acos_implEf(float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef double @_Z9test_acosIdEDcT_(double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef double @_ZN4sycl3_V111__acos_implEd(double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z9test_acosIN4sycl3_V16detail9half_impl4halfEEDcT_(i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V111__acos_implENS0_6detail9half_impl4halfE(i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local float @_Z9test_acosIN4sycl3_V13vecIfLi1EEEEDcT_(float %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call float @_ZN4sycl3_V111__acos_implENS0_3vecIfLi1EEE(float %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local <2 x float> @_Z9test_acosIN4sycl3_V13vecIfLi2EEEEDcT_(<2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__acos_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z9test_acosIN4sycl3_V13vecIfLi3EEEEDcT_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__acos_implENS0_3vecIfLi3EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z9test_acosIN4sycl3_V13vecIfLi4EEEEDcT_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__acos_implENS0_3vecIfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local double @_Z9test_acosIN4sycl3_V13vecIdLi1EEEEDcT_(double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call double @_ZN4sycl3_V111__acos_implENS0_3vecIdLi1EEE(double %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { double, double } @_Z9test_acosIN4sycl3_V13vecIdLi2EEEEDcT_(double %{{.*}}, double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { double, double } @_ZN4sycl3_V111__acos_implENS0_3vecIdLi2EEE(double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret { double, double } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z9test_acosIN4sycl3_V13vecIdLi4EEEEDcT_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V111__acos_implENS0_3vecIdLi4EEE(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_(i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V111__acos_implENS0_3vecINS0_6detail9half_impl4halfELi1EEE(i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_(i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V111__acos_implENS0_3vecINS0_6detail9half_impl4halfELi2EEE(i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z9test_acosIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_(i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V111__acos_implENS0_3vecINS0_6detail9half_impl4halfELi4EEE(i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z12test_swizzleN4sycl3_V13vecIfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__acos_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
