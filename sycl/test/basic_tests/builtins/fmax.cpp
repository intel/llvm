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

template <typename T> decltype(auto) test_fmax(T x, T y) {
  return sycl::fmax(x, y);
}

#define TEST_FMAX(TYPE)                                                        \
  template decltype(auto) SYCL_EXTERNAL test_fmax<TYPE>(TYPE, TYPE);

TEST_ALL_TYPES(TEST_FMAX)

decltype(auto) SYCL_EXTERNAL test_fmax_swizzle(float_v4 x) {
  return sycl::fmax(x.swizzle<1, 0>(), x.swizzle<3, 2>());
}

decltype(auto) SYCL_EXTERNAL test_scalar_arg1(float_v4 x, float y) {
  return sycl::fmax(x, y);
}

decltype(auto) SYCL_EXTERNAL test_scalar_arg1(float_v4 x, double y) {
  return sycl::fmax(x, y);
}

decltype(auto) SYCL_EXTERNAL test_scalar_arg1(float_v4 x, sycl::half y) {
  return sycl::fmax(x, y);
}

decltype(auto) SYCL_EXTERNAL test_scalar_arg1(double_v4 x, float y) {
  return sycl::fmax(x, y);
}

decltype(auto) SYCL_EXTERNAL test_swizzle_scalar(float_v4 x, float y) {
  return sycl::fmax(x.swizzle<0, 2>(), y);
}

struct S {
  char padding[128];
  float v;
  operator float() const { return v; }
};

decltype(auto) SYCL_EXTERNAL test_marray_S(sycl::vec<float, 2> x, S y) {
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec_S(sycl::marray<float, 2> x, S y) {
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_swizzle_S(sycl::vec<float, 4> x, S y) {
  return sycl::fmax(x.swizzle<0, 3>(), y);
}

decltype(auto) SYCL_EXTERNAL test_marray_S_2(sycl::vec<double, 2> x, S y) {
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec_S_2(sycl::marray<double, 2> x, S y) {
  return sycl::fmax(x, y);
}
decltype(auto) SYCL_EXTERNAL test_vec_swizzle(sycl::vec<float, 2> x,
                                              sycl::vec<float, 4> y) {
  return sycl::fmax(x, y.swizzle<2, 3>());
}
decltype(auto) SYCL_EXTERNAL test_swizzle_vec(sycl::vec<float, 4> x,
                                              sycl::vec<float, 2> y) {
  return sycl::fmax(x.swizzle<2, 3>(), y);
}
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define weak_odr dso_local spir_func noundef float @_Z9test_fmaxIfEDcT_S0_(float noundef %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z16__spirv_ocl_fmaxff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef double @_Z9test_fmaxIdEDcT_S0_(double noundef %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z16__spirv_ocl_fmaxdd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V16detail9half_impl4halfEEDcT_S5_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z16__spirv_ocl_fmaxDF16_DF16_(half noundef %{{.*}}, half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIfLi1EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z16__spirv_ocl_fmaxff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIfLi2EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIfLi3EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x float> @_Z16__spirv_ocl_fmaxDv3_fS_(<3 x float> noundef %{{.*}}, <3 x float> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x float> %{{.*}}, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIfLi4EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z16__spirv_ocl_fmaxDv4_fS_(<4 x float> noundef %{{.*}}, <4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIdLi1EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef double @_Z16__spirv_ocl_fmaxdd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   store double %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIdLi2EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x double> @_Z16__spirv_ocl_fmaxDv2_dS_(<2 x double> noundef %{{.*}}, <2 x double> noundef %{{.*}})
// CHECK-NEXT:   store <2 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecIdLi4EEEEDcT_S4_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x double> @_Z16__spirv_ocl_fmaxDv4_dS_(<4 x double> noundef %{{.*}}, <4 x double> noundef %{{.*}})
// CHECK-NEXT:   store <4 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef half @_Z16__spirv_ocl_fmaxDF16_DF16_(half noundef %{{.*}}, half noundef %{{.*}})
// CHECK-NEXT:   store half %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x half>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <2 x half>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x half> @_Z16__spirv_ocl_fmaxDv2_DF16_S_(<2 x half> noundef %{{.*}}, <2 x half> noundef %{{.*}})
// CHECK-NEXT:   store <2 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_S7_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x half>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x half>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x half> @_Z16__spirv_ocl_fmaxDv4_DF16_S_(<4 x half> noundef %{{.*}}, <4 x half> noundef %{{.*}})
// CHECK-NEXT:   store <4 x half> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z17test_fmax_swizzleN4sycl3_V13vecIfLi4EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 3, i32 2>
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEEf(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z16__spirv_ocl_fmaxDv4_fS_(<4 x float> noundef %{{.*}}, <4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEEd(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = fptrunc double %{{.*}} to float
// CHECK-NEXT:   %{{.*}} = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z16__spirv_ocl_fmaxDv4_fS_(<4 x float> noundef %{{.*}}, <4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEENS0_6detail9half_impl4halfE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x float>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load half, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = fpext half %{{.*}} to float
// CHECK-NEXT:   %{{.*}} = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x float> @_Z16__spirv_ocl_fmaxDv4_fS_(<4 x float> noundef %{{.*}}, <4 x float> noundef %{{.*}})
// CHECK-NEXT:   store <4 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z16test_scalar_arg1N4sycl3_V13vecIdLi4EEEf(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 32 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 32 %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x double>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   %{{.*}} = insertelement <4 x double> poison, double %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x double> %{{.*}}, <4 x double> poison, <4 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x double> @_Z16__spirv_ocl_fmaxDv4_dS_(<4 x double> noundef %{{.*}}, <4 x double> noundef %{{.*}})
// CHECK-NEXT:   store <4 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z19test_swizzle_scalarN4sycl3_V13vecIfLi4EEEf(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 0, i32 2>
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z13test_marray_SN4sycl3_V13vecIfLi2EEE1S(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z10test_vec_SN4sycl3_V16marrayIfLm2EEE1S(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 4 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = alloca %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 4
// CHECK-NEXT:   store float 0.000000e+00, ptr %{{.*}}, align 4
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: for.cond.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i64 [ %{{.*}}, %{{.*}} ], [ 0, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = icmp ult i64 %{{.*}}, 2
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: for.body.i.i.i:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds [2 x float], ptr %{{.*}}, i64 0, i64 %{{.*}}
// CHECK-NEXT:   store float %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = add nuw nsw i64 %{{.*}}, 1
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZN4sycl3_V16marrayIfLm2EEC2ERKf.exit.i:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 1
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: arrayinit.body.i.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i64 [ 0, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = add nuw nsw i64 %{{.*}}, 4
// CHECK-NEXT:   %{{.*}} = icmp eq i64 %{{.*}}, 8
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZN4sycl3_V14fmaxINS0_6marrayIfLm2EEEEENS0_6detail14builtin_enableINS4_16default_ret_typeENS4_12fp_elem_typeENS4_15non_scalar_onlyENS4_14same_elem_typeEJT_EE4typeESA_NS4_13get_elem_typeISA_vE4typeE.exit:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z14test_swizzle_SN4sycl3_V13vecIfLi4EEE1S(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <2 x i32> <i32 0, i32 3>
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z15test_marray_S_2N4sycl3_V13vecIdLi2EEE1S(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 16 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x double>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   %{{.*}} = insertelement <2 x double> poison, double %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x double> %{{.*}}, <2 x double> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x double> @_Z16__spirv_ocl_fmaxDv2_dS_(<2 x double> noundef %{{.*}}, <2 x double> noundef %{{.*}})
// CHECK-NEXT:   store <2 x double> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z12test_vec_S_2N4sycl3_V16marrayIdLm2EEE1S(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = alloca %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = alloca %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %{{.*}})
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK-NEXT:   store double 0.000000e+00, ptr %{{.*}}, align 8
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: for.cond.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i64 [ %{{.*}}, %{{.*}} ], [ 0, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = icmp ult i64 %{{.*}}, 2
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: for.body.i.i.i:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds [2 x double], ptr %{{.*}}, i64 0, i64 %{{.*}}
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = add nuw nsw i64 %{{.*}}, 1
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZN4sycl3_V16marrayIdLm2EEC2ERKd.exit.i:
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK-NEXT:   %{{.*}} = load double, ptr %{{.*}}, align 1
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %{{.*}})
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: arrayinit.body.i.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i64 [ 0, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 %{{.*}}
// CHECK-NEXT:   store double 0.000000e+00, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = add nuw nsw i64 %{{.*}}, 8
// CHECK-NEXT:   %{{.*}} = icmp eq i64 %{{.*}}, 16
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZN4sycl3_V14fmaxINS0_6marrayIdLm2EEEEENS0_6detail14builtin_enableINS4_16default_ret_typeENS4_12fp_elem_typeENS4_15non_scalar_onlyENS4_14same_elem_typeEJT_EE4typeESA_NS4_13get_elem_typeISA_vE4typeE.exit:
// CHECK-NEXT:   %{{.*}} = insertelement <2 x double> undef, double %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 1
// CHECK-NEXT:   %{{.*}} = insertelement <2 x double> undef, double %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = insertelement <2 x double> %{{.*}}, double %{{.*}}, i64 1
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x double> @_Z16__spirv_ocl_fmaxDv2_dS_(<2 x double> noundef %{{.*}}, <2 x double> noundef %{{.*}})
// CHECK-NEXT:   store <2 x double> %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   call void @llvm.memcpy.p4.p0.i64(ptr addrspace(4) align 8 %{{.*}}, ptr align 8 %{{.*}}, i64 16, i1 false)
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %{{.*}})
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %{{.*}})
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
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
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
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z16__spirv_ocl_fmaxDv2_fS_(<2 x float> noundef %{{.*}}, <2 x float> noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define weak_odr dso_local noundef float @_Z9test_fmaxIfEDcT_S0_(float noundef %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V111__fmax_implEff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef double @_Z9test_fmaxIdEDcT_S0_(double noundef %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef double @_ZN4sycl3_V111__fmax_implEdd(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z9test_fmaxIN4sycl3_V16detail9half_impl4halfEEDcT_S5_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V111__fmax_implENS0_6detail9half_impl4halfES3_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local float @_Z9test_fmaxIN4sycl3_V13vecIfLi1EEEEDcT_S4_(float %{{.*}}, float %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call float @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi1EEES2_(float %{{.*}}, float %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local <2 x float> @_Z9test_fmaxIN4sycl3_V13vecIfLi2EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z9test_fmaxIN4sycl3_V13vecIfLi3EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi3EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { <2 x float>, <2 x float> } @_Z9test_fmaxIN4sycl3_V13vecIfLi4EEEEDcT_S4_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local double @_Z9test_fmaxIN4sycl3_V13vecIdLi1EEEEDcT_S4_(double %{{.*}}, double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call double @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi1EEES2_(double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret double %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { double, double } @_Z9test_fmaxIN4sycl3_V13vecIdLi2EEEEDcT_S4_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { double, double } @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi2EEES2_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret { double, double } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z9test_fmaxIN4sycl3_V13vecIdLi4EEEEDcT_S4_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi1EEEEDcT_S7_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V111__fmax_implENS0_3vecINS0_6detail9half_impl4halfELi1EEES5_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi2EEEEDcT_S7_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V111__fmax_implENS0_3vecINS0_6detail9half_impl4halfELi2EEES5_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z9test_fmaxIN4sycl3_V13vecINS1_6detail9half_impl4halfELi4EEEEDcT_S7_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V111__fmax_implENS0_3vecINS0_6detail9half_impl4halfELi4EEES5_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z17test_fmax_swizzleN4sycl3_V13vecIfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> undef, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local { <2 x float>, <2 x float> } @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEEf(<2 x float> %{{.*}}, <2 x float> %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> undef, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local { <2 x float>, <2 x float> } @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEEd(<2 x float> %{{.*}}, <2 x float> %{{.*}}, double noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = fptrunc double %{{.*}} to float
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> undef, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local { <2 x float>, <2 x float> } @_Z16test_scalar_arg1N4sycl3_V13vecIfLi4EEENS0_6detail9half_impl4halfE(<2 x float> %{{.*}}, <2 x float> %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = lshr i16 %{{.*}}, 10
// CHECK-NEXT:   %{{.*}} = and i16 %{{.*}}, 31
// CHECK-NEXT:   %{{.*}} = and i16 %{{.*}}, 1023
// CHECK-NEXT:   switch i16 %{{.*}}, label %{{.*}} [
// CHECK-NEXT:     i16 31, label %{{.*}}
// CHECK-NEXT:     i16 0, label %{{.*}}
// CHECK-NEXT:   ]
// CHECK-EMPTY:
// CHECK-NEXT: if.else15.i.i.i:
// CHECK-NEXT:   %{{.*}} = add nuw nsw i16 %{{.*}}, 112
// CHECK-NEXT:   %{{.*}} = zext nneg i16 %{{.*}} to i32
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: if.end17.i.i.i:
// CHECK-NEXT:   %{{.*}} = icmp eq i16 %{{.*}}, 0
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK:      do.body.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i8 [ %{{.*}}, %{{.*}} ], [ 0, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = phi i16 [ %{{.*}}, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = add i8 %{{.*}}, 1
// CHECK-NEXT:   %{{.*}} = shl i16 %{{.*}}, 1
// CHECK-NEXT:   %{{.*}} = and i16 %{{.*}}, 512
// CHECK-NEXT:   %{{.*}} = icmp eq i16 %{{.*}}, 0
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK:      do.end.i.i.i:
// CHECK-NEXT:   %{{.*}} = and i16 %{{.*}}, 1022
// CHECK-NEXT:   %{{.*}} = zext i8 %{{.*}} to i32
// CHECK-NEXT:   %{{.*}} = sub nsw i32 113, %{{.*}}
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZNK4sycl3_V16detail9half_impl4halfcvfEv.exit:
// CHECK-NEXT:   %{{.*}} = phi i32 [ %{{.*}}, %{{.*}} ], [ 0, %{{.*}} ], [ 255, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = phi i16 [ %{{.*}}, %{{.*}} ], [ 0, %{{.*}} ], [ %{{.*}}, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = sext i16 %{{.*}} to i32
// CHECK-NEXT:   %{{.*}} = and i32 %{{.*}}, -2147483648
// CHECK-NEXT:   %{{.*}} = zext nneg i16 %{{.*}} to i32
// CHECK-NEXT:   %{{.*}} = shl nuw nsw i32 %{{.*}}, 13
// CHECK-NEXT:   %{{.*}} = shl nsw i32 %{{.*}}, 23
// CHECK-NEXT:   %{{.*}} = or i32 %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = or disjoint i32 %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = insertelement <2 x i32> undef, i32 %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = bitcast <2 x i32> %{{.*}} to <2 x float>
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call { <2 x float>, <2 x float> } @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi4EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret { <2 x float>, <2 x float> } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local void @_Z16test_scalar_arg1N4sycl3_V13vecIdLi4EEEf(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 32 %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = alloca %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 32, ptr nonnull %{{.*}})
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 8
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 16
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 24
// CHECK-NEXT:   store double %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   tail call void @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 32, ptr nonnull %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z19test_swizzle_scalarN4sycl3_V13vecIfLi4EEEf(<2 x float> %{{.*}}, <2 x float> %{{.*}}, float noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x i32> <i32 0, i32 2>
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> undef, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z13test_marray_SN4sycl3_V13vecIfLi2EEE1S(<2 x float> %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> undef, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z10test_vec_SN4sycl3_V16marrayIfLm2EEE1S(<2 x float> %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> poison, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z14test_swizzle_SN4sycl3_V13vecIfLi4EEE1S(<2 x float> %{{.*}}, <2 x float> %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x i32> <i32 0, i32 3>
// CHECK-NEXT:   %{{.*}} = insertelement <2 x float> undef, float %{{.*}}, i64 0
// CHECK-NEXT:   %{{.*}} = shufflevector <2 x float> %{{.*}}, <2 x float> poison, <2 x i32> zeroinitializer
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local { double, double } @_Z15test_marray_S_2N4sycl3_V13vecIdLi2EEE1S(double %{{.*}}, double %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   %{{.*}} = tail call { double, double } @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi2EEES2_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret { double, double } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local { double, double } @_Z12test_vec_S_2N4sycl3_V16marrayIdLm2EEE1S(double %{{.*}}, double %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds %{{.*}}, ptr %{{.*}}, i64 0, i32 1
// CHECK-NEXT:   %{{.*}} = load float, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = fpext float %{{.*}} to double
// CHECK-NEXT:   %{{.*}} = tail call { double, double } @_ZN4sycl3_V111__fmax_implENS0_3vecIdLi2EEES2_(double %{{.*}}, double %{{.*}}, double %{{.*}}, double %{{.*}})
// CHECK-NEXT:   ret { double, double } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z16test_vec_swizzleN4sycl3_V13vecIfLi2EEENS1_IfLi4EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z16test_swizzle_vecN4sycl3_V13vecIfLi4EEENS1_IfLi2EEE(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmax_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
