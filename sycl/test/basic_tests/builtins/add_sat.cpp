#include <sycl/sycl.hpp>

// RUN: %clangxx -fpreview-breaking-changes -fsycl -O2 -S -emit-llvm -o - %s | FileCheck %s

template <typename T1, typename T2>
decltype(auto) SYCL_EXTERNAL test_add_sat(T1 x, T2 y) {
  return sycl::add_sat(x, y);
}

#define TEST_ADD_SAT(TYPE)                                                     \
  template decltype(auto) test_add_sat(TYPE, TYPE);                            \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 1>,                     \
                                       sycl ::vec<TYPE, 1>);                   \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 2>,                     \
                                       sycl ::vec<TYPE, 2>);                   \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 3>,                     \
                                       sycl ::vec<TYPE, 3>);                   \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 4>,                     \
                                       sycl ::vec<TYPE, 4>);                   \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 8>,                     \
                                       sycl ::vec<TYPE, 8>);                   \
  template decltype(auto) test_add_sat(sycl::vec<TYPE, 16>,                    \
                                       sycl ::vec<TYPE, 16>);

TEST_ADD_SAT(char)
TEST_ADD_SAT(signed char)
TEST_ADD_SAT(short)
TEST_ADD_SAT(int)
TEST_ADD_SAT(long)
TEST_ADD_SAT(long long)

TEST_ADD_SAT(unsigned char)
TEST_ADD_SAT(unsigned short)
TEST_ADD_SAT(unsigned int)
TEST_ADD_SAT(unsigned long)
TEST_ADD_SAT(unsigned long long)

decltype(auto) test_vec_swizzle(sycl::vec<int, 2> x, sycl::vec<int, 4> y) {
  return sycl::add_sat(x, y.swizzle<2, 0>());
}

decltype(auto) test_swizzle_vec(sycl::vec<int, 4> x, sycl::vec<int, 2> y) {
  return sycl::add_sat(x.swizzle<2, 0>(), y);
}
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define weak_odr dso_local spir_func noundef signext i8 @_Z12test_add_satIccEDcT_T0_(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i8 @_Z21__spirv_ocl_s_add_sataa(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i8 @_Z21__spirv_ocl_s_add_sataa(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   store i8 %{{.*}}, ptr addrspace(4) %{{.*}}, align 1
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i8> @_Z21__spirv_ocl_s_add_satDv2_aS_(<2 x i8> noundef %{{.*}}, <2 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i8> @_Z21__spirv_ocl_s_add_satDv3_aS_(<3 x i8> noundef %{{.*}}, <3 x i8> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i8> %{{.*}}, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i8> @_Z21__spirv_ocl_s_add_satDv4_aS_(<4 x i8> noundef %{{.*}}, <4 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i8> @_Z21__spirv_ocl_s_add_satDv8_aS_(<8 x i8> noundef %{{.*}}, <8 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIcLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i8> @_Z21__spirv_ocl_s_add_satDv16_aS_(<16 x i8> noundef %{{.*}}, <16 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef signext i8 @_Z12test_add_satIaaEDcT_T0_(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i8 @_Z21__spirv_ocl_s_add_sataa(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i8 @_Z21__spirv_ocl_s_add_sataa(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   store i8 %{{.*}}, ptr addrspace(4) %{{.*}}, align 1
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i8> @_Z21__spirv_ocl_s_add_satDv2_aS_(<2 x i8> noundef %{{.*}}, <2 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i8> @_Z21__spirv_ocl_s_add_satDv3_aS_(<3 x i8> noundef %{{.*}}, <3 x i8> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i8> %{{.*}}, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i8> @_Z21__spirv_ocl_s_add_satDv4_aS_(<4 x i8> noundef %{{.*}}, <4 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i8> @_Z21__spirv_ocl_s_add_satDv8_aS_(<8 x i8> noundef %{{.*}}, <8 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIaLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i8> @_Z21__spirv_ocl_s_add_satDv16_aS_(<16 x i8> noundef %{{.*}}, <16 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef signext i16 @_Z12test_add_satIssEDcT_T0_(i16 noundef signext %{{.*}}, i16 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i16 @_Z21__spirv_ocl_s_add_satss(i16 noundef signext %{{.*}}, i16 noundef signext %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i16, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load i16, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef signext i16 @_Z21__spirv_ocl_s_add_satss(i16 noundef signext %{{.*}}, i16 noundef signext %{{.*}})
// CHECK-NEXT:   store i16 %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i16>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <2 x i16>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i16> @_Z21__spirv_ocl_s_add_satDv2_sS_(<2 x i16> noundef %{{.*}}, <2 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i16> %{{.*}}, <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i16> %{{.*}}, <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i16> @_Z21__spirv_ocl_s_add_satDv3_sS_(<3 x i16> noundef %{{.*}}, <3 x i16> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i16> %{{.*}}, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i16> @_Z21__spirv_ocl_s_add_satDv4_sS_(<4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i16>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <8 x i16>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i16> @_Z21__spirv_ocl_s_add_satDv8_sS_(<8 x i16> noundef %{{.*}}, <8 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIsLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i16>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <16 x i16>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i16> @_Z21__spirv_ocl_s_add_satDv16_sS_(<16 x i16> noundef %{{.*}}, <16 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i32 @_Z12test_add_satIiiEDcT_T0_(i32 noundef %{{.*}}, i32 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i32 @_Z21__spirv_ocl_s_add_satii(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i32 @_Z21__spirv_ocl_s_add_satii(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   store i32 %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i32>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <2 x i32>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i32> @_Z21__spirv_ocl_s_add_satDv2_iS_(<2 x i32> noundef %{{.*}}, <2 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i32> @_Z21__spirv_ocl_s_add_satDv3_iS_(<3 x i32> noundef %{{.*}}, <3 x i32> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i32> %{{.*}}, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i32> @_Z21__spirv_ocl_s_add_satDv4_iS_(<4 x i32> noundef %{{.*}}, <4 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i32>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <8 x i32>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i32> @_Z21__spirv_ocl_s_add_satDv8_iS_(<8 x i32> noundef %{{.*}}, <8 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIiLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i32>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i32>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i32> @_Z21__spirv_ocl_s_add_satDv16_iS_(<16 x i32> noundef %{{.*}}, <16 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i64 @_Z12test_add_satIllEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_s_add_satll(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_s_add_satll(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   store i64 %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i64> @_Z21__spirv_ocl_s_add_satDv2_lS_(<2 x i64> noundef %{{.*}}, <2 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i64> @_Z21__spirv_ocl_s_add_satDv3_lS_(<3 x i64> noundef %{{.*}}, <3 x i64> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i64> %{{.*}}, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i64> @_Z21__spirv_ocl_s_add_satDv4_lS_(<4 x i64> noundef %{{.*}}, <4 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i64> @_Z21__spirv_ocl_s_add_satDv8_lS_(<8 x i64> noundef %{{.*}}, <8 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIlLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i64> @_Z21__spirv_ocl_s_add_satDv16_lS_(<16 x i64> noundef %{{.*}}, <16 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i64 @_Z12test_add_satIxxEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_s_add_satll(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_s_add_satll(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   store i64 %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i64> @_Z21__spirv_ocl_s_add_satDv2_lS_(<2 x i64> noundef %{{.*}}, <2 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i64> @_Z21__spirv_ocl_s_add_satDv3_lS_(<3 x i64> noundef %{{.*}}, <3 x i64> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i64> %{{.*}}, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i64> @_Z21__spirv_ocl_s_add_satDv4_lS_(<4 x i64> noundef %{{.*}}, <4 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i64> @_Z21__spirv_ocl_s_add_satDv8_lS_(<8 x i64> noundef %{{.*}}, <8 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIxLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i64> @_Z21__spirv_ocl_s_add_satDv16_lS_(<16 x i64> noundef %{{.*}}, <16 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef zeroext i8 @_Z12test_add_satIhhEDcT_T0_(i8 noundef zeroext %{{.*}}, i8 noundef zeroext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef zeroext i8 @_Z21__spirv_ocl_u_add_sathh(i8 noundef zeroext %{{.*}}, i8 noundef zeroext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}, ptr noundef byval(%{{.*}}) align 1 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = load i8, ptr %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef zeroext i8 @_Z21__spirv_ocl_u_add_sathh(i8 noundef zeroext %{{.*}}, i8 noundef zeroext %{{.*}})
// CHECK-NEXT:   store i8 %{{.*}}, ptr addrspace(4) %{{.*}}, align 1
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load <2 x i8>, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i8> @_Z21__spirv_ocl_u_add_satDv2_hS_(<2 x i8> noundef %{{.*}}, <2 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i8> @_Z21__spirv_ocl_u_add_satDv3_hS_(<3 x i8> noundef %{{.*}}, <3 x i8> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i8> %{{.*}}, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <4 x i8>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i8> @_Z21__spirv_ocl_u_add_satDv4_hS_(<4 x i8> noundef %{{.*}}, <4 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <8 x i8>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i8> @_Z21__spirv_ocl_u_add_satDv8_hS_(<8 x i8> noundef %{{.*}}, <8 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIhLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <16 x i8>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i8> @_Z21__spirv_ocl_u_add_satDv16_hS_(<16 x i8> noundef %{{.*}}, <16 x i8> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i8> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef zeroext i16 @_Z12test_add_satIttEDcT_T0_(i16 noundef zeroext %{{.*}}, i16 noundef zeroext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef zeroext i16 @_Z21__spirv_ocl_u_add_sattt(i16 noundef zeroext %{{.*}}, i16 noundef zeroext %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}, ptr noundef byval(%{{.*}}) align 2 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i16, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = load i16, ptr %{{.*}}, align 2
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef zeroext i16 @_Z21__spirv_ocl_u_add_sattt(i16 noundef zeroext %{{.*}}, i16 noundef zeroext %{{.*}})
// CHECK-NEXT:   store i16 %{{.*}}, ptr addrspace(4) %{{.*}}, align 2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i16>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load <2 x i16>, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i16> @_Z21__spirv_ocl_u_add_satDv2_tS_(<2 x i16> noundef %{{.*}}, <2 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i16> %{{.*}}, <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i16> %{{.*}}, <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i16> @_Z21__spirv_ocl_u_add_satDv3_tS_(<3 x i16> noundef %{{.*}}, <3 x i16> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i16> %{{.*}}, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <4 x i16>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i16> @_Z21__spirv_ocl_u_add_satDv4_tS_(<4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i16>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <8 x i16>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i16> @_Z21__spirv_ocl_u_add_satDv8_tS_(<8 x i16> noundef %{{.*}}, <8 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecItLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i16>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <16 x i16>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i16> @_Z21__spirv_ocl_u_add_satDv16_tS_(<16 x i16> noundef %{{.*}}, <16 x i16> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i16> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i32 @_Z12test_add_satIjjEDcT_T0_(i32 noundef %{{.*}}, i32 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i32 @_Z21__spirv_ocl_u_add_satjj(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}, ptr noundef byval(%{{.*}}) align 4 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load i32, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i32 @_Z21__spirv_ocl_u_add_satjj(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   store i32 %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i32>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load <2 x i32>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i32> @_Z21__spirv_ocl_u_add_satDv2_jS_(<2 x i32> noundef %{{.*}}, <2 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i32> @_Z21__spirv_ocl_u_add_satDv3_jS_(<3 x i32> noundef %{{.*}}, <3 x i32> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i32> %{{.*}}, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <4 x i32>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i32> @_Z21__spirv_ocl_u_add_satDv4_jS_(<4 x i32> noundef %{{.*}}, <4 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i32>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <8 x i32>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i32> @_Z21__spirv_ocl_u_add_satDv8_jS_(<8 x i32> noundef %{{.*}}, <8 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIjLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i32>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i32>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i32> @_Z21__spirv_ocl_u_add_satDv16_jS_(<16 x i32> noundef %{{.*}}, <16 x i32> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i32> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i64 @_Z12test_add_satImmEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_u_add_satmm(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_u_add_satmm(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   store i64 %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i64> @_Z21__spirv_ocl_u_add_satDv2_mS_(<2 x i64> noundef %{{.*}}, <2 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i64> @_Z21__spirv_ocl_u_add_satDv3_mS_(<3 x i64> noundef %{{.*}}, <3 x i64> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i64> %{{.*}}, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i64> @_Z21__spirv_ocl_u_add_satDv4_mS_(<4 x i64> noundef %{{.*}}, <4 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i64> @_Z21__spirv_ocl_u_add_satDv8_mS_(<8 x i64> noundef %{{.*}}, <8 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecImLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i64> @_Z21__spirv_ocl_u_add_satDv16_mS_(<16 x i64> noundef %{{.*}}, <16 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func noundef i64 @_Z12test_add_satIyyEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_u_add_satmm(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi1EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}, ptr noundef byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef i64 @_Z21__spirv_ocl_u_add_satmm(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   store i64 %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi2EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}, ptr noundef byval(%{{.*}}) align 16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = load <2 x i64>, ptr %{{.*}}, align 16
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x i64> @_Z21__spirv_ocl_u_add_satDv2_mS_(<2 x i64> noundef %{{.*}}, <2 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <2 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 16
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi3EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = shufflevector <4 x i64> %{{.*}}, <4 x i64> poison, <3 x i32> <i32 0, i32 1, i32 2>
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <3 x i64> @_Z21__spirv_ocl_u_add_satDv3_mS_(<3 x i64> noundef %{{.*}}, <3 x i64> noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = shufflevector <3 x i64> %{{.*}}, <3 x i64> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi4EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = load <4 x i64>, ptr %{{.*}}, align 32
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <4 x i64> @_Z21__spirv_ocl_u_add_satDv4_mS_(<4 x i64> noundef %{{.*}}, <4 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <4 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 32
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi8EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <8 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <8 x i64> @_Z21__spirv_ocl_u_add_satDv8_mS_(<8 x i64> noundef %{{.*}}, <8 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <8 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local spir_func void @_Z12test_add_satIN4sycl3_V13vecIyLi16EEES3_EDcT_T0_(ptr addrspace(4) noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = load <16 x i64>, ptr %{{.*}}, align 64
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <16 x i64> @_Z21__spirv_ocl_u_add_satDv16_mS_(<16 x i64> noundef %{{.*}}, <16 x i64> noundef %{{.*}})
// CHECK-NEXT:   store <16 x i64> %{{.*}}, ptr addrspace(4) %{{.*}}, align 64
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define weak_odr dso_local noundef signext i8 @_Z12test_add_satIccEDcT_T0_(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef signext i8 @_ZN4sycl3_V114__add_sat_implEcc(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i8 @_Z12test_add_satIN4sycl3_V13vecIcLi1EEES3_EDcT_T0_(i8 %{{.*}}, i8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i8 @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi1EEES2_(i8 %{{.*}}, i8 %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z12test_add_satIN4sycl3_V13vecIcLi2EEES3_EDcT_T0_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi2EEES2_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIcLi3EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi3EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIcLi4EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi4EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIcLi8EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi8EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIcLi16EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIcLi16EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef signext i8 @_Z12test_add_satIaaEDcT_T0_(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef signext i8 @_ZN4sycl3_V114__add_sat_implEaa(i8 noundef signext %{{.*}}, i8 noundef signext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i8 @_Z12test_add_satIN4sycl3_V13vecIaLi1EEES3_EDcT_T0_(i8 %{{.*}}, i8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i8 @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi1EEES2_(i8 %{{.*}}, i8 %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z12test_add_satIN4sycl3_V13vecIaLi2EEES3_EDcT_T0_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi2EEES2_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIaLi3EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi3EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIaLi4EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi4EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIaLi8EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi8EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIaLi16EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIaLi16EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef signext i16 @_Z12test_add_satIssEDcT_T0_(i16 noundef signext %{{.*}}, i16 noundef signext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef signext i16 @_ZN4sycl3_V114__add_sat_implEss(i16 noundef signext %{{.*}}, i16 noundef signext %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z12test_add_satIN4sycl3_V13vecIsLi1EEES3_EDcT_T0_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi1EEES2_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIsLi2EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi2EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIsLi3EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi3EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIsLi4EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi4EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIsLi8EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi8EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIsLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIsLi16EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i32 @_Z12test_add_satIiiEDcT_T0_(i32 noundef %{{.*}}, i32 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i32 @_ZN4sycl3_V114__add_sat_implEii(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIiLi1EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi1EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIiLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi2EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIiLi3EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi3EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIiLi4EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi4EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIiLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi8EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIiLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i64 @_Z12test_add_satIllEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i64 @_ZN4sycl3_V114__add_sat_implEll(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIlLi1EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi1EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIlLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi2EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIlLi3EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi3EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIlLi4EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIlLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi8EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIlLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIlLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i64 @_Z12test_add_satIxxEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i64 @_ZN4sycl3_V114__add_sat_implExx(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIxLi1EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi1EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIxLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi2EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIxLi3EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi3EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIxLi4EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIxLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi8EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIxLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIxLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef zeroext i8 @_Z12test_add_satIhhEDcT_T0_(i8 noundef zeroext %{{.*}}, i8 noundef zeroext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef zeroext i8 @_ZN4sycl3_V114__add_sat_implEhh(i8 noundef zeroext %{{.*}}, i8 noundef zeroext %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i8 @_Z12test_add_satIN4sycl3_V13vecIhLi1EEES3_EDcT_T0_(i8 %{{.*}}, i8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i8 @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi1EEES2_(i8 %{{.*}}, i8 %{{.*}})
// CHECK-NEXT:   ret i8 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z12test_add_satIN4sycl3_V13vecIhLi2EEES3_EDcT_T0_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi2EEES2_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIhLi3EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi3EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIhLi4EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi4EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIhLi8EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi8EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIhLi16EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIhLi16EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef zeroext i16 @_Z12test_add_satIttEDcT_T0_(i16 noundef zeroext %{{.*}}, i16 noundef zeroext %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef zeroext i16 @_ZN4sycl3_V114__add_sat_implEtt(i16 noundef zeroext %{{.*}}, i16 noundef zeroext %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i16 @_Z12test_add_satIN4sycl3_V13vecItLi1EEES3_EDcT_T0_(i16 %{{.*}}, i16 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i16 @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi1EEES2_(i16 %{{.*}}, i16 %{{.*}})
// CHECK-NEXT:   ret i16 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecItLi2EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi2EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecItLi3EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi3EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecItLi4EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi4EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecItLi8EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi8EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecItLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecItLi16EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i32 @_Z12test_add_satIjjEDcT_T0_(i32 noundef %{{.*}}, i32 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i32 @_ZN4sycl3_V114__add_sat_implEjj(i32 noundef %{{.*}}, i32 noundef %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i32 @_Z12test_add_satIN4sycl3_V13vecIjLi1EEES3_EDcT_T0_(i32 %{{.*}}, i32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i32 @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi1EEES2_(i32 %{{.*}}, i32 %{{.*}})
// CHECK-NEXT:   ret i32 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIjLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi2EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIjLi3EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi3EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIjLi4EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi4EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIjLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi8EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIjLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIjLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i64 @_Z12test_add_satImmEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i64 @_ZN4sycl3_V114__add_sat_implEmm(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecImLi1EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi1EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecImLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi2EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecImLi3EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi3EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecImLi4EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecImLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi8EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecImLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecImLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local noundef i64 @_Z12test_add_satIyyEDcT_T0_(i64 noundef %{{.*}}, i64 noundef %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef i64 @_ZN4sycl3_V114__add_sat_implEyy(i64 noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local i64 @_Z12test_add_satIN4sycl3_V13vecIyLi1EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi1EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local { i64, i64 } @_Z12test_add_satIN4sycl3_V13vecIyLi2EEES3_EDcT_T0_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call { i64, i64 } @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi2EEES2_(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret { i64, i64 } %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIyLi3EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi3EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIyLi4EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}, ptr noundef byval(%{{.*}}) align 32 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi4EEES2_(ptr sret(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 32 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIyLi8EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi8EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define weak_odr dso_local void @_Z12test_add_satIN4sycl3_V13vecIyLi16EEES3_EDcT_T0_(ptr noalias sret(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}, ptr noundef byval(%{{.*}}) align 64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN4sycl3_V114__add_sat_implENS0_3vecIyLi16EEES2_(ptr sret(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}}, ptr noundef nonnull byval(%{{.*}}) align 64 %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local i64 @_Z16test_vec_swizzleN4sycl3_V13vecIiLi2EEENS1_IiLi4EEE(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shl i64 %{{.*}}, 32
// CHECK-NEXT:   %{{.*}} = and i64 %{{.*}}, 4294967295
// CHECK-NEXT:   %{{.*}} = or disjoint i64 %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi2EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local i64 @_Z16test_swizzle_vecN4sycl3_V13vecIiLi4EEENS1_IiLi2EEE(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = shl i64 %{{.*}}, 32
// CHECK-NEXT:   %{{.*}} = and i64 %{{.*}}, 4294967295
// CHECK-NEXT:   %{{.*}} = or disjoint i64 %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = tail call i64 @_ZN4sycl3_V114__add_sat_implENS0_3vecIiLi2EEES2_(i64 %{{.*}}, i64 %{{.*}})
// CHECK-NEXT:   ret i64 %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
