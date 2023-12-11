// RUN: %clangxx -fpreview-breaking-changes -fsycl -O2 -S -emit-llvm -o - %s | FileCheck %s
#include <sycl/sycl.hpp>

decltype(auto) SYCL_EXTERNAL test(float x, sycl::global_ptr<float> p) {
  return sycl::fract(x, p);
}

static_assert(sycl::detail::builtin_ptr_check_v<
              sycl::float2, sycl::global_ptr<sycl::float2>>);
decltype(auto) SYCL_EXTERNAL test(sycl::float2 x,
                                  sycl::global_ptr<sycl::float2> p) {
  return sycl::fract(x, p);
}
decltype(auto) SYCL_EXTERNAL test(sycl::marray<float, 2> x,
                                  sycl::global_ptr<sycl::marray<float, 2>> p) {
  return sycl::fract(x, p);
}
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ sycl-spir64-unknown-unknown
// CHECK:      define dso_local spir_func noundef float @_Z4testfN4sycl3_V19multi_ptrIfLNS0_6access13address_spaceE1ELNS2_9decoratedE2EEE(float noundef %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = inttoptr i64 %{{.*}} to ptr addrspace(1)
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z17__spirv_ocl_fractfPU3AS1f(float noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z4testN4sycl3_V13vecIfLi2EEENS0_9multi_ptrIS2_LNS0_6access13address_spaceE1ELNS4_9decoratedE2EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = load <2 x float>, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   %{{.*}} = inttoptr i64 %{{.*}} to ptr addrspace(1)
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef <2 x float> @_Z17__spirv_ocl_fractDv2_fPU3AS1S_(<2 x float> noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr addrspace(4) %{{.*}}, align 8
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local spir_func void @_Z4testN4sycl3_V16marrayIfLm2EEENS0_9multi_ptrIS2_LNS0_6access13address_spaceE1ELNS4_9decoratedE2EEE(ptr addrspace(4) noalias nocapture writeonly sret(%{{.*}}) align 4 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 4 %{{.*}}, ptr nocapture noundef readonly byval(%{{.*}}) align 8 %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = alloca %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 8
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   tail call void @llvm.experimental.noalias.scope.decl(metadata {{.*}})
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %{{.*}})
// CHECK-NEXT:   br label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: arrayinit.body.i.i.i.i:
// CHECK-NEXT:   %{{.*}} = phi i64 [ 0, %{{.*}} ], [ %{{.*}}, %{{.*}} ]
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds i8, ptr %{{.*}}, i64 %{{.*}}
// CHECK-NEXT:   store float 0.000000e+00, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = add nuw nsw i64 %{{.*}}, 4
// CHECK-NEXT:   %{{.*}} = icmp eq i64 %{{.*}}, 8
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK-EMPTY:
// CHECK-NEXT: _ZN4sycl3_V15fractINS0_6marrayIfLm2EEENS0_9multi_ptrIS3_LNS0_6access13address_spaceE1ELNS5_9decoratedE2EEEEENS0_6detail18builtin_enable_ptrIT_T0_E4typeESB_SC_.exit:
// CHECK-NEXT:   %{{.*}} = inttoptr i64 %{{.*}} to ptr addrspace(1)
// CHECK-NEXT:   %{{.*}} = lshr i64 %{{.*}}, 32
// CHECK-NEXT:   %{{.*}} = trunc i64 %{{.*}} to i32
// CHECK-NEXT:   %{{.*}} = bitcast i32 %{{.*}} to float
// CHECK-NEXT:   %{{.*}} = trunc i64 %{{.*}} to i32
// CHECK-NEXT:   %{{.*}} = bitcast i32 %{{.*}} to float
// CHECK-NEXT:   %{{.*}} = addrspacecast ptr addrspace(1) %{{.*}} to ptr addrspace(4)
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) noundef %{{.*}}, i32 noundef 5)
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z17__spirv_ocl_fractfPU3AS1f(float noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds float, ptr addrspace(1) %{{.*}}, i64 1
// CHECK-NEXT:   %{{.*}} = tail call spir_func noundef float @_Z17__spirv_ocl_fractfPU3AS1f(float noundef %{{.*}}, ptr addrspace(1) noundef %{{.*}})
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds [2 x float], ptr %{{.*}}, i64 0, i64 1
// CHECK-NEXT:   store float %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = load i64, ptr %{{.*}}, align 4
// CHECK-NEXT:   store i64 %{{.*}}, ptr addrspace(4) %{{.*}}, align 4
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %{{.*}})
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ sycl-spir64-unknown-unknown
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
// CHECK:      define dso_local noundef float @_Z4testfN4sycl3_V19multi_ptrIfLNS0_6access13address_spaceE1ELNS2_9decoratedE2EEE(float noundef %{{.*}}, ptr nocapture writeonly %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V112__floor_implEf(float noundef %{{.*}})
// CHECK-NEXT:   store float %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = fsub float %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V116__nextafter_implEff(float noundef 1.000000e+00, float noundef 0.000000e+00)
// CHECK-NEXT:   %{{.*}} = tail call noundef float @_ZN4sycl3_V111__fmin_implEff(float noundef %{{.*}}, float noundef %{{.*}})
// CHECK-NEXT:   ret float %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z4testN4sycl3_V13vecIfLi2EEENS0_9multi_ptrIS2_LNS0_6access13address_spaceE1ELNS4_9decoratedE2EEE(<2 x float> %{{.*}}, ptr nocapture writeonly %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__floor_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = fsub <2 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V116__nextafter_implENS0_3vecIfLi2EEES2_(<2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float> zeroinitializer)
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmin_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      define dso_local <2 x float> @_Z4testN4sycl3_V16marrayIfLm2EEENS0_9multi_ptrIS2_LNS0_6access13address_spaceE1ELNS4_9decoratedE2EEE(<2 x float> %{{.*}}, ptr nocapture writeonly %{{.*}}) {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V112__floor_implENS0_3vecIfLi2EEE(<2 x float> %{{.*}})
// CHECK-NEXT:   store <2 x float> %{{.*}}, ptr %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = fsub <2 x float> %{{.*}}, %{{.*}}
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V116__nextafter_implENS0_3vecIfLi2EEES2_(<2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float> zeroinitializer)
// CHECK-NEXT:   %{{.*}} = tail call <2 x float> @_ZN4sycl3_V111__fmin_implENS0_3vecIfLi2EEES2_(<2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK-NEXT:   ret <2 x float> %{{.*}}
// CHECK-NEXT: }
// CHECK-EMPTY:
// CHECK:      ; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
