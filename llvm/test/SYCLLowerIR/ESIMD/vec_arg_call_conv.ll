; RUN: opt -passes=esimd-opt-call-conv -S < %s | FileCheck %s
; This test checks the ESIMDOptimizeVecArgCallConvPass optimization.
; See testcase description below.

; ModuleID = 'opaque_ptr.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd.0" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" = type { <384 x float> }
%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <16 x float> }
%"class.sycl::_V1::ext::intel::esimd::simd.2" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.3" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.3" = type { <8 x i32> }
%"class.sycl::_V1::ext::intel::esimd::simd.4" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.5" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.5" = type { <32 x double> }

@GRF = dso_local global %"class.sycl::_V1::ext::intel::esimd::simd.0" zeroinitializer, align 2048

; // Compilation: clang++ -fsycl -Xclang -opaque-pointers src.cpp
; // Template for the source:
;
; #include <sycl/ext/intel/esimd.hpp>
;
; using namespace sycl::ext::intel::esimd;
;
; ESIMD_PRIVATE simd<float, 3 * 32 * 4> GRF;
; #define V(x, w, i) (x).template select<w, 1>(i)
;
; // insert testcases here
;
; int main() {
;   return 0;
; }

;----- Test1: "Fall-through case", incoming optimizeable parameter is just returned
; __attribute__((noinline))
; SYCL_EXTERNAL simd<float, 16> callee__sret__param(simd<float, 16> x) SYCL_ESIMD_FUNCTION {
;   return x;
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<float, 16> test__sret__fall_through__arr(simd<float, 16> *x, int i) SYCL_ESIMD_FUNCTION {
;   return callee__sret__param(x[i]);
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<float, 16> test__sret__fall_through__glob() SYCL_ESIMD_FUNCTION {
;   return callee__sret__param(V(GRF, 16, 0));
; }
;
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z19callee__sret__param(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd") align 64 %agg.result, ptr noundef %x) local_unnamed_addr #0 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <16 x float> @_Z19callee__sret__param(<16 x float> %[[PARAM:.+]])
entry:
; CHECK:  %[[ALLOCA1:.+]] = alloca <16 x float>, align 64
; CHECK:  %[[CAST1:.+]] = addrspacecast ptr %[[ALLOCA1]] to ptr addrspace(4)
; CHECK:  %[[ALLOCA2:.+]] = alloca <16 x float>, align 64
; CHECK:  store <16 x float> %[[PARAM]], ptr %[[ALLOCA2]], align 64
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
; CHECK:  %[[ALLOCA2_4:.+]] = addrspacecast ptr %[[ALLOCA2]] to ptr addrspace(4)
  %call.i.i.i1 = load <16 x float>, ptr addrspace(4) %x.ascast, align 64
; CHECK:  %[[VAL:.+]] = load <16 x float>, ptr addrspace(4) %[[ALLOCA2_4]], align 64
  store <16 x float> %call.i.i.i1, ptr addrspace(4) %agg.result, align 64
; CHECK:  store <16 x float> %[[VAL]], ptr addrspace(4) %[[CAST1]], align 64
  ret void
; CHECK:  %[[RET:.+]] = load <16 x float>, ptr %[[ALLOCA1]], align 64
; CHECK:  ret <16 x float> %[[RET]]
}

;----- Caller 1 for the "Fall-through case": simd object is read from array
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z29test__sret__fall_through__arr(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd") align 64 %agg.result, ptr addrspace(4) noundef %x, i32 noundef %i) local_unnamed_addr #0 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <16 x float> @_Z29test__sret__fall_through__arr(ptr addrspace(4) noundef %[[PARAM0:.+]], i32 noundef %{{.*}})
entry:
; CHECK:  %[[ALLOCA1:.+]] = alloca <16 x float>, align 64
; CHECK:  %[[CAST1:.+]] = addrspacecast ptr %[[ALLOCA1]] to ptr addrspace(4)
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 64
; CHECK:  %[[ALLOCA2:.+]] = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 64
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", ptr addrspace(4) %x, i64 %idxprom
  %call.i.i.i1 = load <16 x float>, ptr addrspace(4) %arrayidx, align 64
  store <16 x float> %call.i.i.i1, ptr addrspace(4) %agg.tmp.ascast, align 64

; CHECK:  %[[VAL:.+]] = load <16 x float>, ptr %[[ALLOCA2]], align 64
  call spir_func void @_Z19callee__sret__param(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd") align 64 %agg.result, ptr noundef nonnull %agg.tmp) #7
; CHECK:  %[[RES:.+]] = call spir_func <16 x float> @_Z19callee__sret__param(<16 x float> %[[VAL]])
; CHECK:  store <16 x float> %[[RES]], ptr addrspace(4) %[[CAST1]], align 64
  ret void
; CHECK:  %[[RET:.+]] = load <16 x float>, ptr %[[ALLOCA1]], align 64
; CHECK:  ret <16 x float> %[[RET]]
}

;----- Caller 2 for the "Fall-through case": simd object is read from a global
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z30test__sret__fall_through__globv(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd") align 64 %agg.result) local_unnamed_addr #2 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
entry:
; CHECK:  %[[ALLOCA1:.+]] = alloca <16 x float>, align 64
; CHECK:  %[[CAST1:.+]] = addrspacecast ptr %[[ALLOCA1]] to ptr addrspace(4)
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 64
; CHECK:  %[[ALLOCA2:.+]] = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 64
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %call.i.i.i1 = load <384 x float>, ptr addrspace(4) addrspacecast (ptr @GRF to ptr addrspace(4)), align 2048
  %call2.i.i.i.esimd = call <16 x float> @llvm.genx.rdregionf.v16f32.v384f32.i16(<384 x float> %call.i.i.i1, i32 0, i32 16, i32 1, i16 0, i32 0)
  store <16 x float> %call2.i.i.i.esimd, ptr addrspace(4) %agg.tmp.ascast, align 64
; CHECK:  %[[VAL:.+]] = load <16 x float>, ptr %[[ALLOCA2]], align 64
  call spir_func void @_Z19callee__sret__param(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd") align 64 %agg.result, ptr noundef nonnull %agg.tmp) #7
; CHECK:  %[[RES:.+]] = call spir_func <16 x float> @_Z19callee__sret__param(<16 x float> %[[VAL]])
; CHECK:  store <16 x float> %[[RES]], ptr addrspace(4) %[[CAST1]], align 64
  ret void
; CHECK:  %[[RET:.+]] = load <16 x float>, ptr %[[ALLOCA1]], align 64
; CHECK:  ret <16 x float> %[[RET]]
}

; Check only signatures and calls in testcases below.

;----- Test2: Optimized parameter interleaves non-optimizeable ones.
; __attribute__((noinline))
; SYCL_EXTERNAL simd<int, 8> callee__sret__x_param_x(int i, simd<int, 8> x, int j) SYCL_ESIMD_FUNCTION {
;   return x + (i + j);
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<int, 8> test__sret__x_param_x(simd<int, 8> x) SYCL_ESIMD_FUNCTION {
;   return callee__sret__x_param_x(2, x, 1);
; }
;
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z23callee__sret__x_param_x(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, i32 noundef %i, ptr noundef %x, i32 noundef %j) local_unnamed_addr #3 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <8 x i32> @_Z23callee__sret__x_param_x(i32 noundef %{{.*}}, <8 x i32> %{{.*}}, i32 noundef %{{.*}})
entry:
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %add = add nsw i32 %i, %j
  %splat.splatinsert.i.i.i = insertelement <8 x i32> poison, i32 %add, i64 0
  %splat.splat.i.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %call.i.i.i1 = load <8 x i32>, ptr addrspace(4) %x.ascast, align 32
  %add.i.i.i.i.i = add <8 x i32> %call.i.i.i1, %splat.splat.i.i.i
  store <8 x i32> %add.i.i.i.i.i, ptr addrspace(4) %agg.result, align 32
  ret void
}

;----- Test2 caller.
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z21test__sret__x_param_x(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, ptr noundef %x) local_unnamed_addr #3 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <8 x i32> @_Z21test__sret__x_param_x(<8 x i32> %{{.*}})
entry:
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.2", align 32
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %call.i.i.i1 = load <8 x i32>, ptr addrspace(4) %x.ascast, align 32
  store <8 x i32> %call.i.i.i1, ptr addrspace(4) %agg.tmp.ascast, align 32
  call spir_func void @_Z23callee__sret__x_param_x(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, i32 noundef 2, ptr noundef nonnull %agg.tmp, i32 noundef 1) #7
; CHECK:  %{{.*}} = call spir_func <8 x i32> @_Z23callee__sret__x_param_x(i32 2, <8 x i32> %{{.*}}, i32 1)
  ret void
}

;----- Test3: "2-level fall through", bottom-level callee
; __attribute__((noinline))
; SYCL_EXTERNAL simd<double, 32> callee__all_fall_through0(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
;   return x;
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<double, 32> callee__all_fall_through1(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
;   return callee__all_fall_through0(x);
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<double, 32> test__all_fall_through(simd<double, 32> x) SYCL_ESIMD_FUNCTION {
;   return callee__all_fall_through1(x);
; }
;
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z25callee__all_fall_through0(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.4") align 256 %agg.result, ptr noundef %x) local_unnamed_addr #5 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <32 x double> @_Z25callee__all_fall_through0(<32 x double> %{{.*}})
entry:
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %call.i.i.i1 = load <32 x double>, ptr addrspace(4) %x.ascast, align 256
  store <32 x double> %call.i.i.i1, ptr addrspace(4) %agg.result, align 256
  ret void
}

;----- Test3 intermediate caller/callee.
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z25callee__all_fall_through1(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.4") align 256 %agg.result, ptr noundef %x) local_unnamed_addr #5 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <32 x double> @_Z25callee__all_fall_through1(<32 x double> %{{.*}})
entry:
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.4", align 256
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %call.i.i.i1 = load <32 x double>, ptr addrspace(4) %x.ascast, align 256
  store <32 x double> %call.i.i.i1, ptr addrspace(4) %agg.tmp.ascast, align 256
  call spir_func void @_Z25callee__all_fall_through0(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd.4") align 256 %agg.result, ptr noundef nonnull %agg.tmp) #7
; CHECK:  %{{.*}} = call spir_func <32 x double> @_Z25callee__all_fall_through0(<32 x double> %{{.*}})
  ret void
}

;----- Test3 top caller.
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z22test__all_fall_through(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.4") align 256 %agg.result, ptr noundef %x) local_unnamed_addr #5 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <32 x double> @_Z22test__all_fall_through(<32 x double> %{{.*}})
entry:
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.4", align 256
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %call.i.i.i1 = load <32 x double>, ptr addrspace(4) %x.ascast, align 256
  store <32 x double> %call.i.i.i1, ptr addrspace(4) %agg.tmp.ascast, align 256
  call spir_func void @_Z25callee__all_fall_through1(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd.4") align 256 %agg.result, ptr noundef nonnull %agg.tmp) #7
; CHECK:  %{{.*}} = call spir_func <32 x double> @_Z25callee__all_fall_through1(<32 x double> %{{.*}})
  ret void
}

; Function Attrs: alwaysinline nounwind readnone
declare !genx_intrinsic_id !10 <16 x float> @llvm.genx.rdregionf.v16f32.v384f32.i16(<384 x float>, i32, i32, i32, i16, i32) #6

%"class.sycl::_V1::ext::intel::esimd::simd.6" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.6" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.6" = type { <8 x i32> }

;----- Test4. First argument is passed by reference and updated in the callee,
;             must not be optimized.
; __attribute__((noinline))
; SYCL_EXTERNAL void callee_void__noopt_opt(simd<int, 8> &x, simd<int, 8> y) SYCL_ESIMD_FUNCTION {
;   x = x + y;
; }
;
; __attribute__((noinline))
; SYCL_EXTERNAL simd<int, 8> test__sret__noopt_opt(simd<int, 8> x) SYCL_ESIMD_FUNCTION {
;   callee_void__noopt_opt(x, x);
;   return x;
; }
;

define dso_local spir_func void @_Z22callee_void__noopt_opt(ptr addrspace(4) noundef %x, ptr noundef %y) !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func void @_Z22callee_void__noopt_opt(ptr addrspace(4) noundef %{{.*}}, <8 x i32> %{{.*}})
entry:
  %y.ascast = addrspacecast ptr %y to ptr addrspace(4)
  %call.i.i1 = load <8 x i32>, ptr addrspace(4) %x, align 32
  %call.i5.i2 = load <8 x i32>, ptr addrspace(4) %y.ascast, align 32
  %add.i.i.i.i = add <8 x i32> %call.i.i1, %call.i5.i2
  store <8 x i32> %add.i.i.i.i, ptr addrspace(4) %x, align 32
  ret void
}

define dso_local spir_func void @_Z21test__sret__noopt_opt(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.6") align 32 %agg.result, ptr noundef %x) !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <8 x i32> @_Z21test__sret__noopt_opt(ptr noundef %{{.*}})
entry:
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.6", align 32
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %call.i.i.i2 = load <8 x i32>, ptr addrspace(4) %x.ascast, align 32
  store <8 x i32> %call.i.i.i2, ptr addrspace(4) %agg.tmp.ascast, align 32
  call spir_func void @_Z22callee_void__noopt_opt(ptr addrspace(4) noundef align 32 dereferenceable(32) %x.ascast, ptr noundef nonnull %agg.tmp) #5
; CHECK:  call spir_func void @_Z22callee_void__noopt_opt(ptr addrspace(4) %{{.*}}, <8 x i32> %{{.*}})
  %call.i.i.i13 = load <8 x i32>, ptr addrspace(4) %x.ascast, align 32
  store <8 x i32> %call.i.i.i13, ptr addrspace(4) %agg.result, align 32
  ret void
}

;----- Test4: IR contains all-zero GEP instructions in parameter use-def chains
; Based on Test2.
define dso_local spir_func void @_Z23callee__sret__x_param_x1(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, i32 noundef %i, ptr noundef %x, i32 noundef %j) local_unnamed_addr #3 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <8 x i32> @_Z23callee__sret__x_param_x1(i32 noundef %{{.*}}, <8 x i32> %{{.*}}, i32 noundef %{{.*}})
entry:
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %add = add nsw i32 %i, %j
  %splat.splatinsert.i.i.i = insertelement <8 x i32> poison, i32 %add, i64 0
  %splat.splat.i.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.3", ptr addrspace(4) %x.ascast, i64 0, i32 0
  %call.i.i.i1 = load <8 x i32>, ptr addrspace(4) %M_data.i.i.i, align 32
  %add.i.i.i.i.i = add <8 x i32> %call.i.i.i1, %splat.splat.i.i.i
  store <8 x i32> %add.i.i.i.i.i, ptr addrspace(4) %agg.result, align 32
  ret void
}

;----- Test4 caller.
; Function Attrs: convergent noinline norecurse
define dso_local spir_func void @_Z21test__sret__x_param_x1(ptr addrspace(4) noalias sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, ptr noundef %x) local_unnamed_addr #3 !sycl_explicit_simd !8 !intel_reqd_sub_group_size !9 {
; CHECK: define dso_local spir_func <8 x i32> @_Z21test__sret__x_param_x1(<8 x i32> %{{.*}})
entry:
  %agg.tmp = alloca %"class.sycl::_V1::ext::intel::esimd::simd.2", align 32
  %agg.tmp.ascast = addrspacecast ptr %agg.tmp to ptr addrspace(4)
  %x.ascast = addrspacecast ptr %x to ptr addrspace(4)
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.3", ptr addrspace(4) %x.ascast, i64 0, i32 0
  %call.i.i.i1 = load <8 x i32>, ptr addrspace(4) %M_data.i.i.i, align 32
  store <8 x i32> %call.i.i.i1, ptr addrspace(4) %agg.tmp.ascast, align 32
  call spir_func void @_Z23callee__sret__x_param_x1(ptr addrspace(4) sret(%"class.sycl::_V1::ext::intel::esimd::simd.2") align 32 %agg.result, i32 noundef 2, ptr noundef nonnull %agg.tmp, i32 noundef 1) #7
; CHECK:  %{{.*}} = call spir_func <8 x i32> @_Z23callee__sret__x_param_x1(i32 2, <8 x i32> %{{.*}}, i32 1)
  ret void
}

attributes #0 = { convergent noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="512" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../opaque_ptr.cpp" }
attributes #1 = { alwaysinline convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="12288" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../opaque_ptr.cpp" }
attributes #3 = { convergent noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../opaque_ptr.cpp" }
attributes #4 = { alwaysinline argmemonly nocallback nofree nosync nounwind willreturn }
attributes #5 = { convergent noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="2048" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../opaque_ptr.cpp" }
attributes #6 = { alwaysinline nounwind readnone }
attributes #7 = { convergent }

!llvm.dependent-libraries = !{!0, !0, !0}
!opencl.spir.version = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!spirv.Source = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.ident = !{!4, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!llvm.module.flags = !{!6, !7}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}
!genx.kernels = !{}

!0 = !{!"libcpmt"}
!1 = !{i32 1, i32 2}
!2 = !{i32 0, i32 100000}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 16.0.0 (https://github.com/kbobrovs/llvm 3a1daae5481305320c2f8e8ab94fb71f565475b8)"}
!5 = !{!"clang version 16.0.0 (https://github.com/kbobrovs/llvm b4efcdb38a05b386e03091e8c9518b6a51c0bf7d)"}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{}
!9 = !{i32 1}
!10 = !{i32 11881}
