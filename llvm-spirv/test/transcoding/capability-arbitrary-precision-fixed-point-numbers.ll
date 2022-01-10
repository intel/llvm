; SYCL source (compiled with -S -emit-llvm -fsycl-device-only):
; template <int W, int rW, bool S, int I, int rI>
; void sqrt() {
;   ap_int<W> a;
;   auto ap_fixed_Sqrt = __spirv_FixedSqrtINTEL<W,rW>(a, S, I, rI);
;   ap_int<rW> b;
;   auto ap_fixed_Sqrt_b = __spirv_FixedSqrtINTEL<rW, W>(b, S, I, rI);
;   ap_int<rW> c;
;   auto ap_fixed_Sqrt_c = __spirv_FixedSqrtINTEL<rW, W>(c, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void recip() {
;   ap_int<W> a;
;   auto ap_fixed_Recip = __spirv_FixedRecipINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void rsqrt() {
;   ap_int<W> a;
;   auto ap_fixed_Rsqrt = __spirv_FixedRsqrtINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void sin() {
;   ap_int<W> a;
;   auto ap_fixed_Sin = __spirv_FixedSinINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void cos() {
;   ap_int<W> a;
;   auto ap_fixed_Cos = __spirv_FixedCosINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void sin_cos() {
;   ap_int<W> a;
;   auto ap_fixed_SinCos = __spirv_FixedSinCosINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void sin_pi() {
;   ap_int<W> a;
;   auto ap_fixed_SinPi = __spirv_FixedSinPiINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void cos_pi() {
;   ap_int<W> a;
;   auto ap_fixed_CosPi = __spirv_FixedCosPiINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void sin_cos_pi() {
;   ap_int<W> a;
;   auto ap_fixed_SinCosPi = __spirv_FixedSinCosPiINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void log() {
;   ap_int<W> a;
;   auto ap_fixed_Log = __spirv_FixedLogINTEL<W,rW>(a, S, I, rI);
; }

; template <int W, int rW, bool S, int I, int rI>
; void exp() {
;   ap_int<W> a;
;   auto ap_fixed_Exp = __spirv_FixedExpINTEL<W,rW>(a, S, I, rI);
; }

; template <typename name, typename Func>
; __attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
;   kernelFunc();
; }

; int main() {
;   kernel_single_task<class kernel_function>([]() {
;     sqrt<13, 5, false, 2, 2>();
;     recip<3, 8, true, 4, 4>();
;     rsqrt<11, 10, false, 8, 6>();
;     sin<17, 11, true, 7, 5>();
;     cos<35, 28, false, 9, 3>();
;     sin_cos<31, 20, true, 10, 12>();
;     sin_pi<60, 5, false, 2, 2>();
;     cos_pi<28, 16, false, 8, 5>();
;     sin_cos_pi<13, 5, false, 2, 2>();
;     log<64, 44, true, 24, 22>();
;     exp<44, 34, false, 20, 20>();
;     exp<68, 68, false, 20, 20>();
;   });
;   return 0;
; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_arbitrary_precision_integers,+SPV_INTEL_arbitrary_precision_fixed_point -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_arbitrary_precision_integers -spirv-text -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ERROR: Fixed point instructions can't be translated correctly without enabled SPV_INTEL_arbitrary_precision_fixed_point extension!

; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 2 Capability Kernel
; CHECK-SPIRV: 2 Capability ArbitraryPrecisionIntegersINTEL
; CHECK-SPIRV: 2 Capability ArbitraryPrecisionFixedPointINTEL
; CHECK-SPIRV: 12 Extension "SPV_INTEL_arbitrary_precision_fixed_point"
; CHECK-SPIRV: 11 Extension "SPV_INTEL_arbitrary_precision_integers"

; CHECK-SPIRV: 4 TypeInt [[Ty_8:[0-9]+]] 8 0
; CHECK-SPIRV: 4 TypeInt [[Ty_13:[0-9]+]] 13 0
; CHECK-SPIRV: 4 TypeInt [[Ty_5:[0-9]+]] 5 0
; CHECK-SPIRV: 4 TypeInt [[Ty_3:[0-9]+]] 3 0
; CHECK-SPIRV: 4 TypeInt [[Ty_11:[0-9]+]] 11 0
; CHECK-SPIRV: 4 TypeInt [[Ty_10:[0-9]+]] 10 0
; CHECK-SPIRV: 4 TypeInt [[Ty_17:[0-9]+]] 17 0
; CHECK-SPIRV: 4 TypeInt [[Ty_35:[0-9]+]] 35 0
; CHECK-SPIRV: 4 TypeInt [[Ty_28:[0-9]+]] 28 0
; CHECK-SPIRV: 4 TypeInt [[Ty_31:[0-9]+]] 31 0
; CHECK-SPIRV: 4 TypeInt [[Ty_40:[0-9]+]] 40 0
; CHECK-SPIRV: 4 TypeInt [[Ty_60:[0-9]+]] 60 0
; CHECK-SPIRV: 4 TypeInt [[Ty_16:[0-9]+]] 16 0
; CHECK-SPIRV: 4 TypeInt [[Ty_64:[0-9]+]] 64 0
; CHECK-SPIRV: 4 TypeInt [[Ty_44:[0-9]+]] 44 0
; CHECK-SPIRV: 4 TypeInt [[Ty_34:[0-9]+]] 34 0
; CHECK-SPIRV: 4 TypeInt [[Ty_66:[0-9]+]] 66 0
; CHECK-SPIRV: 4 TypeInt [[Ty_68:[0-9]+]] 68 0

; CHECK-SPIRV: 6 Load [[Ty_13]] [[Sqrt_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSqrtINTEL [[Ty_5]] [[#]] [[Sqrt_InId]] 0 2 2 0 0
; CHECK-SPIRV: 6 Load [[Ty_5]] [[Sqrt_InId_B:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSqrtINTEL [[Ty_13]] [[#]] [[Sqrt_InId_B]] 0 2 2 0 0
; CHECK-SPIRV: 6 Load [[Ty_5]] [[Sqrt_InId_C:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSqrtINTEL [[Ty_13]] [[#]] [[Sqrt_InId_C]] 0 2 2 0 0

; CHECK-SPIRV: 6 Load [[Ty_3]] [[Recip_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedRecipINTEL [[Ty_8]] [[#]] [[Recip_InId]] 1 4 4 0 0

; CHECK-SPIRV: 6 Load [[Ty_11]] [[Rsqrt_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedRsqrtINTEL [[Ty_10]] [[#]] [[Rsqrt_InId]] 0 8 6 0 0

; CHECK-SPIRV: 6 Load [[Ty_17]] [[Sin_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSinINTEL [[Ty_11]] [[#]] [[Sin_InId]] 1 7 5 0 0

; CHECK-SPIRV: 6 Load [[Ty_35]] [[Cos_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedCosINTEL [[Ty_28]] [[#]] [[Cos_InId]] 0 9 3 0 0

; CHECK-SPIRV: 6 Load [[Ty_31]] [[SinCos_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSinCosINTEL [[Ty_40]] [[#]] [[SinCos_InId]] 1 10 12 0 0

; CHECK-SPIRV: 6 Load [[Ty_60]] [[SinPi_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSinPiINTEL [[Ty_5]] [[#]] [[SinPi_InId]] 0 2 2 0 0

; CHECK-SPIRV: 6 Load [[Ty_28]] [[CosPi_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedCosPiINTEL [[Ty_16]] [[#]] [[CosPi_InId]] 0 8 5 0 0

; CHECK-SPIRV: 6 Load [[Ty_13]] [[SinCosPi_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSinCosPiINTEL [[Ty_10]] [[#]] [[SinCosPi_InId]] 0 2 2 0 0

; CHECK-SPIRV: 6 Load [[Ty_64]] [[Log_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedLogINTEL [[Ty_44]] [[#]] [[Log_InId]] 1 24 22 0 0

; CHECK-SPIRV: 6 Load [[Ty_44]] [[Exp_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedExpINTEL [[Ty_34]] [[#]] [[Exp_InId]] 0 20 20 0 0

; CHECK-SPIRV: 6 Load [[Ty_34]] [[SinCos_InId:[0-9]+]]
; CHECK-SPIRV-NEXT: 9 FixedSinCosINTEL [[Ty_66]] [[SinCos_ResultId:[0-9]+]] [[SinCos_InId]] 1 3 2 0 0
; CHECK-SPIRV: 3 Store [[#]] [[SinCos_ResultId]]

; CHECK-SPIRV: 6 Load [[Ty_68]] [[ResId:[0-9]+]]
; CHECK-SPIRV-NEXT: 5 Store [[PtrId:[0-9]+]] [[ResId]]
; CHECK-SPIRV-NEXT: 4 Load [[Ty_68]] [[ExpInId2:[0-9]+]] [[PtrId]]
; CHECK-SPIRV-NEXT: 9 FixedExpINTEL [[Ty_68]] [[#]] [[ExpInId2]] 0 20 20 0 0

; CHECK-LLVM: call i5 @intel_arbitrary_fixed_sqrt.i5.i13(i13 %[[#]], i1 false, i32 2, i32 2, i32 0, i32 0)
; CHECK-LLVM: call i13 @intel_arbitrary_fixed_sqrt.i13.i5(i5 %[[#]], i1 false, i32 2, i32 2, i32 0, i32 0)
; CHECK-LLVM: call i13 @intel_arbitrary_fixed_sqrt.i13.i5(i5 %[[#]], i1 false, i32 2, i32 2, i32 0, i32 0)
; CHECK-LLVM: call i8 @intel_arbitrary_fixed_recip.i8.i3(i3 %[[#]], i1 true, i32 4, i32 4, i32 0, i32 0)
; CHECK-LLVM: call i10 @intel_arbitrary_fixed_rsqrt.i10.i11(i11 %[[#]], i1 false, i32 8, i32 6, i32 0, i32 0)
; CHECK-LLVM: call i11 @intel_arbitrary_fixed_sin.i11.i17(i17 %[[#]], i1 true, i32 7, i32 5, i32 0, i32 0)
; CHECK-LLVM: call i28 @intel_arbitrary_fixed_cos.i28.i35(i35 %[[#]], i1 false, i32 9, i32 3, i32 0, i32 0)
; CHECK-LLVM: call i40 @intel_arbitrary_fixed_sincos.i40.i31(i31 %[[#]], i1 true, i32 10, i32 12, i32 0, i32 0)
; CHECK-LLVM: call i5 @intel_arbitrary_fixed_sinpi.i5.i60(i60 %[[#]], i1 false, i32 2, i32 2, i32 0, i32 0)
; CHECK-LLVM: call i16 @intel_arbitrary_fixed_cospi.i16.i28(i28 %[[#]], i1 false, i32 8, i32 5, i32 0, i32 0)
; CHECK-LLVM: call i10 @intel_arbitrary_fixed_sincospi.i10.i13(i13 %[[#]], i1 false, i32 2, i32 2, i32 0, i32 0)
; CHECK-LLVM: call i44 @intel_arbitrary_fixed_log.i44.i64(i64 %[[#]], i1 true, i32 24, i32 22, i32 0, i32 0)
; CHECK-LLVM: call i34 @intel_arbitrary_fixed_exp.i34.i44(i44 %[[#]], i1 false, i32 20, i32 20, i32 0, i32 0)
; CHECK-LLVM: call void @intel_arbitrary_fixed_sincos.i66.i34(i66 addrspace(4)* sret(i66) %[[#]], i34 %[[#]], i1 true, i32 3, i32 2, i32 0, i32 0)
; CHECK-LLVM: call void @intel_arbitrary_fixed_exp.i68.i68(i68 addrspace(4)* sret(i68) %[[#]], i68 %[[#]], i1 false, i32 20, i32 20, i32 0, i32 0)

; ModuleID = 'ap_fixed.cpp'
source_filename = "ap_fixed.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

$_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv = comdat any

$_Z5recipILi3ELi8ELb1ELi4ELi4EEvv = comdat any

$_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv = comdat any

$_Z3sinILi17ELi11ELb1ELi7ELi5EEvv = comdat any

$_Z3cosILi35ELi28ELb0ELi9ELi3EEvv = comdat any

$_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv = comdat any

$_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv = comdat any

$_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv = comdat any

$_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv = comdat any

$_Z3logILi64ELi44ELb1ELi24ELi22EEvv = comdat any

$_Z3expILi44ELi34ELb0ELi20ELi20EEvv = comdat any

$_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_ = comdat any

$_Z3expILi68ELi68ELb0ELi20ELi20EEvv = comdat any

; Function Attrs: norecurse
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %1 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #5
  %2 = addrspacecast %"class._ZTSZ4mainE3$_0.anon"* %0 to %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %2)
  %3 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #5
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint norecurse
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this) #2 align 2 {
entry:
  %this.addr = alloca %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, align 8
  store %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8, !tbaa !5
  call spir_func void @_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z5recipILi3ELi8ELb1ELi4ELi4EEvv()
  call spir_func void @_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv()
  call spir_func void @_Z3sinILi17ELi11ELb1ELi7ELi5EEvv()
  call spir_func void @_Z3cosILi35ELi28ELb0ELi9ELi3EEvv()
  call spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv()
  call spir_func void @_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv()
  call spir_func void @_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv()
  call spir_func void @_Z3logILi64ELi44ELb1ELi24ELi22EEvv()
  call spir_func void @_Z3expILi44ELi34ELb0ELi20ELi20EEvv()
  call spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_()
  call spir_func void @_Z3expILi68ELi68ELb0ELi20ELi20EEvv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z4sqrtILi13ELi5ELb0ELi2ELi2EEvv() #3 comdat {
entry:
  %a = alloca i13, align 2
  %ap_fixed_Sqrt = alloca i5, align 1
  %b = alloca i5, align 1
  %ap_fixed_Sqrt_b = alloca i13, align 2
  %c = alloca i5, align 1
  %ap_fixed_Sqrt_c = alloca i13, align 2
  %0 = bitcast i13* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %0) #5
  %1 = bitcast i5* %ap_fixed_Sqrt to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #5
  %2 = load i13, i13* %a, align 2, !tbaa !9
  %call = call spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %2, i1 zeroext false, i32 2, i32 2, i32 0, i32 0) #5
  store i5 %call, i5* %ap_fixed_Sqrt, align 1, !tbaa !11
  %3 = bitcast i5* %b to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %3) #5
  %4 = bitcast i13* %ap_fixed_Sqrt_b to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %4) #5
  %5 = load i5, i5* %b, align 1, !tbaa !11
  %call1 = call spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext %5, i1 zeroext false, i32 2, i32 2, i32 0, i32 0) #5
  store i13 %call1, i13* %ap_fixed_Sqrt_b, align 2, !tbaa !9
  %6 = bitcast i5* %c to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %6) #5
  %7 = bitcast i13* %ap_fixed_Sqrt_c to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %7) #5
  %8 = load i5, i5* %c, align 1, !tbaa !11
  %call2 = call spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext %8, i1 zeroext false, i32 2, i32 2, i32 0, i32 0) #5
  store i13 %call2, i13* %ap_fixed_Sqrt_c, align 2, !tbaa !9
  %9 = bitcast i13* %ap_fixed_Sqrt_c to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %9) #5
  %10 = bitcast i5* %c to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %10) #5
  %11 = bitcast i13* %ap_fixed_Sqrt_b to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %11) #5
  %12 = bitcast i5* %b to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %12) #5
  %13 = bitcast i5* %ap_fixed_Sqrt to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %13) #5
  %14 = bitcast i13* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %14) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z5recipILi3ELi8ELb1ELi4ELi4EEvv() #3 comdat {
entry:
  %a = alloca i3, align 1
  %ap_fixed_Recip = alloca i8, align 1
  %0 = bitcast i3* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #5
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %ap_fixed_Recip) #5
  %1 = load i3, i3* %a, align 1, !tbaa !13
  %call = call spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext %1, i1 zeroext true, i32 4, i32 4, i32 0, i32 0) #5
  store i8 %call, i8* %ap_fixed_Recip, align 1, !tbaa !15
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %ap_fixed_Recip) #5
  %2 = bitcast i3* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z5rsqrtILi11ELi10ELb0ELi8ELi6EEvv() #3 comdat {
entry:
  %a = alloca i11, align 2
  %ap_fixed_Rsqrt = alloca i10, align 2
  %0 = bitcast i11* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %0) #5
  %1 = bitcast i10* %ap_fixed_Rsqrt to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %1) #5
  %2 = load i11, i11* %a, align 2, !tbaa !17
  %call = call spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext %2, i1 zeroext false, i32 8, i32 6, i32 0, i32 0) #5
  store i10 %call, i10* %ap_fixed_Rsqrt, align 2, !tbaa !19
  %3 = bitcast i10* %ap_fixed_Rsqrt to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %3) #5
  %4 = bitcast i11* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3sinILi17ELi11ELb1ELi7ELi5EEvv() #3 comdat {
entry:
  %a = alloca i17, align 4
  %ap_fixed_Sin = alloca i11, align 2
  %0 = bitcast i17* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #5
  %1 = bitcast i11* %ap_fixed_Sin to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %1) #5
  %2 = load i17, i17* %a, align 4, !tbaa !21
  %call = call spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext %2, i1 zeroext true, i32 7, i32 5, i32 0, i32 0) #5
  store i11 %call, i11* %ap_fixed_Sin, align 2, !tbaa !17
  %3 = bitcast i11* %ap_fixed_Sin to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %3) #5
  %4 = bitcast i17* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3cosILi35ELi28ELb0ELi9ELi3EEvv() #3 comdat {
entry:
  %a = alloca i35, align 8
  %ap_fixed_Cos = alloca i28, align 4
  %0 = bitcast i35* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #5
  %1 = bitcast i28* %ap_fixed_Cos to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #5
  %2 = load i35, i35* %a, align 8, !tbaa !23
  %call = call spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35 %2, i1 zeroext false, i32 9, i32 3, i32 0, i32 0) #5
  store i28 %call, i28* %ap_fixed_Cos, align 4, !tbaa !25
  %3 = bitcast i28* %ap_fixed_Cos to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #5
  %4 = bitcast i35* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv() #3 comdat {
entry:
  %a = alloca i31, align 4
  %ap_fixed_SinCos = alloca i40, align 8
  %0 = bitcast i31* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #5
  %1 = bitcast i40* %ap_fixed_SinCos to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #5
  %2 = load i31, i31* %a, align 4, !tbaa !27
  %call = call spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext %2, i1 zeroext true, i32 10, i32 12, i32 0, i32 0) #5
  store i40 %call, i40* %ap_fixed_SinCos, align 8, !tbaa !29
  %3 = bitcast i40* %ap_fixed_SinCos to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #5
  %4 = bitcast i31* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z6sin_piILi60ELi5ELb0ELi2ELi2EEvv() #3 comdat {
entry:
  %a = alloca i60, align 8
  %ap_fixed_SinPi = alloca i5, align 1
  %0 = bitcast i60* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #5
  %1 = bitcast i5* %ap_fixed_SinPi to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #5
  %2 = load i60, i60* %a, align 8, !tbaa !31
  %call = call spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60 %2, i1 zeroext false, i32 2, i32 2, i32 0, i32 0) #5
  store i5 %call, i5* %ap_fixed_SinPi, align 1, !tbaa !11
  %3 = bitcast i5* %ap_fixed_SinPi to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #5
  %4 = bitcast i60* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z6cos_piILi28ELi16ELb0ELi8ELi5EEvv() #3 comdat {
entry:
  %a = alloca i28, align 4
  %ap_fixed_CosPi = alloca i16, align 2
  %0 = bitcast i28* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #5
  %1 = bitcast i16* %ap_fixed_CosPi to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %1) #5
  %2 = load i28, i28* %a, align 4, !tbaa !25
  %call = call spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext %2, i1 zeroext false, i32 8, i32 5, i32 0, i32 0) #5
  store i16 %call, i16* %ap_fixed_CosPi, align 2, !tbaa !33
  %3 = bitcast i16* %ap_fixed_CosPi to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %3) #5
  %4 = bitcast i28* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z10sin_cos_piILi13ELi5ELb0ELi2ELi2EEvv() #3 comdat {
entry:
  %a = alloca i13, align 2
  %ap_fixed_SinCosPi = alloca i10, align 2
  %0 = bitcast i13* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %0) #5
  %1 = bitcast i10* %ap_fixed_SinCosPi to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* %1) #5
  %2 = load i13, i13* %a, align 2, !tbaa !9
  %call = call spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext %2, i1 zeroext false, i32 2, i32 2, i32 0, i32 0) #5
  store i10 %call, i10* %ap_fixed_SinCosPi, align 2, !tbaa !19
  %3 = bitcast i10* %ap_fixed_SinCosPi to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %3) #5
  %4 = bitcast i13* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 2, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3logILi64ELi44ELb1ELi24ELi22EEvv() #3 comdat {
entry:
  %a = alloca i64, align 8
  %ap_fixed_Log = alloca i44, align 8
  %0 = bitcast i64* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #5
  %1 = bitcast i44* %ap_fixed_Log to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #5
  %2 = load i64, i64* %a, align 8, !tbaa !35
  %call = call spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64 %2, i1 zeroext true, i32 24, i32 22, i32 0, i32 0) #5
  store i44 %call, i44* %ap_fixed_Log, align 8, !tbaa !37
  %3 = bitcast i44* %ap_fixed_Log to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #5
  %4 = bitcast i64* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3expILi44ELi34ELb0ELi20ELi20EEvv() #3 comdat {
entry:
  %a = alloca i44, align 8
  %ap_fixed_Exp = alloca i34, align 8
  %0 = bitcast i44* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #5
  %1 = bitcast i34* %ap_fixed_Exp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %1) #5
  %2 = load i44, i44* %a, align 8, !tbaa !37
  %call = call spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44 %2, i1 zeroext false, i32 20, i32 20, i32 0, i32 0) #5
  store i34 %call, i34* %ap_fixed_Exp, align 8, !tbaa !39
  %3 = bitcast i34* %ap_fixed_Exp to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #5
  %4 = bitcast i44* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #5
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z7sin_cosILi31ELi20ELb1ELi10ELi12EEvv_() #3 comdat {
entry:
  %0 = alloca i34, align 8
  %1 = addrspacecast i34* %0 to i34 addrspace(4)*
  %2 = alloca i66, align 8
  %3 = addrspacecast i66* %2 to i66 addrspace(4)*
  %4 = bitcast i34* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %4)
  %5 = bitcast i66* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %5)
  %6 = load i34, i34 addrspace(4)* %1, align 8
  call spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi66EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i66 addrspace(4)* sret(i66) align 8 %3, i34 %6, i1 zeroext true, i32 3, i32 2, i32 0, i32 0) #5
  %7 = load i66, i66 addrspace(4)* %3, align 8
  store i66 %7, i66 addrspace(4)* %3, align 8
  %8 = bitcast i66* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %8)
  %9 = bitcast i34* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %9)
  ret void
}

; Function Attrs: norecurse nounwind
define linkonce_odr dso_local spir_func void @_Z3expILi68ELi68ELb0ELi20ELi20EEvv() #3 comdat {
entry:
  %a = alloca i68, align 8
  %a.ascast = addrspacecast i68* %a to i68 addrspace(4)*
  %ap_fixed_Exp = alloca i68, align 8
  %ap_fixed_Exp.ascast = addrspacecast i68* %ap_fixed_Exp to i68 addrspace(4)*
  %tmp = alloca i68, align 8
  %tmp.ascast = addrspacecast i68* %tmp to i68 addrspace(4)*
  %indirect-arg-temp = alloca i68, align 8
  %0 = bitcast i68* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %0)
  %1 = bitcast i68* %ap_fixed_Exp to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %1)
  %2 = load i68, i68 addrspace(4)* %a.ascast, align 8
  store i68 %2, i68* %indirect-arg-temp, align 8
  call spir_func void @_Z21__spirv_FixedExpINTELILi68ELi68EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i68 addrspace(4)* sret(i68) align 8 %tmp.ascast, i68* byval(i68) align 8 %indirect-arg-temp, i1 zeroext false, i32 20, i32 20, i32 0, i32 0) #4
  %3 = load i68, i68 addrspace(4)* %tmp.ascast, align 8
  store i68 %3, i68 addrspace(4)* %ap_fixed_Exp.ascast, align 8
  %4 = bitcast i68* %ap_fixed_Exp to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %4)
  %5 = bitcast i68* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %5)
  ret void
}


; Function Attrs: nounwind
declare dso_local spir_func signext i5 @_Z22__spirv_FixedSqrtINTELILi13ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i13 @_Z22__spirv_FixedSqrtINTELILi5ELi13EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i5 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i8 @_Z23__spirv_FixedRecipINTELILi3ELi8EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i3 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i10 @_Z23__spirv_FixedRsqrtINTELILi11ELi10EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i11 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i11 @_Z21__spirv_FixedSinINTELILi17ELi11EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i17 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i28 @_Z21__spirv_FixedCosINTELILi35ELi28EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i35, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func i40 @_Z24__spirv_FixedSinCosINTELILi31ELi20EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i31 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i5 @_Z23__spirv_FixedSinPiINTELILi60ELi5EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i60, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i16 @_Z23__spirv_FixedCosPiINTELILi28ELi16EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i28 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func signext i10 @_Z26__spirv_FixedSinCosPiINTELILi13ELi5EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i13 signext, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func i44 @_Z21__spirv_FixedLogINTELILi64ELi44EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i64, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func i34 @_Z21__spirv_FixedExpINTELILi44ELi34EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i44, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: nounwind
declare dso_local spir_func void @_Z24__spirv_FixedSinCosINTELILi34ELi66EEU7_ExtIntIXmlLi2ET0_EEiU7_ExtIntIXT_EEibiiii(i66 addrspace(4)* sret(i66) align 8, i34, i1 zeroext, i32, i32, i32, i32) #4

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z21__spirv_FixedExpINTELILi68ELi68EEU7_ExtIntIXT0_EEiU7_ExtIntIXT_EEibiiii(i68 addrspace(4)* sret(i68) align 8, i68* byval(i68) align 8, i1 zeroext, i32, i32, i32, i32) #4

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="ap_fixed.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"_ExtInt(13)", !7, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"_ExtInt(5)", !7, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"_ExtInt(3)", !7, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"_ExtInt(8)", !7, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"_ExtInt(11)", !7, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"_ExtInt(10)", !7, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"_ExtInt(17)", !7, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"_ExtInt(35)", !7, i64 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"_ExtInt(28)", !7, i64 0}
!27 = !{!28, !28, i64 0}
!28 = !{!"_ExtInt(31)", !7, i64 0}
!29 = !{!30, !30, i64 0}
!30 = !{!"_ExtInt(40)", !7, i64 0}
!31 = !{!32, !32, i64 0}
!32 = !{!"_ExtInt(60)", !7, i64 0}
!33 = !{!34, !34, i64 0}
!34 = !{!"_ExtInt(16)", !7, i64 0}
!35 = !{!36, !36, i64 0}
!36 = !{!"_ExtInt(64)", !7, i64 0}
!37 = !{!38, !38, i64 0}
!38 = !{!"_ExtInt(44)", !7, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"_ExtInt(34)", !7, i64 0}
