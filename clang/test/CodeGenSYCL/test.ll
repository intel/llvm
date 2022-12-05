; ModuleID = 'unique_stable_name.cpp'
source_filename = "unique_stable_name.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%class.anon = type { i8 }
%class.anon.0 = type { i8 }
%class.anon.2 = type { i8 }
%class.anon.4 = type { i8 }
%class.anon.6 = type { i8 }
%class.anon.8 = type { i8 }
%class.anon.10 = type { i8 }
%class.anon.12 = type { i8 }
%class.anon.14 = type { i8 }
%class.anon.16 = type { i8 }
%class.anon.18 = type { i8 }
%class.anon.20 = type { i8 }
%class.anon.22 = type { i8 }
%class.anon.24 = type { i8 }
%class.anon.26 = type { i8 }
%class.anon.28 = type { i8 }
%class.anon.30 = type { i8 }

$_Z14template_paramIiEvv = comdat any

$_Z28lambda_in_dependent_functionIiEvv = comdat any

$_Z13lambda_no_depIidEvT_T0_ = comdat any

@__usn_str = private unnamed_addr addrspace(1) constant [6 x i8] c"_ZTSi\00", align 1
@__usn_str.1 = private unnamed_addr addrspace(1) constant [31 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE_\00", align 1
@__usn_str.2 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE0_\00", align 1
@__usn_str.3 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE1_\00", align 1
@__usn_str.4 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE2_\00", align 1
@__usn_str.5 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE3_\00", align 1
@__usn_str.6 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE4_\00", align 1
@__usn_str.7 = private unnamed_addr addrspace(1) constant [32 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE5_\00", align 1
@__usn_str.8 = private unnamed_addr addrspace(1) constant [6 x i8] c"_ZTSi\00", align 1
@__usn_str.9 = private unnamed_addr addrspace(1) constant [31 x i8] c"_ZTSZZ4mainENKUlvE0_clEvEUlvE_\00", align 1
@__usn_str.10 = private unnamed_addr addrspace(1) constant [47 x i8] c"_ZTSZ28lambda_in_dependent_functionIiEvvEUlvE_\00", align 1
@__usn_str.11 = private unnamed_addr addrspace(1) constant [72 x i8] c"_ZTSZ28lambda_in_dependent_functionIZZ4mainENKUlvE0_clEvEUlvE_EvvEUlvE_\00", align 1
@__usn_str.12 = private unnamed_addr addrspace(1) constant [38 x i8] c"_ZTSZ13lambda_no_depIidEvT_T0_EUlidE_\00", align 1
@__usn_str.13 = private unnamed_addr addrspace(1) constant [81 x i8] c"_ZTSZ14lambda_two_depIZZ4mainENKUlvE0_clEvEUliE_ZZ4mainENKS0_clEvEUldE_EvvEUlvE_\00", align 1
@__usn_str.14 = private unnamed_addr addrspace(1) constant [81 x i8] c"_ZTSZ14lambda_two_depIZZ4mainENKUlvE0_clEvEUldE_ZZ4mainENKS0_clEvEUliE_EvvEUlvE_\00", align 1

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local spir_kernel void @_ZTSZ4mainEUlPZ4mainEUlvE_E_() #0 !srcloc !4 !kernel_arg_buffer_location !5 {
entry:
  %retval.i = alloca i32, align 4
  %this.addr.i = alloca %class.anon addrspace(4)*, align 8
  %l.addr.i = alloca %class.anon.0 addrspace(4)*, align 8
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  %0 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  %retval.ascast.i = addrspacecast i32* %retval.i to i32 addrspace(4)*
  %this.addr.ascast.i = addrspacecast %class.anon addrspace(4)** %this.addr.i to %class.anon addrspace(4)* addrspace(4)*
  %l.addr.ascast.i = addrspacecast %class.anon.0 addrspace(4)** %l.addr.i to %class.anon.0 addrspace(4)* addrspace(4)*
  store %class.anon addrspace(4)* %__SYCLKernel.ascast, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !6
  store %class.anon.0 addrspace(4)* null, %class.anon.0 addrspace(4)* addrspace(4)* %l.addr.ascast.i, align 8, !tbaa !6
  %this1.i = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %1 = bitcast %class.anon* %__SYCLKernel to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local spir_kernel void @_ZTSZ4mainEUlvE0_() #0 !srcloc !10 !kernel_arg_buffer_location !5 {
entry:
  %this.addr.i = alloca %class.anon.2 addrspace(4)*, align 8
  %x.i = alloca %class.anon.4, align 1
  %MACRO_X.i = alloca %class.anon.6, align 1
  %MACRO_Y.i = alloca %class.anon.8, align 1
  %MACRO_X2.i = alloca %class.anon.10, align 1
  %MACRO_Y3.i = alloca %class.anon.12, align 1
  %MACRO_X4.i = alloca %class.anon.14, align 1
  %MACRO_Y5.i = alloca %class.anon.16, align 1
  %a.i = alloca i32, align 4
  %b.i = alloca double, align 8
  %y.i = alloca %class.anon.18, align 1
  %z.i = alloca %class.anon.20, align 1
  %__SYCLKernel = alloca %class.anon.2, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon.2* %__SYCLKernel to %class.anon.2 addrspace(4)*
  %0 = bitcast %class.anon.2* %__SYCLKernel to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  %this.addr.ascast.i = addrspacecast %class.anon.2 addrspace(4)** %this.addr.i to %class.anon.2 addrspace(4)* addrspace(4)*
  %x.ascast.i = addrspacecast %class.anon.4* %x.i to %class.anon.4 addrspace(4)*
  %MACRO_X.ascast.i = addrspacecast %class.anon.6* %MACRO_X.i to %class.anon.6 addrspace(4)*
  %MACRO_Y.ascast.i = addrspacecast %class.anon.8* %MACRO_Y.i to %class.anon.8 addrspace(4)*
  %MACRO_X2.ascast.i = addrspacecast %class.anon.10* %MACRO_X2.i to %class.anon.10 addrspace(4)*
  %MACRO_Y3.ascast.i = addrspacecast %class.anon.12* %MACRO_Y3.i to %class.anon.12 addrspace(4)*
  %MACRO_X4.ascast.i = addrspacecast %class.anon.14* %MACRO_X4.i to %class.anon.14 addrspace(4)*
  %MACRO_Y5.ascast.i = addrspacecast %class.anon.16* %MACRO_Y5.i to %class.anon.16 addrspace(4)*
  %a.ascast.i = addrspacecast i32* %a.i to i32 addrspace(4)*
  %b.ascast.i = addrspacecast double* %b.i to double addrspace(4)*
  %y.ascast.i = addrspacecast %class.anon.18* %y.i to %class.anon.18 addrspace(4)*
  %z.ascast.i = addrspacecast %class.anon.20* %z.i to %class.anon.20 addrspace(4)*
  store %class.anon.2 addrspace(4)* %__SYCLKernel.ascast, %class.anon.2 addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8, !tbaa !6
  %this1.i = load %class.anon.2 addrspace(4)*, %class.anon.2 addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(1)* @__usn_str, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.4* %x.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([31 x i8], [31 x i8] addrspace(1)* @__usn_str.1, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %2 = bitcast %class.anon.6* %MACRO_X.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %2) #3
  %3 = bitcast %class.anon.8* %MACRO_Y.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %3) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.2, i32 0, i32 0) to i8 addrspace(4)*)) #4
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.3, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %4 = bitcast %class.anon.10* %MACRO_X2.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %4) #3
  %5 = bitcast %class.anon.12* %MACRO_Y3.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %5) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.4, i32 0, i32 0) to i8 addrspace(4)*)) #4
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.5, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %6 = bitcast %class.anon.12* %MACRO_Y3.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %6) #3
  %7 = bitcast %class.anon.10* %MACRO_X2.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %7) #3
  %8 = bitcast %class.anon.14* %MACRO_X4.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %8) #3
  %9 = bitcast %class.anon.16* %MACRO_Y5.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %9) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.6, i32 0, i32 0) to i8 addrspace(4)*)) #4
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(1)* @__usn_str.7, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %10 = bitcast %class.anon.16* %MACRO_Y5.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %10) #3
  %11 = bitcast %class.anon.14* %MACRO_X4.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %11) #3
  call spir_func void @_Z14template_paramIiEvv() #4
  call spir_func void @_Z14template_paramIZZ4mainENKUlvE0_clEvEUlvE_Evv() #4
  call spir_func void @_Z28lambda_in_dependent_functionIiEvv() #4
  call spir_func void @_Z28lambda_in_dependent_functionIZZ4mainENKUlvE0_clEvEUlvE_Evv() #4
  call spir_func void @_Z13lambda_no_depIidEvT_T0_(i32 noundef 3, double noundef 5.500000e+00) #4
  %12 = bitcast i32* %a.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %12) #3
  store i32 5, i32 addrspace(4)* %a.ascast.i, align 4, !tbaa !11
  %13 = bitcast double* %b.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %13) #3
  store double 1.070000e+01, double addrspace(4)* %b.ascast.i, align 8, !tbaa !13
  %14 = bitcast %class.anon.18* %y.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %14) #3
  %15 = bitcast %class.anon.20* %z.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %15) #3
  call spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUliE_ZZ4mainENKS0_clEvEUldE_Evv() #4
  call spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUldE_ZZ4mainENKS0_clEvEUliE_Evv() #4
  %16 = bitcast %class.anon.20* %z.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %16) #3
  %17 = bitcast %class.anon.18* %y.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %17) #3
  %18 = bitcast double* %b.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %18) #3
  %19 = bitcast i32* %a.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %19) #3
  %20 = bitcast %class.anon.8* %MACRO_Y.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %20) #3
  %21 = bitcast %class.anon.6* %MACRO_X.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %21) #3
  %22 = bitcast %class.anon.4* %x.i to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %22) #3
  %23 = bitcast %class.anon.2* %__SYCLKernel to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %23) #3
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define dso_local spir_func void @puts(i8 addrspace(4)* noundef %0) #2 !srcloc !15 {
entry:
  %.addr = alloca i8 addrspace(4)*, align 8
  %.addr.ascast = addrspacecast i8 addrspace(4)** %.addr to i8 addrspace(4)* addrspace(4)*
  store i8 addrspace(4)* %0, i8 addrspace(4)* addrspace(4)* %.addr.ascast, align 8, !tbaa !6
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr spir_func void @_Z14template_paramIiEvv() #2 comdat !srcloc !16 {
entry:
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(1)* @__usn_str.8, i32 0, i32 0) to i8 addrspace(4)*)) #4
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define internal spir_func void @_Z14template_paramIZZ4mainENKUlvE0_clEvEUlvE_Evv() #2 !srcloc !16 {
entry:
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([31 x i8], [31 x i8] addrspace(1)* @__usn_str.9, i32 0, i32 0) to i8 addrspace(4)*)) #4
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr spir_func void @_Z28lambda_in_dependent_functionIiEvv() #2 comdat !srcloc !17 {
entry:
  %y = alloca %class.anon.22, align 1
  %y.ascast = addrspacecast %class.anon.22* %y to %class.anon.22 addrspace(4)*
  %0 = bitcast %class.anon.22* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([47 x i8], [47 x i8] addrspace(1)* @__usn_str.10, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.22* %y to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define internal spir_func void @_Z28lambda_in_dependent_functionIZZ4mainENKUlvE0_clEvEUlvE_Evv() #2 !srcloc !17 {
entry:
  %y = alloca %class.anon.24, align 1
  %y.ascast = addrspacecast %class.anon.24* %y to %class.anon.24 addrspace(4)*
  %0 = bitcast %class.anon.24* %y to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([72 x i8], [72 x i8] addrspace(1)* @__usn_str.11, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.24* %y to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr spir_func void @_Z13lambda_no_depIidEvT_T0_(i32 noundef %a, double noundef %b) #2 comdat !srcloc !18 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca double, align 8
  %p = alloca %class.anon.26, align 1
  %a.addr.ascast = addrspacecast i32* %a.addr to i32 addrspace(4)*
  %b.addr.ascast = addrspacecast double* %b.addr to double addrspace(4)*
  %p.ascast = addrspacecast %class.anon.26* %p to %class.anon.26 addrspace(4)*
  store i32 %a, i32 addrspace(4)* %a.addr.ascast, align 4, !tbaa !11
  store double %b, double addrspace(4)* %b.addr.ascast, align 8, !tbaa !13
  %0 = bitcast %class.anon.26* %p to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([38 x i8], [38 x i8] addrspace(1)* @__usn_str.12, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.26* %p to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define internal spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUliE_ZZ4mainENKS0_clEvEUldE_Evv() #2 !srcloc !19 {
entry:
  %z = alloca %class.anon.28, align 1
  %z.ascast = addrspacecast %class.anon.28* %z to %class.anon.28 addrspace(4)*
  %0 = bitcast %class.anon.28* %z to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([81 x i8], [81 x i8] addrspace(1)* @__usn_str.13, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.28* %z to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define internal spir_func void @_Z14lambda_two_depIZZ4mainENKUlvE0_clEvEUldE_ZZ4mainENKS0_clEvEUliE_Evv() #2 !srcloc !19 {
entry:
  %z = alloca %class.anon.30, align 1
  %z.ascast = addrspacecast %class.anon.30* %z to %class.anon.30 addrspace(4)*
  %0 = bitcast %class.anon.30* %z to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %0) #3
  call spir_func void @puts(i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* getelementptr inbounds ([81 x i8], [81 x i8] addrspace(1)* @__usn_str.14, i32 0, i32 0) to i8 addrspace(4)*)) #4
  %1 = bitcast %class.anon.30* %z to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %1) #3
  ret void
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="unique_stable_name.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { convergent mustprogress norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 16.0.0 (https://github.com/otcshare/llvm 36cb14095d98d379a35023a9123f16137ab6e206)"}
!4 = !{i32 4643}
!5 = !{}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{i32 6311}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !8, i64 0}
!15 = !{i32 2816}
!16 = !{i32 2866}
!17 = !{i32 2961}
!18 = !{i32 3249}
!19 = !{i32 3112}
