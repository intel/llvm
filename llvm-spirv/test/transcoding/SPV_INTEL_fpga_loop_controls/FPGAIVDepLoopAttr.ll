; This LLVM IR was generated using Intel SYCL Clang compiler (https://github.com/intel/llvm)
;
; SYCL source code of this program can be found below.
;
; void ivdep_embedded_multiple_dimensions() {
;   int a[10];
;   int b[10];
;   [[intelfpga::ivdep]]
;   for (int i = 0; i != 10; ++i) {
;     a[i] = i;
;     b[i] = i;
;     [[intelfpga::ivdep]]
;     for (int j = 0; j != 10; ++j) {
;       a[j] += j;
;       b[j] += j;
;       [[intelfpga::ivdep]]
;       for (int k = 0; k != 10; ++k) {
;         a[k] += k;
;         b[k] += k;
;       }
;     }
;   }
; }
;
; void ivdep_mul_arrays_and_global() {
;   int a[10];
;   int b[10];
;   int c[10];
;   int d[10];
;   [[intelfpga::ivdep(5)]]
;   [[intelfpga::ivdep(b, 6)]]
;   [[intelfpga::ivdep(c)]]
;   for (int i = 0; i != 10; ++i) {
;     a[i] = 0;
;     b[i] = 0;
;     c[i] = 0;
;     d[i] = 0;
;   }
; }
;
; void ivdep_embedded_inner_access() {
;   int a[10][10][10];
;   int b[10][10][10];
;   [[intelfpga::ivdep]]
;   for (int i = 0; i != 10; ++i) {
;     [[intelfpga::ivdep]]
;     for (int j = 0; j != 10; ++j) {
;       [[intelfpga::ivdep]]
;       for (int k = 0; k != 10; ++k) {
;         a[i][j][k] = b[i][j][k];
;       }
;     }
;   }
; }
;
; template <typename name, typename Func>
; __attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
;   kernelFunc();
; }
;
; int main() {
;   kernel_single_task<class kernel_function>([]() {
;     ivdep_embedded_multiple_dimensions();
;     ivdep_mul_arrays_and_global();
;     ivdep_embedded_inner_access();
;   });
;   return 0;
; }

; RUN: llvm-as < %s > %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_loop_controls -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll

; CHECK-LLVM is the base prefix, which includes simple checks for
; "llvm.loop.parallel_access_indices" MD nodes with only 1 index group operand
; CHECK-LLVM-MD-OP-<N> is the group of prefixes to check for more
; complicated cases of "llvm.loop.parallel_access_indices" nodes, the ones
; containing multiple index group operands that could come in indeterminate order
; RUN: FileCheck < %t.rev.ll %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-MD-OP1
; RUN: FileCheck < %t.rev.ll %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-MD-OP2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%"class._ZTSZ4mainE3$_0.anon" = type { i8 }

; CHECK-SPIRV: 2 Capability FPGALoopControlsINTEL
; CHECK-SPIRV: 9 Extension "SPV_INTEL_fpga_loop_controls"

; CHECK-SPIRV-DAG: TypeInt [[TYPE_INT_64:[0-9]+]] 64 0
; CHECK-SPIRV-DAG: TypeInt [[TYPE_INT_32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_64]] [[SIZE:[0-9]+]] 10 0
; CHECK-SPIRV-DAG: TypeArray [[TYPE_ARRAY:[0-9]+]] [[TYPE_INT_32]] [[SIZE]]
; CHECK-SPIRV-DAG: TypePointer [[TYPE_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_ARRAY]]
; CHECK-SPIRV-DAG: TypeArray [[TYPE_2_DIM_ARRAY:[0-9]+]] [[TYPE_ARRAY]] [[SIZE]]
; CHECK-SPIRV-DAG: TypeArray [[TYPE_3_DIM_ARRAY:[0-9]+]] [[TYPE_2_DIM_ARRAY]] [[SIZE]]
; CHECK-SPIRV-DAG: TypePointer [[TYPE_3_DIM_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_3_DIM_ARRAY]]

; CHECK-SPIRV: Function
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_function() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
  %1 = alloca %"class._ZTSZ4mainE3$_0.anon", align 1
  %2 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %2) #4
  %3 = addrspacecast %"class._ZTSZ4mainE3$_0.anon"* %1 to %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %3)
  %4 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %4) #4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inlinehint nounwind
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %0) #2 align 2 {
  %2 = alloca %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, align 8
  store %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %0, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %2, align 8, !tbaa !5
  call spir_func void @_Z34ivdep_embedded_multiple_dimensionsv()
  call spir_func void @_Z27ivdep_mul_arrays_and_globalv()
  call spir_func void @_Z27ivdep_embedded_inner_accessv()
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; CHECK-SPIRV: Function
; Function Attrs: nounwind
define dso_local spir_func void @_Z34ivdep_embedded_multiple_dimensionsv() #3 {
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_A:[0-9]+]]
  ; CHECK-LLVM: %[[EMB_ARRAY_A:[0-9]+]] = alloca [10 x i32]
  %1 = alloca [10 x i32], align 4
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_B:[0-9]+]]
  ; CHECK-LLVM: %[[EMB_ARRAY_B:[0-9]+]] = alloca [10 x i32]
  %2 = alloca [10 x i32], align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = bitcast [10 x i32]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %7) #4
  %8 = bitcast [10 x i32]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %8) #4
  %9 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #4
  store i32 0, i32* %3, align 4, !tbaa !9
  br label %10

10:                                               ; preds = %70, %0
  %11 = load i32, i32* %3, align 4, !tbaa !9
  %12 = icmp ne i32 %11, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_A]] 0 [[ARRAY_B]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %12, label %15, label %13

13:                                               ; preds = %10
  store i32 2, i32* %4, align 4
  %14 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %14) #4
  br label %73

15:                                               ; preds = %10
  %16 = load i32, i32* %3, align 4, !tbaa !9
  %17 = load i32, i32* %3, align 4, !tbaa !9
  %18 = sext i32 %17 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_A]], {{.*}}, !llvm.index.group ![[EMB_A_IDX_GR_DIM_1:[0-9]+]]
  %19 = getelementptr inbounds [10 x i32], [10 x i32]* %1, i64 0, i64 %18, !llvm.index.group !11
  store i32 %16, i32* %19, align 4, !tbaa !9
  %20 = load i32, i32* %3, align 4, !tbaa !9
  %21 = load i32, i32* %3, align 4, !tbaa !9
  %22 = sext i32 %21 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_B]], {{.*}}, !llvm.index.group ![[EMB_B_IDX_GR_DIM_1:[0-9]+]]
  %23 = getelementptr inbounds [10 x i32], [10 x i32]* %2, i64 0, i64 %22, !llvm.index.group !12
  store i32 %20, i32* %23, align 4, !tbaa !9
  %24 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %24) #4
  store i32 0, i32* %5, align 4, !tbaa !9
  br label %25

25:                                               ; preds = %66, %15
  %26 = load i32, i32* %5, align 4, !tbaa !9
  %27 = icmp ne i32 %26, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_A]] 0 [[ARRAY_B]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %27, label %30, label %28

28:                                               ; preds = %25
  store i32 5, i32* %4, align 4
  %29 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %29) #4
  br label %69

30:                                               ; preds = %25
  %31 = load i32, i32* %5, align 4, !tbaa !9
  %32 = load i32, i32* %5, align 4, !tbaa !9
  %33 = sext i32 %32 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_A]], {{.*}}, !llvm.index.group ![[EMB_A_IDX_GR_DIM_2:[0-9]+]]
  %34 = getelementptr inbounds [10 x i32], [10 x i32]* %1, i64 0, i64 %33, !llvm.index.group !13
  %35 = load i32, i32* %34, align 4, !tbaa !9
  %36 = add nsw i32 %35, %31
  store i32 %36, i32* %34, align 4, !tbaa !9
  %37 = load i32, i32* %5, align 4, !tbaa !9
  %38 = load i32, i32* %5, align 4, !tbaa !9
  %39 = sext i32 %38 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_B]], {{.*}}, !llvm.index.group ![[EMB_B_IDX_GR_DIM_2:[0-9]+]]
  %40 = getelementptr inbounds [10 x i32], [10 x i32]* %2, i64 0, i64 %39, !llvm.index.group !15
  %41 = load i32, i32* %40, align 4, !tbaa !9
  %42 = add nsw i32 %41, %37
  store i32 %42, i32* %40, align 4, !tbaa !9
  %43 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %43) #4
  store i32 0, i32* %6, align 4, !tbaa !9
  br label %44

44:                                               ; preds = %62, %30
  %45 = load i32, i32* %6, align 4, !tbaa !9
  %46 = icmp ne i32 %45, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_A]] 0 [[ARRAY_B]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %46, label %49, label %47

47:                                               ; preds = %44
  store i32 8, i32* %4, align 4
  %48 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %48) #4
  br label %65

49:                                               ; preds = %44
  %50 = load i32, i32* %6, align 4, !tbaa !9
  %51 = load i32, i32* %6, align 4, !tbaa !9
  %52 = sext i32 %51 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_A]], {{.*}}, !llvm.index.group ![[EMB_A_IDX_GR_DIM_3:[0-9]+]]
  %53 = getelementptr inbounds [10 x i32], [10 x i32]* %1, i64 0, i64 %52, !llvm.index.group !17
  %54 = load i32, i32* %53, align 4, !tbaa !9
  %55 = add nsw i32 %54, %50
  store i32 %55, i32* %53, align 4, !tbaa !9
  %56 = load i32, i32* %6, align 4, !tbaa !9
  %57 = load i32, i32* %6, align 4, !tbaa !9
  %58 = sext i32 %57 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[EMB_ARRAY_B]], {{.*}}, !llvm.index.group ![[EMB_B_IDX_GR_DIM_3:[0-9]+]]
  %59 = getelementptr inbounds [10 x i32], [10 x i32]* %2, i64 0, i64 %58, !llvm.index.group !19
  %60 = load i32, i32* %59, align 4, !tbaa !9
  %61 = add nsw i32 %60, %56
  store i32 %61, i32* %59, align 4, !tbaa !9
  br label %62

62:                                               ; preds = %49
  %63 = load i32, i32* %6, align 4, !tbaa !9
  %64 = add nsw i32 %63, 1
  store i32 %64, i32* %6, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_MD_LOOP_DIM_3:[0-9]+]]
  br label %44, !llvm.loop !21

65:                                               ; preds = %47
  br label %66

66:                                               ; preds = %65
  %67 = load i32, i32* %5, align 4, !tbaa !9
  %68 = add nsw i32 %67, 1
  store i32 %68, i32* %5, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_MD_LOOP_DIM_2:[0-9]+]]
  br label %25, !llvm.loop !24

69:                                               ; preds = %28
  br label %70

70:                                               ; preds = %69
  %71 = load i32, i32* %3, align 4, !tbaa !9
  %72 = add nsw i32 %71, 1
  store i32 %72, i32* %3, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_MD_LOOP_DIM_1:[0-9]+]]
  br label %10, !llvm.loop !26

73:                                               ; preds = %13
  %74 = bitcast [10 x i32]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %74) #4
  %75 = bitcast [10 x i32]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %75) #4
  ret void
}

; CHECK-SPIRV: Function
; Function Attrs: nounwind
define dso_local spir_func void @_Z27ivdep_mul_arrays_and_globalv() #3 {
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_A:[0-9]+]]
  ; CHECK-LLVM: %[[SIMPLE_ARRAY_A:[0-9]+]] = alloca [10 x i32]
  %1 = alloca [10 x i32], align 4
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_B:[0-9]+]]
  ; CHECK-LLVM: %[[SIMPLE_ARRAY_B:[0-9]+]] = alloca [10 x i32]
  %2 = alloca [10 x i32], align 4
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_C:[0-9]+]]
  ; CHECK-LLVM: %[[SIMPLE_ARRAY_C:[0-9]+]] = alloca [10 x i32]
  %3 = alloca [10 x i32], align 4
  ; CHECK-SPIRV: Variable [[TYPE_PTR]] [[ARRAY_D:[0-9]+]]
  ; CHECK-LLVM: %[[SIMPLE_ARRAY_D:[0-9]+]] = alloca [10 x i32]
  %4 = alloca [10 x i32], align 4
  %5 = alloca i32, align 4
  %6 = bitcast [10 x i32]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %6) #4
  %7 = bitcast [10 x i32]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %7) #4
  %8 = bitcast [10 x i32]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %8) #4
  %9 = bitcast [10 x i32]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* %9) #4
  %10 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %10) #4
  store i32 0, i32* %5, align 4, !tbaa !9
  br label %11

11:                                               ; preds = %29, %0
  %12 = load i32, i32* %5, align 4, !tbaa !9
  %13 = icmp ne i32 %12, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyLengthMask = 0x40000 & 0x00000008 = 262152
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262152 5 4 [[ARRAY_A]] 5 [[ARRAY_D]] 5 [[ARRAY_B]] 6 [[ARRAY_C]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %13, label %16, label %14

14:                                               ; preds = %11
  %15 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %15) #4
  br label %32

16:                                               ; preds = %11
  %17 = load i32, i32* %5, align 4, !tbaa !9
  %18 = sext i32 %17 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[SIMPLE_ARRAY_A]], {{.*}}, !llvm.index.group ![[SIMPLE_A_IDX_GR:[0-9]+]]
  %19 = getelementptr inbounds [10 x i32], [10 x i32]* %1, i64 0, i64 %18, !llvm.index.group !28
  store i32 0, i32* %19, align 4, !tbaa !9
  %20 = load i32, i32* %5, align 4, !tbaa !9
  %21 = sext i32 %20 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[SIMPLE_ARRAY_B]], {{.*}}, !llvm.index.group ![[SIMPLE_B_IDX_GR:[0-9]+]]
  %22 = getelementptr inbounds [10 x i32], [10 x i32]* %2, i64 0, i64 %21, !llvm.index.group !29
  store i32 0, i32* %22, align 4, !tbaa !9
  %23 = load i32, i32* %5, align 4, !tbaa !9
  %24 = sext i32 %23 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[SIMPLE_ARRAY_C]], {{.*}}, !llvm.index.group ![[SIMPLE_C_IDX_GR:[0-9]+]]
  %25 = getelementptr inbounds [10 x i32], [10 x i32]* %3, i64 0, i64 %24, !llvm.index.group !30
  store i32 0, i32* %25, align 4, !tbaa !9
  %26 = load i32, i32* %5, align 4, !tbaa !9
  %27 = sext i32 %26 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32]* %[[SIMPLE_ARRAY_D]], {{.*}}, !llvm.index.group ![[SIMPLE_D_IDX_GR:[0-9]+]]
  %28 = getelementptr inbounds [10 x i32], [10 x i32]* %4, i64 0, i64 %27, !llvm.index.group !31
  store i32 0, i32* %28, align 4, !tbaa !9
  br label %29

29:                                               ; preds = %16
  %30 = load i32, i32* %5, align 4, !tbaa !9
  %31 = add nsw i32 %30, 1
  store i32 %31, i32* %5, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[SIMPLE_MD_LOOP:[0-9]+]]
  br label %11, !llvm.loop !32

32:                                               ; preds = %14
  %33 = bitcast [10 x i32]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %33) #4
  %34 = bitcast [10 x i32]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %34) #4
  %35 = bitcast [10 x i32]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %35) #4
  %36 = bitcast [10 x i32]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 40, i8* %36) #4
  ret void
}

; Function Attrs: norecurse nounwind
define dso_local spir_func void @_Z27ivdep_embedded_inner_accessv() #3 {
  ; CHECK-SPIRV: Variable [[TYPE_3_DIM_PTR]] [[ARRAY_A:[0-9]+]]
  ; CHECK-LLVM: %[[EMB_INNER_ARRAY_A:[0-9]+]] = alloca [10 x [10 x [10 x i32]]]
  %1 = alloca [10 x [10 x [10 x i32]]], align 4
  ; CHECK-SPIRV: Variable [[TYPE_3_DIM_PTR]] [[ARRAY_B:[0-9]+]]
  ; CHECK-LLVM: %[[EMB_INNER_ARRAY_B:[0-9]+]] = alloca [10 x [10 x [10 x i32]]]
  %2 = alloca [10 x [10 x [10 x i32]]], align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = bitcast [10 x [10 x [10 x i32]]]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %7) #4
  %8 = bitcast [10 x [10 x [10 x i32]]]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4000, i8* %8) #4
  %9 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #4
  store i32 0, i32* %3, align 4, !tbaa !9
  br label %10

10:                                               ; preds = %57, %0
  %11 = load i32, i32* %3, align 4, !tbaa !9
  %12 = icmp ne i32 %11, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_B]] 0 [[ARRAY_A]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %12, label %15, label %13

13:                                               ; preds = %10
  store i32 2, i32* %4, align 4
  %14 = bitcast i32* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %14) #4
  br label %60

15:                                               ; preds = %10
  %16 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %16) #4
  store i32 0, i32* %5, align 4, !tbaa !9
  br label %17

17:                                               ; preds = %53, %15
  %18 = load i32, i32* %5, align 4, !tbaa !9
  %19 = icmp ne i32 %18, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_B]] 0 [[ARRAY_A]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %19, label %22, label %20

20:                                               ; preds = %17
  store i32 5, i32* %4, align 4
  %21 = bitcast i32* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %21) #4
  br label %56

22:                                               ; preds = %17
  %23 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %23) #4
  store i32 0, i32* %6, align 4, !tbaa !9
  br label %24

24:                                               ; preds = %49, %22
  %25 = load i32, i32* %6, align 4, !tbaa !9
  %26 = icmp ne i32 %25, 10
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyInfiniteMask = 0x40000 & 0x00000004 = 262148
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262148 2 [[ARRAY_B]] 0 [[ARRAY_A]] 0
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %26, label %29, label %27

27:                                               ; preds = %24
  store i32 8, i32* %4, align 4
  %28 = bitcast i32* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %28) #4
  br label %52

29:                                               ; preds = %24
  %30 = load i32, i32* %3, align 4, !tbaa !9
  %31 = sext i32 %30 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* %[[EMB_INNER_ARRAY_B]], {{.*}}, !llvm.index.group ![[EMB_INNER_B_IDX_GR:[0-9]+]]
  %32 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* %2, i64 0, i64 %31, !llvm.index.group !37
  %33 = load i32, i32* %5, align 4, !tbaa !9
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* %32, i64 0, i64 %34
  %36 = load i32, i32* %6, align 4, !tbaa !9
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [10 x i32], [10 x i32]* %35, i64 0, i64 %37
  %39 = load i32, i32* %38, align 4, !tbaa !9
  %40 = load i32, i32* %3, align 4, !tbaa !9
  %41 = sext i32 %40 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* %[[EMB_INNER_ARRAY_A]], {{.*}}, !llvm.index.group ![[EMB_INNER_A_IDX_GR:[0-9]+]]
  %42 = getelementptr inbounds [10 x [10 x [10 x i32]]], [10 x [10 x [10 x i32]]]* %1, i64 0, i64 %41, !llvm.index.group !41
  %43 = load i32, i32* %5, align 4, !tbaa !9
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* %42, i64 0, i64 %44
  %46 = load i32, i32* %6, align 4, !tbaa !9
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds [10 x i32], [10 x i32]* %45, i64 0, i64 %47
  store i32 %39, i32* %48, align 4, !tbaa !9
  br label %49

49:                                               ; preds = %29
  %50 = load i32, i32* %6, align 4, !tbaa !9
  %51 = add nsw i32 %50, 1
  store i32 %51, i32* %6, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_INNER_MD_LOOP_DIM_3:[0-9]+]]
  br label %24, !llvm.loop !45

52:                                               ; preds = %27
  br label %53

53:                                               ; preds = %52
  %54 = load i32, i32* %5, align 4, !tbaa !9
  %55 = add nsw i32 %54, 1
  store i32 %55, i32* %5, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_INNER_MD_LOOP_DIM_2:[0-9]+]]
  br label %17, !llvm.loop !47

56:                                               ; preds = %20
  br label %57

57:                                               ; preds = %56
  %58 = load i32, i32* %3, align 4, !tbaa !9
  %59 = add nsw i32 %58, 1
  store i32 %59, i32* %3, align 4, !tbaa !9
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_INNER_MD_LOOP_DIM_1:[0-9]+]]
  br label %10, !llvm.loop !49

60:                                               ; preds = %13
  %61 = bitcast [10 x [10 x [10 x i32]]]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %61) #4
  %62 = bitcast [10 x [10 x [10 x i32]]]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4000, i8* %62) #4
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="ivdep-spirv.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 10.0.0"}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
; Legacy metadata
; CHECK-LLVM-DAG: ![[IVDEP_LEGACY:[0-9]+]] = !{!"llvm.loop.ivdep.enable"}
;
; Accesses within each dimension of a multi-dimensional (n > 2) loop
; Index group(s) of each inner loop should have a subnode that points to the containing loop's index group subnode
;
; Loop dimension 3 (the innermost loop)
; CHECK-LLVM-DAG: ![[EMB_A_IDX_GR_DIM_3]] = !{![[EMB_A_IDX_GR_DIM_1]], ![[EMB_A_IDX_NODE_DIM_2:[0-9]+]], ![[EMB_A_IDX_NODE_DIM_3:[0-9]+]]}
; CHECK-LLVM-DAG: ![[EMB_A_IDX_NODE_DIM_3]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_B_IDX_GR_DIM_3]] = !{![[EMB_B_IDX_GR_DIM_1]], ![[EMB_B_IDX_NODE_DIM_2:[0-9]+]], ![[EMB_B_IDX_NODE_DIM_3:[0-9]+]]}
; CHECK-LLVM-DAG: ![[EMB_B_IDX_NODE_DIM_3]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_MD_LOOP_DIM_3]] = distinct !{![[EMB_MD_LOOP_DIM_3]], ![[IVDEP_LEGACY]], ![[EMB_IVDEP_DIM_3:[0-9]+]]{{.*}}}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_IVDEP_DIM_3]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_A_IDX_NODE_DIM_3]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_IVDEP_DIM_3]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_B_IDX_NODE_DIM_3]]{{.*}}}
;
; Loop dimension 2
; CHECK-LLVM-DAG: ![[EMB_A_IDX_GR_DIM_2]] = !{![[EMB_A_IDX_GR_DIM_1]], ![[EMB_A_IDX_NODE_DIM_2]]}
; CHECK-LLVM-DAG: ![[EMB_A_IDX_NODE_DIM_2]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_B_IDX_GR_DIM_2]] = !{![[EMB_B_IDX_GR_DIM_1]], ![[EMB_B_IDX_NODE_DIM_2]]}
; CHECK-LLVM-DAG: ![[EMB_B_IDX_NODE_DIM_2]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_MD_LOOP_DIM_2]] = distinct !{![[EMB_MD_LOOP_DIM_2]], ![[IVDEP_LEGACY]], ![[EMB_IVDEP_DIM_2:[0-9]+]]}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_IVDEP_DIM_2]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_A_IDX_NODE_DIM_2]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_IVDEP_DIM_2]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_B_IDX_NODE_DIM_2]]{{.*}}}
;
; Loop dimension 1 (the outermost loop)
; CHECK-LLVM-DAG: ![[EMB_A_IDX_GR_DIM_1]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_MD_LOOP_DIM_1]] = distinct !{![[EMB_MD_LOOP_DIM_1]], ![[IVDEP_LEGACY]], ![[EMB_IVDEP_DIM_1:[0-9]+]]}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_IVDEP_DIM_1]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_A_IDX_GR_DIM_1]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_IVDEP_DIM_1]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_B_IDX_GR_DIM_1]]{{.*}}}
!11 = distinct !{}
!12 = distinct !{}
!13 = !{!11, !14}
!14 = distinct !{}
!15 = !{!12, !16}
!16 = distinct !{}
!17 = !{!11, !14, !18}
!18 = distinct !{}
!19 = !{!12, !16, !20}
!20 = distinct !{}
!21 = distinct !{!21, !22, !23}
!22 = !{!"llvm.loop.parallel_access_indices", !18, !20}
!23 = !{!"llvm.loop.ivdep.enable"}
!24 = distinct !{!24, !25, !23}
!25 = !{!"llvm.loop.parallel_access_indices", !14, !16}
!26 = distinct !{!26, !27, !23}
!27 = !{!"llvm.loop.parallel_access_indices", !11, !12}
; Multiple arrays with specific safelens in a simple one-dimensional loop
; CHECK-LLVM-DAG: ![[SIMPLE_A_IDX_GR]] = distinct !{}
; CHECK-LLVM-DAG: ![[SIMPLE_B_IDX_GR]] = distinct !{}
; CHECK-LLVM-DAG: ![[SIMPLE_C_IDX_GR]] = distinct !{}
; CHECK-LLVM-DAG: ![[SIMPLE_D_IDX_GR]] = distinct !{}
!28 = distinct !{}
!29 = distinct !{}
!30 = distinct !{}
!31 = distinct !{}
; Legacy metadata
; CHECK-LLVM-DAG: ![[IVDEP_LEGACY_SAFELEN_5:[0-9]+]] = !{!"llvm.loop.ivdep.safelen", i32 5}
;
; CHECK-LLVM-DAG: ![[SIMPLE_MD_LOOP]] = distinct !{![[SIMPLE_MD_LOOP]], ![[IVDEP_LEGACY_SAFELEN_5]], ![[SIMPLE_IVDEP_C:[0-9]+]], ![[SIMPLE_IVDEP_A_D:[0-9]+]], ![[SIMPLE_IVDEP_B:[0-9]+]]}
!32 = distinct !{!32, !33, !34, !35, !36}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[SIMPLE_IVDEP_A_D]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[SIMPLE_A_IDX_GR]],{{.*}} i32 5}
; CHECK-LLVM-MD-OP2-DAG: ![[SIMPLE_IVDEP_A_D]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[SIMPLE_D_IDX_GR]],{{.*}} i32 5}
!33 = !{!"llvm.loop.parallel_access_indices", !28, !31, i32 5}
!34 = !{!"llvm.loop.ivdep.safelen", i32 5}
; CHECK-LLVM-DAG: ![[SIMPLE_IVDEP_B]] = !{!"llvm.loop.parallel_access_indices", ![[SIMPLE_B_IDX_GR]], i32 6}
!35 = !{!"llvm.loop.parallel_access_indices", !29, i32 6}
; CHECK-LLVM-DAG: ![[SIMPLE_IVDEP_C]] = !{!"llvm.loop.parallel_access_indices", ![[SIMPLE_C_IDX_GR]]}
!36 = !{!"llvm.loop.parallel_access_indices", !30}
; Accesses within the inner loop of a multi-dimensional (n > 2) loop nest
;
; Loop dimension 3 (the innermost loop)
; CHECK-LLVM-DAG: ![[EMB_INNER_A_IDX_GR]] = !{![[EMB_INNER_A_IDX_NODE_DIM_1:[0-9]+]], ![[EMB_INNER_A_IDX_NODE_DIM_2:[0-9]+]], ![[EMB_INNER_A_IDX_NODE_DIM_3:[0-9]+]]}
; CHECK-LLVM-DAG: ![[EMB_INNER_A_IDX_NODE_DIM_3]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_B_IDX_GR]] = !{![[EMB_INNER_B_IDX_NODE_DIM_1:[0-9]+]], ![[EMB_INNER_B_IDX_NODE_DIM_2:[0-9]+]], ![[EMB_INNER_B_IDX_NODE_DIM_3:[0-9]+]]}
; CHECK-LLVM-DAG: ![[EMB_INNER_B_IDX_NODE_DIM_3]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_MD_LOOP_DIM_3]] = distinct !{![[EMB_INNER_MD_LOOP_DIM_3]], ![[IVDEP_LEGACY]], ![[EMB_INNER_IVDEP_DIM_3:[0-9]+]]{{.*}}}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_INNER_IVDEP_DIM_3]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_A_IDX_NODE_DIM_3]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_INNER_IVDEP_DIM_3]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_B_IDX_NODE_DIM_3]]{{.*}}}
;
; Loop dimension 2
; CHECK-LLVM-DAG: ![[EMB_INNER_A_IDX_NODE_DIM_2]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_B_IDX_NODE_DIM_2]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_MD_LOOP_DIM_2]] = distinct !{![[EMB_INNER_MD_LOOP_DIM_2]], ![[IVDEP_LEGACY]], ![[EMB_INNER_IVDEP_DIM_2:[0-9]+]]}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_INNER_IVDEP_DIM_2]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_A_IDX_NODE_DIM_2]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_INNER_IVDEP_DIM_2]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_B_IDX_NODE_DIM_2]]{{.*}}}
;
; Loop dimension 1 (the outermost loop)
; CHECK-LLVM-DAG: ![[EMB_INNER_A_IDX_NODE_DIM_1]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_B_IDX_NODE_DIM_1]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_MD_LOOP_DIM_1]] = distinct !{![[EMB_INNER_MD_LOOP_DIM_1]], ![[IVDEP_LEGACY]], ![[EMB_INNER_IVDEP_DIM_1:[0-9]+]]}
; The next directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[EMB_INNER_IVDEP_DIM_1]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_A_IDX_NODE_DIM_1]]{{.*}}}
; CHECK-LLVM-MD-OP2-DAG: ![[EMB_INNER_IVDEP_DIM_1]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[EMB_INNER_B_IDX_NODE_DIM_1]]{{.*}}}
!37 = !{!38, !39, !40}
!38 = distinct !{}
!39 = distinct !{}
!40 = distinct !{}
!41 = !{!42, !43, !44}
!42 = distinct !{}
!43 = distinct !{}
!44 = distinct !{}
!45 = distinct !{!45, !46, !23}
!46 = !{!"llvm.loop.parallel_access_indices", !40, !44}
!47 = distinct !{!47, !48, !23}
!48 = !{!"llvm.loop.parallel_access_indices", !39, !43}
!49 = distinct !{!49, !50, !23}
!50 = !{!"llvm.loop.parallel_access_indices", !38, !42}
