; This LLVM IR was generated using Intel SYCL Clang compiler (https://github.com/intel/llvm)
;
; SYCL source code of this program can be found below.
;
; template<typename N, typename Func>
; __attribute__((sycl_kernel))
; void kernel(Func f) {
;   f();
; }
;
; int main() {
;   int buf1[10], buf2[10];
;   const int c = 42;
;
;   kernel<class EmbeddedLoopTest>([=]() mutable {
;     [[intel::ivdep(buf1, 3)]]
;     for (int i = 0; i < 6; ++i) {
;       buf1[i] *= (buf2[i + 4] + c);
;       [[intel::ivdep(2)]]
;       for (int j = 0; j < 7; ++j)
;         buf2[i] *= (buf1[i] + buf2[i + 3]);
;     }
;   });
;
;   int *ptr1 = buf1;
;   int *ptr2 = buf2;
;   kernel<class VaryingSafelenOnPointersTest>([=]() mutable {
;     [[intel::ivdep(ptr1, 3)]]
;     [[intel::ivdep(ptr2, 2)]]
;     for (int i = 0; i < 6; ++i) {
;       ptr1[i] *= (ptr2[i + 4] + c);
;       ptr2[i] *= (ptr1[i] + ptr2[i + 3]);
;     }
;   });
;
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

; CHECK-SPIRV: 2 Capability FPGALoopControlsINTEL
; CHECK-SPIRV: 9 Extension "SPV_INTEL_fpga_loop_controls"
; CHECK-SPIRV-DAG: TypeInt [[TYPE_INT_64:[0-9]+]] 64 0
; CHECK-SPIRV-DAG: TypeInt [[TYPE_INT_32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_64]] [[SIZE:[0-9]+]] 10 0
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_32]] [[OFFSET_CONST_0:[0-9]+]] 0
; CHECK-SPIRV-DAG: Constant [[TYPE_INT_32]] [[OFFSET_CONST_1:[0-9]+]] 1
; CHECK-SPIRV: TypeArray [[TYPE_ARRAY:[0-9]+]] [[TYPE_INT_32]] [[SIZE]]
; CHECK-SPIRV: TypeStruct [[TYPE_EMB_CLOSURE_STRUCT:[0-9]+]] [[TYPE_ARRAY]] [[TYPE_ARRAY]]
; The next type is only used when initializing the memory fields
; CHECK-SPIRV: TypePointer [[TYPE_CLOSURE_INIT_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_EMB_CLOSURE_STRUCT]]
; This is the type used in the kernel function
; CHECK-SPIRV: TypePointer [[TYPE_EMB_CLOSURE_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_EMB_CLOSURE_STRUCT]]
; CHECK-SPIRV: TypeFunction [[TYPE_EMB_FUNC:[0-9]+]] {{[0-9]+}} [[TYPE_EMB_CLOSURE_PTR]]
; CHECK-SPIRV: TypePointer [[TYPE_EMB_CLOSURE_PARAM_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_EMB_CLOSURE_PTR]]
; CHECK-SPIRV: TypePointer [[TYPE_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_ARRAY]]
; CHECK-SPIRV: TypePointer [[TYPE_INT_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_INT_32]]
; CHECK-SPIRV: TypeStruct [[TYPE_SFLN_CLOSURE_STRUCT:[0-9]+]] [[TYPE_INT_PTR]] [[TYPE_INT_PTR]]
; The next type is only used when initializing the memory fields
; CHECK-SPIRV: TypePointer [[TYPE_CLOSURE_INIT_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_SFLN_CLOSURE_STRUCT]]
; This is the type used in the kernel function
; CHECK-SPIRV: TypePointer [[TYPE_SFLN_CLOSURE_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_SFLN_CLOSURE_STRUCT]]
; CHECK-SPIRV: TypeFunction [[TYPE_SFLN_FUNC:[0-9]+]] {{[0-9]+}} [[TYPE_SFLN_CLOSURE_PTR]]
; CHECK-SPIRV: TypePointer [[TYPE_SFLN_CLOSURE_PARAM_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_SFLN_CLOSURE_PTR]]
; CHECK-SPIRV: TypePointer [[TYPE_SFLN_INT_PTR:[0-9]+]] {{[0-9]+}} [[TYPE_INT_PTR]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct._ZTS15__wrapper_class.__wrapper_class = type { [10 x i32] }
%struct._ZTS15__wrapper_class.__wrapper_class.0 = type { [10 x i32] }
; CHECK-LLVM: %[[CLOSURE_NAME_EMB:"class.*"]] = type { [10 x i32], [10 x i32] }
; CHECK-LLVM: %[[CLOSURE_NAME_SFLN:"class.*"]] = type { i32 addrspace(4)*, i32 addrspace(4)* }
%"class._ZTSZ4mainE3$_0.anon" = type { [10 x i32], [10 x i32] }
%"class._ZTSZ4mainE3$_0.anon.0" = type { i32 addrspace(4)*, i32 addrspace(4)* }

; Function Attrs: convergent norecurse uwtable
define dso_local spir_kernel void @_ZTSZ4mainE16EmbeddedLoopTest(%struct._ZTS15__wrapper_class.__wrapper_class* byval(%struct._ZTS15__wrapper_class.__wrapper_class) align 4 %_arg_, %struct._ZTS15__wrapper_class.__wrapper_class.0* byval(%struct._ZTS15__wrapper_class.__wrapper_class.0) align 4 %_arg_1) #0 !kernel_arg_buffer_location !4 {
entry:
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon", align 4
  %1 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 80, i8* %1) #3
  %2 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon"* %0, i32 0, i32 0
  %3 = getelementptr inbounds %struct._ZTS15__wrapper_class.__wrapper_class, %struct._ZTS15__wrapper_class.__wrapper_class* %_arg_, i32 0, i32 0
  %arrayinit.begin = getelementptr inbounds [10 x i32], [10 x i32]* %2, i64 0, i64 0
  br label %arrayinit.body

arrayinit.body:                                   ; preds = %arrayinit.body, %entry
  %arrayinit.index = phi i64 [ 0, %entry ], [ %arrayinit.next, %arrayinit.body ]
  %4 = getelementptr inbounds i32, i32* %arrayinit.begin, i64 %arrayinit.index
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %3, i64 0, i64 %arrayinit.index
  %5 = load i32, i32* %arrayidx, align 4, !tbaa !5
  store i32 %5, i32* %4, align 4, !tbaa !5
  %arrayinit.next = add nuw i64 %arrayinit.index, 1
  %arrayinit.done = icmp eq i64 %arrayinit.next, 10
  br i1 %arrayinit.done, label %arrayinit.end, label %arrayinit.body

arrayinit.end:                                    ; preds = %arrayinit.body
  %6 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon"* %0, i32 0, i32 1
  %7 = getelementptr inbounds %struct._ZTS15__wrapper_class.__wrapper_class.0, %struct._ZTS15__wrapper_class.__wrapper_class.0* %_arg_1, i32 0, i32 0
  %arrayinit.begin2 = getelementptr inbounds [10 x i32], [10 x i32]* %6, i64 0, i64 0
  br label %arrayinit.body3

arrayinit.body3:                                  ; preds = %arrayinit.body3, %arrayinit.end
  %arrayinit.index4 = phi i64 [ 0, %arrayinit.end ], [ %arrayinit.next6, %arrayinit.body3 ]
  %8 = getelementptr inbounds i32, i32* %arrayinit.begin2, i64 %arrayinit.index4
  %arrayidx5 = getelementptr inbounds [10 x i32], [10 x i32]* %7, i64 0, i64 %arrayinit.index4
  %9 = load i32, i32* %arrayidx5, align 4, !tbaa !5
  store i32 %9, i32* %8, align 4, !tbaa !5
  %arrayinit.next6 = add nuw i64 %arrayinit.index4, 1
  %arrayinit.done7 = icmp eq i64 %arrayinit.next6, 10
  br i1 %arrayinit.done7, label %arrayinit.end8, label %arrayinit.body3

arrayinit.end8:                                   ; preds = %arrayinit.body3
  %10 = addrspacecast %"class._ZTSZ4mainE3$_0.anon"* %0 to %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*
  call spir_func void @"_ZZ4mainEN3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %10) #4
  %11 = bitcast %"class._ZTSZ4mainE3$_0.anon"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 80, i8* %11) #3
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; CHECK-SPIRV: Function {{.*}} [[TYPE_EMB_FUNC]]
; CHECK-LLVM: define internal spir_func void {{.*}}(%[[CLOSURE_NAME_EMB]] addrspace(4)* %this)
; Function Attrs: convergent inlinehint norecurse nounwind uwtable
define internal spir_func void @"_ZZ4mainEN3$_0clEv"(%"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this) #2 align 2 {
entry:
  ; CHECK-SPIRV: Variable [[TYPE_EMB_CLOSURE_PARAM_PTR]] [[THIS_EMB_ID:[0-9]+]]
  ; CHECK-LLVM: %this.addr = alloca %[[CLOSURE_NAME_EMB]]
  %this.addr = alloca %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, align 8
  %i = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %j = alloca i32, align 4
  ; CHECK-SPIRV: Load [[TYPE_EMB_CLOSURE_PTR]] [[THIS_EMB_LOAD:[0-9]+]] [[THIS_EMB_ID]]
  ; CHECK-LLVM: %this1 = load %[[CLOSURE_NAME_EMB]] addrspace(4)*, %[[CLOSURE_NAME_EMB]] addrspace(4)** %this.addr
  store %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8, !tbaa !9
  %this1 = load %"class._ZTSZ4mainE3$_0.anon" addrspace(4)*, %"class._ZTSZ4mainE3$_0.anon" addrspace(4)** %this.addr, align 8
  %0 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  store i32 0, i32* %i, align 4, !tbaa !5
  br label %for.cond

for.cond:                                         ; preds = %for.inc18, %entry
  %1 = load i32, i32* %i, align 4, !tbaa !5
  %cmp = icmp slt i32 %1, 6
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  store i32 2, i32* %cleanup.dest.slot, align 4
  %2 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #3
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL = 0x40000 = 262144
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262144 2 [[BUF1_EMB_OUTER_ID:[0-9]+]] 3 [[BUF1_EMB_INNER_ID:[0-9]+]] 3
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br label %for.end20

for.body:                                         ; preds = %for.cond
  ; CHECK-LLVM: %[[BUF2_EMB_OUTER_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_EMB]], %[[CLOSURE_NAME_EMB]] addrspace(4)* %this1, i32 0, i32 1
  %3 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this1, i32 0, i32 1
  %4 = load i32, i32* %i, align 4, !tbaa !5
  %add = add nsw i32 %4, 4
  %idxprom = sext i32 %add to i64
  ; CHECK-LLVM-NOT: getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[BUF2_EMB_OUTER_CLOSURE_ACCESS]]{{.*}}, !llvm.index.group
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %3, i64 0, i64 %idxprom
  %5 = load i32, i32 addrspace(4)* %arrayidx, align 4, !tbaa !5
  %add2 = add nsw i32 %5, 42
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_PTR]] [[BUF1_EMB_OUTER_ID]] [[THIS_EMB_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_0]]
  ; CHECK-LLVM: %[[BUF1_EMB_OUTER_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_EMB]], %[[CLOSURE_NAME_EMB]] addrspace(4)* %this1, i32 0, i32 0
  %6 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this1, i32 0, i32 0
  %7 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom3 = sext i32 %7 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[BUF1_EMB_OUTER_CLOSURE_ACCESS]]{{.*}}, !llvm.index.group ![[BUF1_EMB_OUTER_IDX_GR:[0-9]+]]
  %arrayidx4 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %6, i64 0, i64 %idxprom3, !llvm.index.group !11
  %8 = load i32, i32 addrspace(4)* %arrayidx4, align 4, !tbaa !5
  %mul = mul nsw i32 %8, %add2
  store i32 %mul, i32 addrspace(4)* %arrayidx4, align 4, !tbaa !5
  %9 = bitcast i32* %j to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %9) #3
  store i32 0, i32* %j, align 4, !tbaa !5
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc, %for.body
  %10 = load i32, i32* %j, align 4, !tbaa !5
  %cmp6 = icmp slt i32 %10, 7
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL & LoopControlDependencyLengthMask = 0x40000 & 0x00000008 = 262152
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262152 2 3 [[BUF1_EMB_INNER_ID]] 2 [[BUF2_EMB_INNER_PRE_ADD_ID:[0-9]+]] 2 [[BUF2_EMB_INNER_PRE_MUL_ID:[0-9]+]] 2
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %cmp6, label %for.body8, label %for.cond.cleanup7

for.cond.cleanup7:                                ; preds = %for.cond5
  store i32 5, i32* %cleanup.dest.slot, align 4
  %11 = bitcast i32* %j to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %11) #3
  br label %for.end

for.body8:                                        ; preds = %for.cond5
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_PTR]] [[BUF1_EMB_INNER_ID]] [[THIS_EMB_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_0]]
  ; CHECK-LLVM: %[[BUF1_EMB_INNER_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_EMB]], %[[CLOSURE_NAME_EMB]] addrspace(4)* %this1, i32 0, i32 0
  %12 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this1, i32 0, i32 0
  %13 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom9 = sext i32 %13 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[BUF1_EMB_INNER_CLOSURE_ACCESS]]{{.*}}, !llvm.index.group ![[BUF1_EMB_INNER_IDX_GR:[0-9]+]]
  %arrayidx10 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %12, i64 0, i64 %idxprom9, !llvm.index.group !12
  %14 = load i32, i32 addrspace(4)* %arrayidx10, align 4, !tbaa !5
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_PTR]] [[BUF2_EMB_INNER_PRE_ADD_ID]] [[THIS_EMB_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_1]]
  ; CHECK-LLVM: %[[BUF2_EMB_INNER_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_EMB]], %[[CLOSURE_NAME_EMB]] addrspace(4)* %this1, i32 0, i32 1
  %15 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this1, i32 0, i32 1
  %16 = load i32, i32* %i, align 4, !tbaa !5
  %add11 = add nsw i32 %16, 3
  %idxprom12 = sext i32 %add11 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[BUF2_EMB_INNER_CLOSURE_ACCESS]]{{.*}}, !llvm.index.group ![[BUF2_EMB_INNER_IDX_GR:[0-9]+]]
  %arrayidx13 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %15, i64 0, i64 %idxprom12, !llvm.index.group !14
  %17 = load i32, i32 addrspace(4)* %arrayidx13, align 4, !tbaa !5
  %add14 = add nsw i32 %14, %17
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_PTR]] [[BUF2_EMB_INNER_PRE_MUL_ID]] [[THIS_EMB_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_1]]
  ; CHECK-LLVM: %[[BUF2_EMB_INNER_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_EMB]], %[[CLOSURE_NAME_EMB]] addrspace(4)* %this1, i32 0, i32 1
  %18 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon", %"class._ZTSZ4mainE3$_0.anon" addrspace(4)* %this1, i32 0, i32 1
  %19 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom15 = sext i32 %19 to i64
  ; CHECK-LLVM: getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[BUF2_EMB_INNER_CLOSURE_ACCESS]]{{.*}}, !llvm.index.group ![[BUF2_EMB_INNER_IDX_GR]]
  %arrayidx16 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %18, i64 0, i64 %idxprom15, !llvm.index.group !14
  %20 = load i32, i32 addrspace(4)* %arrayidx16, align 4, !tbaa !5
  %mul17 = mul nsw i32 %20, %add14
  store i32 %mul17, i32 addrspace(4)* %arrayidx16, align 4, !tbaa !5
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %21 = load i32, i32* %j, align 4, !tbaa !5
  %inc = add nsw i32 %21, 1
  store i32 %inc, i32* %j, align 4, !tbaa !5
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_INNER_MD_LOOP:[0-9]+]]
  br label %for.cond5, !llvm.loop !15

for.end:                                          ; preds = %for.cond.cleanup7
  br label %for.inc18

for.inc18:                                        ; preds = %for.end
  %22 = load i32, i32* %i, align 4, !tbaa !5
  %inc19 = add nsw i32 %22, 1
  store i32 %inc19, i32* %i, align 4, !tbaa !5
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[EMB_OUTER_MD_LOOP:[0-9]+]]
  br label %for.cond, !llvm.loop !18

for.end20:                                        ; preds = %for.cond.cleanup
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent norecurse uwtable
define dso_local spir_kernel void @_ZTSZ4mainE18VaryingSafelenOnPointersTest(i32 addrspace(1)* %_arg_, i32 addrspace(1)* %_arg_1) #0 !kernel_arg_buffer_location !4 {
entry:
  %_arg_.addr = alloca i32 addrspace(1)*, align 8
  %_arg_.addr2 = alloca i32 addrspace(1)*, align 8
  %0 = alloca %"class._ZTSZ4mainE3$_0.anon.0", align 8
  store i32 addrspace(1)* %_arg_, i32 addrspace(1)** %_arg_.addr, align 8, !tbaa !9
  store i32 addrspace(1)* %_arg_1, i32 addrspace(1)** %_arg_.addr2, align 8, !tbaa !9
  %1 = bitcast %"class._ZTSZ4mainE3$_0.anon.0"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %1) #3
  %2 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0"* %0, i32 0, i32 0
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %_arg_.addr, align 8, !tbaa !9
  %4 = addrspacecast i32 addrspace(1)* %3 to i32 addrspace(4)*
  store i32 addrspace(4)* %4, i32 addrspace(4)** %2, align 8, !tbaa !20
  %5 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0"* %0, i32 0, i32 1
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %_arg_.addr2, align 8, !tbaa !9
  %7 = addrspacecast i32 addrspace(1)* %6 to i32 addrspace(4)*
  store i32 addrspace(4)* %7, i32 addrspace(4)** %5, align 8, !tbaa !22
  %8 = addrspacecast %"class._ZTSZ4mainE3$_0.anon.0"* %0 to %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)*
  call spir_func void @"_ZZ4mainEN3$_1clEv"(%"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %8) #4
  %9 = bitcast %"class._ZTSZ4mainE3$_0.anon.0"* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %9) #3
  ret void
}

; CHECK-SPIRV: Function {{.*}} [[TYPE_SFLN_FUNC]]
; CHECK-LLVM: define internal spir_func void {{.*}}(%[[CLOSURE_NAME_SFLN]] addrspace(4)* %this)
; Function Attrs: convergent inlinehint norecurse nounwind
define internal spir_func void @"_ZZ4mainEN3$_1clEv"(%"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this) #2 align 2 {
entry:
  ; CHECK-SPIRV: Variable [[TYPE_SFLN_CLOSURE_PARAM_PTR]] [[THIS_SFLN_ID:[0-9]+]]
  ; CHECK-LLVM: %this.addr = alloca %[[CLOSURE_NAME_SFLN]]
  %this.addr = alloca %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)*, align 8
  %i = alloca i32, align 4
  store %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this, %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)** %this.addr, align 8, !tbaa !9
  ; CHECK-SPIRV: Load [[TYPE_SFLN_CLOSURE_PTR]] [[THIS_SFLN_LOAD:[0-9]+]] [[THIS_SFLN_ID]]
  ; CHECK-LLVM: %this1 = load %[[CLOSURE_NAME_SFLN]] addrspace(4)*, %[[CLOSURE_NAME_SFLN]] addrspace(4)** %this.addr
  %this1 = load %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)*, %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)** %this.addr, align 8
  %0 = bitcast i32* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  store i32 0, i32* %i, align 4, !tbaa !5
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !tbaa !5
  %cmp = icmp slt i32 %1, 6
  ; Per SPIR-V spec extension INTEL/SPV_INTEL_fpga_loop_controls,
  ; DependencyArrayINTEL = 0x40000 = 262144
  ; CHECK-SPIRV: LoopMerge [[MERGE_BLOCK:[0-9]+]] {{[0-9]+}} 262144 5 [[BUF1_SFLN_PRE_MUL_ID:[0-9]+]] 3 [[BUF1_SFLN_PRE_STORE_ID:[0-9]+]] 3 [[BUF2_SFLN_PRE_ADD_1_ID:[0-9]+]] 2 [[BUF2_SFLN_PRE_ADD_2_ID:[0-9]+]] 2 [[BUF2_SFLN_PRE_STORE_ID:[0-9]+]] 2
  ; CHECK-SPIRV-NEXT: BranchConditional {{[0-9]+}} {{[0-9]+}} [[MERGE_BLOCK]]
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %2 = bitcast i32* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %2) #3
  br label %for.end

for.body:                                         ; preds = %for.cond
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_SFLN_INT_PTR]] [[BUF2_SFLN_PRE_ADD_1_ID]] [[THIS_SFLN_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_1]]
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_SFLN]], %[[CLOSURE_NAME_SFLN]] addrspace(4)* %this1, i32 0, i32 1
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS]]
  %3 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this1, i32 0, i32 1
  %4 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %3, align 8, !tbaa !22
  %5 = load i32, i32* %i, align 4, !tbaa !5
  %add = add nsw i32 %5, 4
  %idxprom = sext i32 %add to i64
  ; CHECK-LLVM: getelementptr inbounds i32, i32 addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD]]{{.*}}, !llvm.index.group ![[BUF2_SFLN_INDEX_GROUP:[0-9]+]]
  %ptridx = getelementptr inbounds i32, i32 addrspace(4)* %4, i64 %idxprom, !llvm.index.group !23
  %6 = load i32, i32 addrspace(4)* %ptridx, align 4, !tbaa !5
  %add2 = add nsw i32 %6, 42
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_SFLN_INT_PTR]] [[BUF1_SFLN_PRE_MUL_ID]] [[THIS_SFLN_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_0]]
  ; CHECK-LLVM: %[[BUF1_SFLN_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_SFLN]], %[[CLOSURE_NAME_SFLN]] addrspace(4)* %this1, i32 0, i32 0
  ; CHECK-LLVM: %[[BUF1_SFLN_CLOSURE_ACCESS_LOAD:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[BUF1_SFLN_CLOSURE_ACCESS]]
  %7 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this1, i32 0, i32 0
  %8 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %7, align 8, !tbaa !20
  %9 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom3 = sext i32 %9 to i64
  ; CHECK-LLVM: getelementptr inbounds i32, i32 addrspace(4)* %[[BUF1_SFLN_CLOSURE_ACCESS_LOAD]]{{.*}}, !llvm.index.group ![[BUF1_SFLN_INDEX_GROUP:[0-9]+]]
  %ptridx4 = getelementptr inbounds i32, i32 addrspace(4)* %8, i64 %idxprom3, !llvm.index.group !24
  %10 = load i32, i32 addrspace(4)* %ptridx4, align 4, !tbaa !5
  %mul = mul nsw i32 %10, %add2
  store i32 %mul, i32 addrspace(4)* %ptridx4, align 4, !tbaa !5
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_SFLN_INT_PTR]] [[BUF1_SFLN_PRE_STORE_ID]] [[THIS_SFLN_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_0]]
  ; CHECK-LLVM: %[[BUF1_SFLN_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_SFLN]], %[[CLOSURE_NAME_SFLN]] addrspace(4)* %this1, i32 0, i32 0
  ; CHECK-LLVM: %[[BUF1_SFLN_CLOSURE_ACCESS_LOAD:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[BUF1_SFLN_CLOSURE_ACCESS]]
  %11 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this1, i32 0, i32 0
  %12 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %11, align 8, !tbaa !20
  %13 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom5 = sext i32 %13 to i64
  ; CHECK-LLVM: getelementptr inbounds i32, i32 addrspace(4)* %[[BUF1_SFLN_CLOSURE_ACCESS_LOAD]]{{.*}}, !llvm.index.group ![[BUF1_SFLN_INDEX_GROUP]]
  %ptridx6 = getelementptr inbounds i32, i32 addrspace(4)* %12, i64 %idxprom5, !llvm.index.group !24
  %14 = load i32, i32 addrspace(4)* %ptridx6, align 4, !tbaa !5
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_SFLN_INT_PTR]] [[BUF2_SFLN_PRE_ADD_2_ID]] [[THIS_SFLN_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_1]]
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_SFLN]], %[[CLOSURE_NAME_SFLN]] addrspace(4)* %this1, i32 0, i32 1
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS]]
  %15 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this1, i32 0, i32 1
  %16 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %15, align 8, !tbaa !22
  %17 = load i32, i32* %i, align 4, !tbaa !5
  %add7 = add nsw i32 %17, 3
  %idxprom8 = sext i32 %add7 to i64
  ; CHECK-LLVM: getelementptr inbounds i32, i32 addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD]]{{.*}}, !llvm.index.group ![[BUF2_SFLN_INDEX_GROUP]]
  %ptridx9 = getelementptr inbounds i32, i32 addrspace(4)* %16, i64 %idxprom8, !llvm.index.group !23
  %18 = load i32, i32 addrspace(4)* %ptridx9, align 4, !tbaa !5
  %add10 = add nsw i32 %14, %18
  ; CHECK-SPIRV: InBoundsPtrAccessChain [[TYPE_SFLN_INT_PTR]] [[BUF2_SFLN_PRE_STORE_ID]] [[THIS_SFLN_LOAD]] [[OFFSET_CONST_0]] [[OFFSET_CONST_1]]
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS:[0-9]+]] = getelementptr inbounds %[[CLOSURE_NAME_SFLN]], %[[CLOSURE_NAME_SFLN]] addrspace(4)* %this1, i32 0, i32 1
  ; CHECK-LLVM: %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS]]
  %19 = getelementptr inbounds %"class._ZTSZ4mainE3$_0.anon.0", %"class._ZTSZ4mainE3$_0.anon.0" addrspace(4)* %this1, i32 0, i32 1
  %20 = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %19, align 8, !tbaa !22
  %21 = load i32, i32* %i, align 4, !tbaa !5
  %idxprom11 = sext i32 %21 to i64
  ; CHECK-LLVM: getelementptr inbounds i32, i32 addrspace(4)* %[[BUF2_SFLN_CLOSURE_ACCESS_LOAD]]{{.*}}, !llvm.index.group ![[BUF2_SFLN_INDEX_GROUP]]
  %ptridx12 = getelementptr inbounds i32, i32 addrspace(4)* %20, i64 %idxprom11, !llvm.index.group !23
  %22 = load i32, i32 addrspace(4)* %ptridx12, align 4, !tbaa !5
  %mul13 = mul nsw i32 %22, %add10
  store i32 %mul13, i32 addrspace(4)* %ptridx12, align 4, !tbaa !5
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %23 = load i32, i32* %i, align 4, !tbaa !5
  %inc = add nsw i32 %23, 1
  store i32 %inc, i32* %i, align 4, !tbaa !5
  ; CHECK-LLVM: br label %{{.*}}, !llvm.loop ![[SFLN_MD_LOOP:[0-9]+]]
  br label %for.cond, !llvm.loop !25

for.end:                                          ; preds = %for.cond.cleanup
  ret void
}

attributes #0 = { convergent norecurse uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "tune-cpu"="generic" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { convergent inlinehint norecurse nounwind uwtable "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !7, i64 0}
; Double-nested (embedded) loop example
;
; Legacy metadata
; CHECK-LLVM-DAG: ![[IVDEP_LEGACY_SFLN_2:[0-9]+]] = !{!"llvm.loop.ivdep.safelen", i32 2}
; Inner loop
; CHECK-LLVM-DAG: ![[BUF1_EMB_INNER_IDX_GR]] = !{![[BUF1_EMB_OUTER_IDX_GR]], ![[BUF1_EMB_INNER_IDX_NODE:[0-9]+]]}
; CHECK-LLVM-DAG: ![[BUF1_EMB_INNER_IDX_NODE]] = distinct !{}
; CHECK-LLVM-DAG: ![[BUF2_EMB_INNER_IDX_GR]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_INNER_MD_LOOP]] = distinct !{![[EMB_INNER_MD_LOOP]], ![[IVDEP_LEGACY_SFLN_2]], ![[IVDEP_INNER_EMB:[0-9]+]]}
; The next 2 directives should overlap
; CHECK-LLVM-MD-OP1-DAG: ![[IVDEP_INNER_EMB]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[BUF1_EMB_INNER_IDX_NODE]]{{.*}}, i32 2}
; CHECK-LLVM-MD-OP2-DAG: ![[IVDEP_INNER_EMB]] = !{!"llvm.loop.parallel_access_indices",{{.*}} ![[BUF2_EMB_INNER_IDX_GR]]{{.*}}, i32 2}
;
; Outer loop
; CHECK-LLVM-DAG: ![[BUF1_EMB_OUTER_IDX_GR]] = distinct !{}
; CHECK-LLVM-DAG: ![[EMB_OUTER_MD_LOOP]] = distinct !{![[EMB_OUTER_MD_LOOP]], ![[IVDEP_OUTER_EMB:[0-9]+]]}
; CHECK-LLVM-DAG: ![[IVDEP_OUTER_EMB]] = !{!"llvm.loop.parallel_access_indices", ![[BUF1_EMB_OUTER_IDX_GR]], i32 3}
!11 = distinct !{}
!12 = !{!11, !13}
!13 = distinct !{}
!14 = distinct !{}
!15 = distinct !{!15, !16, !17}
!16 = !{!"llvm.loop.parallel_access_indices", !13, !14, i32 2}
!17 = !{!"llvm.loop.ivdep.safelen", i32 2}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.parallel_access_indices", !11, i32 3}
!20 = !{!21, !10, i64 0}
!21 = !{!"_ZTSZ4mainE3$_1", !10, i64 0, !10, i64 8}
!22 = !{!21, !10, i64 8}
; One-dimensional loop with varying ivdep parameters example
;
; CHECK-LLVM-DAG: ![[BUF1_SFLN_INDEX_GROUP]] = distinct !{}
; CHECK-LLVM-DAG: ![[BUF2_SFLN_INDEX_GROUP]] = distinct !{}
; CHECK-LLVM-DAG: ![[SFLN_MD_LOOP]] = distinct !{![[SFLN_MD_LOOP]], ![[IVDEP_BUF2_SFLN:[0-9]+]], ![[IVDEP_BUF1_SFLN:[0-9]+]]}
; CHECK-LLVM-DAG: ![[IVDEP_BUF1_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[BUF1_SFLN_INDEX_GROUP]], i32 3}
; CHECK-LLVM-DAG: ![[IVDEP_BUF2_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[BUF2_SFLN_INDEX_GROUP]], i32 2}
!23 = distinct !{}
!24 = distinct !{}
!25 = distinct !{!25, !26, !27}
!26 = !{!"llvm.loop.parallel_access_indices", !24, i32 3}
!27 = !{!"llvm.loop.parallel_access_indices", !23, i32 2}
