; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o %t.reg.bc
; RUN: llvm-dis %t.reg.bc -o - | FileCheck --check-prefix=CHECK-LLVM %s

target triple = "spir64-unknown-unknown"

@G.0 = addrspace(1) global i1 false
@G.1 = addrspace(1) global i1 true
@G.2 = addrspace(1) global <8 x i1> <i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>

define spir_func void @test_lshr_i1(i1 %a, i1 %b) {
entry:
  %0 = lshr i1 %a, %b
; CHECK-LLVM: [[AI32_0:%[0-9]+]] = select i1 %a, i32 1, i32 0
; CHECK-LLVM: [[BI32_0:%[0-9]+]] = select i1 %b, i32 1, i32 0
; CHECK-LLVM: [[LSHRI32_0:%[0-9]+]] = lshr i32 [[AI32_0]], [[BI32_0]]
; CHECK-LLVM: [[TRUNC_0:%[0-9]+]] = icmp ne i32 [[LSHRI32_0]], 0
  %1 = zext i1 %0 to i32
  %2 = zext i1 %0 to i32
; CHECK-LLVM-NOT zext
; CHECK-LLVM-NOT select
  store i1 %0, ptr addrspace(1) @G.0, align 1
; CHECK-LLVM: store i1 [[TRUNC_0]], ptr addrspace(1) @G.0, align 1
  ret void
}

define spir_func void @test_shl_i1(i1 %a, i1 %b) {
entry:
  %0 = shl i1 %a, %b
; CHECK-LLVM: [[AI32_1:%[0-9]+]] = select i1 %a, i32 1, i32 0
; CHECK-LLVM: [[BI32_1:%[0-9]+]] = select i1 %b, i32 1, i32 0
; CHECK-LLVM: [[LSHR32_1:%[0-9]+]] = lshr i32 [[AI32_1]], [[BI32_1]]
; CHECK-LLVM: [[TRUNC_1:%[0-9]+]] = icmp ne i32 [[LSHR32_1]], 0
  %1 = zext i1 %0 to i32
  %2 = zext i1 %0 to i32
; CHECK-LLVM-NOT: zext
; CHECK-LLVM-NOT: select
  store i1 %0, ptr addrspace(1) @G.1, align 1
; CHECK-LLVM: store i1 [[TRUNC_1]], ptr addrspace(1) @G.1, align 1
  ret void
}

define spir_func void @test_shl_vec_i1(<8 x i1> %a, <8 x i1> %b) {
entry:
  %0 = shl <8 x i1> %a, %b
; CHECK-LLVM: [[AI32_2:%[0-9]+]] = select <8 x i1> %a, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32> zeroinitializer
; CHECK-LLVM: [[BI32_2:%[0-9]+]] = select <8 x i1> %b, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <8 x i32> zeroinitializer
; CHECK-LLVM: [[LSHR32_2:%[0-9]+]] = lshr <8 x i32> [[AI32_2]], [[BI32_2]]
; CHECK-LLVM: [[TRUNC_2:%[0-9]+]] = icmp ne <8 x i32> [[LSHR32_2]], zeroinitializer
  %1 = zext <8 x i1> %0 to <8 x i32>
  %2 = zext <8 x i1> %0 to <8 x i32>
; CHECK-LLVM-NOT: zext
; CHECK-LLVM-NOT: select
  store <8 x i1> %0, ptr addrspace(1) @G.2, align 1
; CHECK-LLVM: store <8 x i1> [[TRUNC_2]], ptr addrspace(1) @G.2, align 1
  ret void
}