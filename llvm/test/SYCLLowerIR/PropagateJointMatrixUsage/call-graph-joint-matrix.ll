; RUN: opt -passes=sycl-propagate-joint-matrix-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about used joint_matrix
; through a call graph
;
;    K1  K2         F5
;    |   / \        / \
;    |  F4 JMM1   JM5 JM6
;    | /  \
;    F1   JM4
;   / | \
; JM1 F2 F3
;     |   \
;    JM2  JM3
; 
;
; K* - kernels
; F* - functions
; JM* - joint_matrix ctors
; JMM1 - joint_matrix_mad function

; CHECK: define spir_kernel void @kernel1() !sycl_joint_matrix ![[#ID0:]] {
define spir_kernel void @kernel1() {
  call spir_func void @func1()
  ret void
}

; CHECK: define spir_kernel void @kernel2() !sycl_joint_matrix ![[#ID1:]] !sycl_joint_matrix_mad ![[#ID2:]] {
define spir_kernel void @kernel2() {
  call spir_func void @func4()
  call spir_func void @joint_matrix_mad1()
  ret void
}

; CHECK: define spir_func void @func1() #0 !sycl_joint_matrix ![[#ID0:]] {
define spir_func void @func1() #0 {
  call spir_func void @joint_matrix1()
  call spir_func void @func2()
  call spir_func void @func3()
  ret void
}

; CHECK: define spir_func void @joint_matrix1() #1 {
define spir_func void @joint_matrix1() #1 {
  ret void
}

; CHECK: define spir_func void @func2() #2 !sycl_joint_matrix ![[#ID3:]] {
define spir_func void @func2() #2 {
  call spir_func void @joint_matrix2()
  ret void
}

; CHECK: define spir_func void @joint_matrix2() #3 {
define spir_func void @joint_matrix2() #3 {
  ret void
}

; CHECK: define spir_func void @func3() #4 !sycl_joint_matrix ![[#ID4:]] {
define spir_func void @func3() #4 {
  call spir_func void @joint_matrix3()
  ret void
}

; CHECK: define spir_func void @joint_matrix3() #5 {
define spir_func void @joint_matrix3() #5 {
  ret void
}

; CHECK: define spir_func void @func4() #6 !sycl_joint_matrix ![[#ID1:]] {
define spir_func void @func4() #6 {
  call spir_func void @joint_matrix4()
  call spir_func void @func1()
  ret void
}

; CHECK: define spir_func void @joint_matrix4() #7 {
define spir_func void @joint_matrix4() #7 {
  ret void
}

define spir_func void @joint_matrix_mad1() #8 {
  ret void
}

; CHECK: define spir_func void @func5() #0 !sycl_joint_matrix ![[#ID5:]] {
define spir_func void @func5() #0 {
  call spir_func void @joint_matrix5()
  call spir_func void @joint_matrix6()
  ret void
}

; CHECK: define spir_func void @joint_matrix5() #1 {
define spir_func void @joint_matrix5() #1 {
  ret void
}

; CHECK: define spir_func void @joint_matrix6() #3 {
define spir_func void @joint_matrix6() #3 {
  ret void
}

attributes #0 = { "sycl-joint-matrix-cols"="48" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::a" "sycl-module-id"="test.cpp" }
attributes #1 = { "sycl-joint-matrix-cols"="48" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::a" }
attributes #2 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="48" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::b" "sycl-module-id"="test.cpp" }
attributes #3 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="48" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::b" }
attributes #4 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::a" "sycl-module-id"="test.cpp" }
attributes #5 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint8" "sycl-joint-matrix-use"="use::a" }
attributes #6 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint32" "sycl-joint-matrix-use"="use::accumulator" "sycl-module-id"="test.cpp" }
attributes #7 = { "sycl-joint-matrix-cols"="12" "sycl-joint-matrix-rows"="12" "sycl-joint-matrix-type"="matrix_type::sint32" "sycl-joint-matrix-use"="use::accumulator" }
attributes #8 = { "sycl-joint-matrix-mad-size-M"="12" "sycl-joint-matrix-mad-size-K"="48" "sycl-joint-matrix-mad-size-N"="12" "sycl-joint-matrix-mad-type-A"="matrix_type::sint8" "sycl-joint-matrix-mad-type-B"="matrix_type::sint8" "sycl-joint-matrix-mad-type-C"="matrix_type::sint32" "sycl-joint-matrix-mad-type-D"="matrix_type::sint32" }

; CHECK: ![[#ID0]] = !{!"matrix_type::sint8,use::a,12,12;matrix_type::sint8,use::a,12,48;matrix_type::sint8,use::b,48,12"}
; CHECK: ![[#ID1]] = !{!"matrix_type::sint32,use::accumulator,12,12;matrix_type::sint8,use::a,12,12;matrix_type::sint8,use::a,12,48;matrix_type::sint8,use::b,48,12"}
; CHECK: ![[#ID2]] = !{!"matrix_type::sint8,matrix_type::sint8,matrix_type::sint32,matrix_type::sint32,12,48,12"}
; CHECK: ![[#ID3]] = !{!"matrix_type::sint8,use::b,48,12"}
; CHECK: ![[#ID4]] = !{!"matrix_type::sint8,use::a,12,12"}
; CHECK: ![[#ID5]] = !{!"matrix_type::sint8,use::a,12,48;matrix_type::sint8,use::b,48,12"}
