; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Tests that GEP canonicalization (to [N x i8] element type) is suppressed
; when the element type contains a target extension type.

target datalayout = "e-p:64:64"

; GEP over a nested struct: target extension type is in an inner struct.
; Exercises the recursive descent through the struct branch.
define ptr @gep_nested_struct_target_ext(ptr %p, i64 %idx) {
; CHECK-LABEL: @gep_nested_struct_target_ext(
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr { i32, { target("spirv.DeviceEvent") } }, ptr [[P:%.*]], i64 [[IDX:%.*]]
; CHECK-NEXT:    ret ptr [[GEP]]
;
  %gep = getelementptr { i32, { target("spirv.DeviceEvent") } }, ptr %p, i64 %idx
  ret ptr %gep
}

; GEP over an array of target extension types.
define ptr @gep_array_of_target_ext(ptr %p, i64 %idx) {
; CHECK-LABEL: @gep_array_of_target_ext(
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr [4 x target("spirv.DeviceEvent")], ptr [[P:%.*]], i64 [[IDX:%.*]]
; CHECK-NEXT:    ret ptr [[GEP]]
;
  %gep = getelementptr [4 x target("spirv.DeviceEvent")], ptr %p, i64 %idx
  ret ptr %gep
}

; GEP over a direct target extension type (base case).
define ptr @gep_direct_target_ext(ptr %p, i64 %idx) {
; CHECK-LABEL: @gep_direct_target_ext(
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr target("spirv.DeviceEvent"), ptr [[P:%.*]], i64 [[IDX:%.*]]
; CHECK-NEXT:    ret ptr [[GEP]]
;
  %gep = getelementptr target("spirv.DeviceEvent"), ptr %p, i64 %idx
  ret ptr %gep
}
