;; Ensure @llvm.fptosi.sat.* and @llvm.fptoui.sat.* intrinsics are translated

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability Kernel
; CHECK-SPIRV: Decorate [[SAT1:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT2:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT3:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT4:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT5:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT6:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT7:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT8:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT9:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT10:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT11:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT12:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT13:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT14:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT15:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: Decorate [[SAT16:[0-9]+]] SaturatedConversion

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; signed output
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; float input
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;; output i8
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT1]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_char_satf(float %input)
define spir_kernel void @testfunction_float_to_signed_i8(float %input) {
entry:
   %0 = call i8 @llvm.fptosi.sat.i8.f32(float %input)
   ret void
}
declare i8 @llvm.fptosi.sat.i8.f32(float)

;;;;;;; output i16
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT2]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_short_satf(float %input)
define spir_kernel void @testfunction_float_to_signed_i16(float %input) {
entry:
   %0 = call i16 @llvm.fptosi.sat.i16.f32(float %input)
   ret void
}
declare i16 @llvm.fptosi.sat.i16.f32(float)

;;;;;;; output i32
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT3]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_int_satf(float %input)
define spir_kernel void @testfunction_float_to_signed_i32(float %input) {
entry:
   %0 = call i32 @llvm.fptosi.sat.i32.f32(float %input)
   ret void
}
declare i32 @llvm.fptosi.sat.i32.f32(float)

;;;;;;; output i64
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT4]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_long_satf(float %input)
define spir_kernel void @testfunction_float_to_signed_i64(float %input) {
entry:
   %0 = call i64 @llvm.fptosi.sat.i64.f32(float %input)
   ret void
}
declare i64 @llvm.fptosi.sat.i64.f32(float)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; double input
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;; output i8
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT5]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_char_satd(double %input)
define spir_kernel void @testfunction_double_to_signed_i8(double %input) {
entry:
   %0 = call i8 @llvm.fptosi.sat.i8.f64(double %input)
   ret void
}
declare i8 @llvm.fptosi.sat.i8.f64(double)

;;;;;;; output i16
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT6]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_short_satd(double %input)
define spir_kernel void @testfunction_double_to_signed_i16(double %input) {
entry:
   %0 = call i16 @llvm.fptosi.sat.i16.f64(double %input)
   ret void
}
declare i16 @llvm.fptosi.sat.i16.f64(double)

;;;;;;; output i32
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT7]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_int_satd(double %input)
define spir_kernel void @testfunction_double_to_signed_i32(double %input) {
entry:
   %0 = call i32 @llvm.fptosi.sat.i32.f64(double %input)
   ret void
}
declare i32 @llvm.fptosi.sat.i32.f64(double)

;;;;;;; output i64
; CHECK-SPIRV: ConvertFToS {{[0-9]+}} [[SAT8]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_long_satd(double %input)
define spir_kernel void @testfunction_double_to_signed_i64(double %input) {
entry:
   %0 = call i64 @llvm.fptosi.sat.i64.f64(double %input)
   ret void
}
declare i64 @llvm.fptosi.sat.i64.f64(double)




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; unsigned output
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; float input
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;; output i8
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT9]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_uchar_satf(float %input)
define spir_kernel void @testfunction_float_to_unsigned_i8(float %input) {
entry:
   %0 = call i8 @llvm.fptoui.sat.i8.f32(float %input)
   ret void
}
declare i8 @llvm.fptoui.sat.i8.f32(float)

;;;;;;; output i16
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT10]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_ushort_satf(float %input)
define spir_kernel void @testfunction_float_to_unsigned_i16(float %input) {
entry:
   %0 = call i16 @llvm.fptoui.sat.i16.f32(float %input)
   ret void
}
declare i16 @llvm.fptoui.sat.i16.f32(float)

;;;;;;; output i32
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT11]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_uint_satf(float %input)
define spir_kernel void @testfunction_float_to_unsigned_i32(float %input) {
entry:
   %0 = call i32 @llvm.fptoui.sat.i32.f32(float %input)
   ret void
}
declare i32 @llvm.fptoui.sat.i32.f32(float)

;;;;;;; output i64
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT12]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_ulong_satf(float %input)
define spir_kernel void @testfunction_float_to_unsigned_i64(float %input) {
entry:
   %0 = call i64 @llvm.fptoui.sat.i64.f32(float %input)
   ret void
}
declare i64 @llvm.fptoui.sat.i64.f32(float)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; double input
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;; output i8
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT13]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_uchar_satd(double %input)
define spir_kernel void @testfunction_double_to_unsigned_i8(double %input) {
entry:
   %0 = call i8 @llvm.fptoui.sat.i8.f64(double %input)
   ret void
}
declare i8 @llvm.fptoui.sat.i8.f64(double)

;;;;;;; output i16
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT14]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_ushort_satd(double %input)
define spir_kernel void @testfunction_double_to_unsigned_i16(double %input) {
entry:
   %0 = call i16 @llvm.fptoui.sat.i16.f64(double %input)
   ret void
}
declare i16 @llvm.fptoui.sat.i16.f64(double)

;;;;;;; output i32
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT15]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_uint_satd(double %input)
define spir_kernel void @testfunction_double_to_unsigned_i32(double %input) {
entry:
   %0 = call i32 @llvm.fptoui.sat.i32.f64(double %input)
   ret void
}
declare i32 @llvm.fptoui.sat.i32.f64(double)

;;;;;;; output i64
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT16]]
; CHECK-LLVM: define spir_kernel
; CHECK-LLVM: convert_ulong_satd(double %input)
define spir_kernel void @testfunction_double_to_unsigned_i64(double %input) {
entry:
   %0 = call i64 @llvm.fptoui.sat.i64.f64(double %input)
   ret void
}
declare i64 @llvm.fptoui.sat.i64.f64(double)

