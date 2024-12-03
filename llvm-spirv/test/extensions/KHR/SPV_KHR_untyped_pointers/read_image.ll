; Check that untyped pointers extension does not affect the translation of images.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv --spirv-ext=+SPV_KHR_untyped_pointers %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[#IntTy:]] 32
; CHECK-SPIRV: TypeImage [[#ImageTy:]] [[#]] 2 0 0 0 0 0 0
; CHECK-SPIRV: TypeVector [[#IVecTy:]] [[#IntTy]]
; CHECK-SPIRV: TypeFloat [[#FloatTy:]]
; CHECK-SPIRV: TypeVector [[#FVecTy:]] [[#FloatTy]]

; CHECK-SPIRV: Load [[#ImageTy]] [[#Image0:]]
; CHECK-SPIRV: Load [[#IVecTy]] [[#Coord0:]]
; CHECK-SPIRV: ImageRead [[#IVecTy]] [[#]] [[#Image0]] [[#Coord0]] 8192

; CHECK-SPIRV: Load [[#ImageTy]] [[#Image1:]]
; CHECK-SPIRV: Load [[#IVecTy]] [[#Coord1:]]
; CHECK-SPIRV: ImageRead [[#FVecTy]] [[#]] [[#Image1]] [[#Coord1]]

; CHECK-LLVM: call spir_func <4 x i32> @_Z24__spirv_ImageRead_Ruint4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_ii(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0)
; CHECK-LLVM: call spir_func <4 x float> @_Z25__spirv_ImageRead_Rfloat4PU3AS133__spirv_Image__void_2_0_0_0_0_0_0Dv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @kernelA(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %c = alloca <4 x i32>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1) #2
  store <4 x i32> %call, ptr %c, align 16
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(read)
declare spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), <4 x i32> noundef) #1

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @kernelB(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %f = alloca <4 x float>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1) #2
  store <4 x float> %call, ptr %f, align 16
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(read)
declare spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), <4 x i32> noundef) #1

attributes #0 = { convergent noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent nounwind willreturn memory(read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind willreturn memory(read) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 20.0.0git (https:;github.com/llvm/llvm-project.git 5313d2e6d02d2a8b192e2c007241ff261287e1ca)"}
!3 = !{i32 1}
!4 = !{!"read_only"}
!5 = !{!"image3d_t"}
!6 = !{!""}
