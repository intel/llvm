; RUN: opt < %s -passes=ESIMDLowerVecArg -S | FileCheck %s

; Check that we correctly update metadata to reference the new function

%"class.sycl::_V1::vec" = type { <2 x double> }

$foo = comdat any

define weak_odr dso_local spir_kernel void @foo(%"class.sycl::_V1::vec" addrspace(1)* noundef align 16 %_arg_out) local_unnamed_addr comdat {
entry:
  ret void
}

;CHECK: !genx.kernels = !{![[GenXMD:[0-9]+]]}
!genx.kernels = !{!0}

;CHECK: ![[GenXMD]] = !{void (<2 x double> addrspace(1)*)* @foo, {{.*}}}
!0 = !{void (%"class.sycl::_V1::vec" addrspace(1)*)* @foo, !"foo", !1, i32 0, i32 0, !1, !2, i32 0, i32 0}
!1 = !{i32 0}
!2 = !{!"svmptr_t"}
