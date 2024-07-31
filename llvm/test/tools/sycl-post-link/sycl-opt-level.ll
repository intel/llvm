; This test checks parsing of the attribute 'sycl-optlevel'
; by the sycl-post-link-tool:
; In addition to splitting requested by user, the kernels are also split based
; on their optimization levels.
; sycl-post-link adds 'optLevel' property to the device binary

; RUN: sycl-post-link -properties -split=source -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefixes CHECK-OPT-LEVEL-PROP-0
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefixes CHECK-OPT-LEVEL-PROP-1
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM-1

; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK-EMPTY:

; CHECK-OPT-LEVEL-PROP-0: optLevel=1|2
; CHECK-OPT-LEVEL-PROP-1: optLevel=1|0
; CHECK-SYM-0: _Z3fooii
; CHECK-SYM-1: _Z3booii

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define dso_local spir_func noundef i32 @_Z3fooii(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

define dso_local spir_func noundef i32 @_Z3booii(i32 noundef %a, i32 noundef %b) #1 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %a.addr.ascast = addrspacecast i32* %a.addr to i32 addrspace(4)*
  %b.addr.ascast = addrspacecast i32* %b.addr to i32 addrspace(4)*
  store i32 %a, i32 addrspace(4)* %a.addr.ascast, align 4
  store i32 %b, i32 addrspace(4)* %b.addr.ascast, align 4
  %0 = load i32, i32 addrspace(4)* %a.addr.ascast, align 4
  %1 = load i32, i32 addrspace(4)* %b.addr.ascast, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "sycl-module-id"="test3.cpp" "sycl-optlevel"="2" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "sycl-module-id"="test2.cpp" "sycl-optlevel"="0" }

