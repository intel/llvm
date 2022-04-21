; generic_as.ll
; RUN: opt < %S/generic_as.ll --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 | FileCheck %s --check-prefix=CHECK-BUILTIN
; RUN: opt < %S/generic_as.ll --passes=SYCLMutatePrintfAddrspace -S | FileCheck %s --check-prefix=CHECK-BUILTIN

; generic_as_no_opt.ll
; RUN: opt < %S/generic_as_no_opt.ll --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK-WRAPPER,CHECK-BUILTIN
; RUN: opt < %S/generic_as_no_opt.ll --passes=SYCLMutatePrintfAddrspace -S | FileCheck %s --check-prefixes=CHECK-WRAPPER,CHECK-BUILTIN

; generic_as_variadic.ll
; RUN: opt < %S/generic_as_variadic.ll --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 | FileCheck %s --check-prefix=CHECK-BUILTIN
; RUN: opt < %S/generic_as_variadic.ll --passes=SYCLMutatePrintfAddrspace -S | FileCheck %s --check-prefix=CHECK-BUILTIN

; generic_as_variadic_no_opt.ll
; RUN: opt < %S/generic_as_variadic_no_opt.ll --SYCLMutatePrintfAddrspace -S --enable-new-pm=0 | FileCheck %s --check-prefixes=CHECK-WRAPPER,CHECK-BUILTIN
; RUN: opt < %S/generic_as_variadic_no_opt.ll --passes=SYCLMutatePrintfAddrspace -S | FileCheck %s --check-prefixes=CHECK-WRAPPER,CHECK-BUILTIN

; Check that the wrapper bodies have been deleted after call replacement
; CHECK-WRAPPER-NOT: spir_func i32 @{{.*}}sycl{{.*}}printf

; Make sure the generic AS declarations have been wiped out
; in favor of the single constant AS & variadic declaration:
; CHECK-BUILTIN-NOT: declare dso_local spir_func i32 @_Z18__spirv_ocl_printf{{.*}}(i8 addrspace(4)*, {{.+}})
