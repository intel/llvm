; RUN: opt < %s -passes=lower-spirv-address-space-casts -S | FileCheck %s

declare ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4), i32)
declare ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4), i32)
declare ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4), i32)

; Casting a global pointer to a global pointer. 
; CHECK: @kernel1(ptr addrspace(1) %global)
define i1 @kernel1(ptr addrspace(1) %global) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    ; The uses of c2 will be replaced with %global.
    ; CHECK-NEXT: %c3 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(1) %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) %c3, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a global pointer to a local pointer.
; CHECK: @kernel2(ptr addrspace(1) %global)
define i1 @kernel2(ptr addrspace(1) %global) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(3) %c2 to ptr addrspace(4)
    ; The use of c2 will be replaced with null, which results in an
    ; address space cast from a null pointer. That result of the cast %c3
    ; is then replaced with null.
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a global pointer to a private pointer.
; CHECK: @kernel3(ptr addrspace(1) %global)
define i1 @kernel3(ptr addrspace(1) %global) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a local pointer to a local pointer.
; CHECK: @kernel4(ptr addrspace(3) %local)
define i1 @kernel4(ptr addrspace(3) %local) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    ; CHECK-NEXT: %c3 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(3) %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) %c3, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a local pointer to a global pointer.
; CHECK: @kernel5(ptr addrspace(3) %local)
define i1 @kernel5(ptr addrspace(3) %local) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(1) %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a local pointer to a private pointer.
; CHECK: @kernel6(ptr addrspace(3) %local)
define i1 @kernel6(ptr addrspace(3) %local) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a private pointer to a private pointer.
; CHECK: @kernel7(ptr %private)
define i1 @kernel7(ptr %private) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    ; CHECK-NEXT: %c3 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) %c3, null
    %b1 = icmp eq ptr addrspace(4) %c3, null 
    ret i1 %b1
}

; Casting a private pointer to a global pointer.
; CHECK: @kernel8(ptr %private)
define i1 @kernel8(ptr %private) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(1) %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null
    ret i1 %b1
}

; Casting a private pointer to a local pointer.
; CHECK: @kernel9(ptr %private)
define i1 @kernel9(ptr %private) {
    ; CHECK-NEXT: %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %c1, i32 0)
    %c3 = addrspacecast ptr addrspace(3) %c2 to ptr addrspace(4)
    ; CHECK-NEXT: %b1 = icmp eq ptr addrspace(4) null, null
    %b1 = icmp eq ptr addrspace(4) %c3, null
    ret i1 %b1
}
