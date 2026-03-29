; RUN: opt -passes=sycl-remangle-libspirv --remangle-spirv-target --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test TmpSuffix collision handling for bidirectional transformations.
;
; In normal operation, the pass creates long long variants (Pxiix, Pyiiy) without
; collision. However, this test creates a bidirectional collision scenario by
; providing BOTH Pliil and Pxiix as inputs (even though Pxiix is not in real
; libspirv source). Test the TmpSuffix logic that handles collisions when:
;   Map 1: Pxiix -> Pliil (collision: Pliil already exists)
;   Map 2: Pliil -> Pxiix (collision: Pxiix already exists)

define void @_Z19__spirv_AtomicStorePliil(ptr, i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePxiix(
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePliil(

define void @_Z19__spirv_AtomicStorePmiim(ptr, i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePyiiy(
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePmiim(

define void @_Z19__spirv_AtomicStorePU3AS1liil(ptr addrspace(1), i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS1xiix(ptr addrspace(1)
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS1liil(ptr addrspace(1)

define void @_Z19__spirv_AtomicStorePU3AS3liil(ptr addrspace(3), i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS3xiix(ptr addrspace(3)
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS3liil(ptr addrspace(3)

define void @_Z19__spirv_AtomicStorePU3AS1miim(ptr addrspace(1), i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS1yiiy(ptr addrspace(1)
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS1miim(ptr addrspace(1)

define void @_Z19__spirv_AtomicStorePU3AS3miim(ptr addrspace(3), i32, i32, i64) { unreachable }
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS3yiiy(ptr addrspace(3)
; CHECK-DAG: define void @_Z19__spirv_AtomicStorePU3AS3miim(ptr addrspace(3)

define <4 x half> @_Z30__spirv_ImageSampleExplicitLodImDv4_DhDv3_fET0_T_T1_iS4_S4_(i64, <3 x float>, i32, <3 x float>, <3 x float>) { unreachable }
; CHECK-DAG: define <4 x half> @_Z30__spirv_ImageSampleExplicitLodIyDv4_DF16_Dv3_fET0_T_T1_iS4_S4_(

define <4 x half> @_Z30__spirv_ImageSampleExplicitLodImDv4_DF16_Dv3_fET0_T_T1_iS4_S4_(i64, <3 x float>, i32, <3 x float>, <3 x float>) { unreachable }
; CHECK-DAG: define <4 x half> @_Z30__spirv_ImageSampleExplicitLodImDv4_DF16_Dv3_fET0_T_T1_iS4_S4_(
