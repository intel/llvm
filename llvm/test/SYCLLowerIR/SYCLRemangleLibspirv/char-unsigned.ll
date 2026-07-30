; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=unsigned -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s

; Test checks remangling when char is unsigned.

; scalar char parameter.
define signext range(i8 0, 9) i8 @_Z15__spirv_ocl_clzc(i8 signext) { unreachable }
define zeroext range(i8 0, 9) i8 @_Z15__spirv_ocl_clzh(i8 zeroext) { unreachable }
; CHECK-DAG: define signext range(i8 0, 9) i8 @_Z15__spirv_ocl_clza(i8 signext
; CHECK-DAG: define zeroext range(i8 0, 9) i8 @_Z15__spirv_ocl_clzh(i8 zeroext
; CHECK-DAG: define zeroext range(i8 0, 9) i8 @_Z15__spirv_ocl_clzc(i8 zeroext

; vector of char.
define <4 x i8> @_Z20__spirv_ocl_popcountDv4_c(<4 x i8>) { unreachable }
define <4 x i8> @_Z20__spirv_ocl_popcountDv4_h(<4 x i8>) { unreachable }
; CHECK-DAG: define <4 x i8> @_Z20__spirv_ocl_popcountDv4_a(
; CHECK-DAG: define <4 x i8> @_Z20__spirv_ocl_popcountDv4_h(
; CHECK-DAG: define <4 x i8> @_Z20__spirv_ocl_popcountDv4_c(

; char parameter with substitutions.
define signext i8 @_Z18__spirv_ocl_rotatecc(i8 signext, i8 signext) { unreachable }
define zeroext i8 @_Z18__spirv_ocl_rotatehh(i8 zeroext, i8 zeroext) { unreachable }
; CHECK-DAG: define signext i8 @_Z18__spirv_ocl_rotateaa(i8 signext {{.*}}, i8 signext {{.*}})
; CHECK-DAG: define zeroext i8 @_Z18__spirv_ocl_rotatehh(i8 zeroext {{.*}}, i8 zeroext {{.*}})
; CHECK-DAG: define zeroext i8 @_Z18__spirv_ocl_rotatecc(i8 zeroext {{.*}}, i8 zeroext {{.*}})

; multiple char parameters.
define signext i8 @_Z21__spirv_ocl_bitselectccc(i8 signext, i8 signext, i8 signext) { unreachable }
define zeroext i8 @_Z21__spirv_ocl_bitselecthhh(i8 zeroext, i8 zeroext, i8 zeroext) { unreachable }
; CHECK-DAG: define signext i8 @_Z21__spirv_ocl_bitselectaaa(i8 signext {{.*}}, i8 signext {{.*}}, i8 signext {{.*}})
; CHECK-DAG: define zeroext i8 @_Z21__spirv_ocl_bitselecthhh(i8 zeroext {{.*}}, i8 zeroext {{.*}}, i8 zeroext {{.*}})
; CHECK-DAG: define zeroext i8 @_Z21__spirv_ocl_bitselectccc(i8 zeroext {{.*}}, i8 zeroext {{.*}}, i8 zeroext {{.*}})

; vector with substitution preserved.
define <16 x i8> @_Z21__spirv_ocl_bitselectDv16_cS_S_(<16 x i8>, <16 x i8>, <16 x i8>) { unreachable }
define <16 x i8> @_Z21__spirv_ocl_bitselectDv16_hS_S_(<16 x i8>, <16 x i8>, <16 x i8>) { unreachable }
; CHECK-DAG: define <16 x i8> @_Z21__spirv_ocl_bitselectDv16_aS_S_(
; CHECK-DAG: define <16 x i8> @_Z21__spirv_ocl_bitselectDv16_hS_S_(
; CHECK-DAG: define <16 x i8> @_Z21__spirv_ocl_bitselectDv16_cS_S_(
