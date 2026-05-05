; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=64 --remangle-char-signedness=signed -mtriple=nvptx64-nvidia-cuda -S < %s | FileCheck %s --check-prefix=CHECK-NVPTX64
; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=32 --remangle-char-signedness=signed -mtriple=x86_64-pc-windows-msvc -S < %s | FileCheck %s --check-prefix=CHECK-WIN64
; RUN: opt -passes=sycl-remangle-libspirv --remangle-long-width=32 --remangle-char-signedness=signed -mtriple=i386-pc-linux-gnu -S < %s | FileCheck %s --check-prefix=CHECK-I386

; Test long/long long type (signed and unsigned) remangling.

; long parameter (i64 on 64-bit Linux, i32 on Windows/32-bit).
define i64 @_Z17__spirv_ocl_s_absl(i64) { unreachable }
define i32 @_Z17__spirv_ocl_s_absi(i32) { unreachable }
; CHECK-NVPTX64-DAG: define i64 @_Z17__spirv_ocl_s_absx(
; CHECK-NVPTX64-DAG: define i32 @_Z17__spirv_ocl_s_absi(
; CHECK-NVPTX64-DAG: define i64 @_Z17__spirv_ocl_s_absl(
; CHECK-WIN64-DAG: define i64 @_Z17__spirv_ocl_s_absx(
; CHECK-WIN64-DAG: define i32 @_Z17__spirv_ocl_s_absi(
; CHECK-WIN64-DAG: define i32 @_Z17__spirv_ocl_s_absl(
; CHECK-I386-DAG: define i64 @_Z17__spirv_ocl_s_absx(
; CHECK-I386-DAG: define i32 @_Z17__spirv_ocl_s_absi(
; CHECK-I386-DAG: define i32 @_Z17__spirv_ocl_s_absl(

; unsigned long parameter.
define i64 @_Z17__spirv_ocl_u_absm(i64) { unreachable }
define i32 @_Z17__spirv_ocl_u_absj(i32) { unreachable }
; CHECK-NVPTX64-DAG: define i64 @_Z17__spirv_ocl_u_absy(
; CHECK-NVPTX64-DAG: define i32 @_Z17__spirv_ocl_u_absj(
; CHECK-NVPTX64-DAG: define i64 @_Z17__spirv_ocl_u_absm(
; CHECK-WIN64-DAG: define i64 @_Z17__spirv_ocl_u_absy(
; CHECK-WIN64-DAG: define i32 @_Z17__spirv_ocl_u_absj(
; CHECK-WIN64-DAG: define i32 @_Z17__spirv_ocl_u_absm(
; CHECK-I386-DAG: define i64 @_Z17__spirv_ocl_u_absy(
; CHECK-I386-DAG: define i32 @_Z17__spirv_ocl_u_absj(
; CHECK-I386-DAG: define i32 @_Z17__spirv_ocl_u_absm(

; vec8 long (substitutions preserved).
define <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(<8 x i64>, <8 x i64>) { unreachable }
define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(<8 x i32>, <8 x i32>) { unreachable }
; CHECK-NVPTX64-DAG: define <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_xS_(
; CHECK-NVPTX64-DAG: define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(
; CHECK-NVPTX64-DAG: define <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(
; CHECK-WIN64-DAG: define <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_xS_(
; CHECK-WIN64-DAG: define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(
; CHECK-WIN64-DAG: define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(
; CHECK-I386-DAG: define <8 x i64> @_Z20__spirv_ocl_s_mul_hiDv8_xS_(
; CHECK-I386-DAG: define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_iS_(
; CHECK-I386-DAG: define <8 x i32> @_Z20__spirv_ocl_s_mul_hiDv8_lS_(

; vec4 unsigned long.
define <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(<4 x i64>, <4 x i64>) { unreachable }
define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(<4 x i32>, <4 x i32>) { unreachable }
; CHECK-NVPTX64-DAG: define <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_yS_(
; CHECK-NVPTX64-DAG: define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(
; CHECK-NVPTX64-DAG: define <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(
; CHECK-WIN64-DAG: define <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_yS_(
; CHECK-WIN64-DAG: define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(
; CHECK-WIN64-DAG: define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(
; CHECK-I386-DAG: define <4 x i64> @_Z20__spirv_ocl_u_mul_hiDv4_yS_(
; CHECK-I386-DAG: define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_jS_(
; CHECK-I386-DAG: define <4 x i32> @_Z20__spirv_ocl_u_mul_hiDv4_mS_(

; AtomicStore with long parameter.
define void @_Z19__spirv_AtomicStorePliil(ptr, i32, i32, i64) { unreachable }
define void @_Z19__spirv_AtomicStorePliii(ptr, i32, i32, i32) { unreachable }
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePxiix(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePxiii(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePliil(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-WIN64-DAG: define void @_Z19__spirv_AtomicStorePxiix(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-WIN64-DAG: define void @_Z19__spirv_AtomicStorePxiii(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
; CHECK-I386-DAG: define void @_Z19__spirv_AtomicStorePxiix(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-I386-DAG: define void @_Z19__spirv_AtomicStorePxiii(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})

; AtomicStore with unsigned long parameter.
define void @_Z19__spirv_AtomicStorePmiim(ptr, i32, i32, i64) { unreachable }
define void @_Z19__spirv_AtomicStorePmiij(ptr, i32, i32, i32) { unreachable }
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePyiiy(ptr %0, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePyiij(ptr %0, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
; CHECK-NVPTX64-DAG: define void @_Z19__spirv_AtomicStorePmiim(ptr %0, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-WIN64-DAG: define void @_Z19__spirv_AtomicStorePyiiy(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-WIN64-DAG: define void @_Z19__spirv_AtomicStorePyiij(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
; CHECK-I386-DAG: define void @_Z19__spirv_AtomicStorePyiiy(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i64 {{.*}})
; CHECK-I386-DAG: define void @_Z19__spirv_AtomicStorePyiij(ptr {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
