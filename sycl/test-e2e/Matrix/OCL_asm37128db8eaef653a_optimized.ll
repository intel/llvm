; ------------------------------------------------
; OCL_asm37128db8eaef653a_optimized.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, float addrspace(1)* align 4 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
.preheader.preheader:
  %10 = extractelement <8 x i32> %payloadHeader, i64 0
  %11 = extractelement <8 x i32> %payloadHeader, i64 1
  %12 = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %13 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %14 = extractelement <8 x i32> %r0, i64 1
  %15 = extractelement <8 x i32> %r0, i64 6
  %16 = mul i64 %const_reg_qword2, %const_reg_qword1
  %17 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %0, i64 %16
  %18 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %17, i64 %const_reg_qword3
  %19 = mul i64 %const_reg_qword6, %const_reg_qword5
  %20 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %19
  %21 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %20, i64 %const_reg_qword7
  %22 = mul i64 %const_reg_qword10, %const_reg_qword9
  %23 = getelementptr float, float addrspace(1)* %7, i64 %22
  %24 = getelementptr float, float addrspace(1)* %23, i64 %const_reg_qword11
  %25 = mul i32 %13, %15
  %localIdY15 = zext i16 %localIdY to i32
  %26 = add i32 %25, %localIdY15
  %27 = add i32 %26, %11
  %28 = call i32 @llvm.genx.GenISA.WaveShuffleIndex.i32.i32.i32(i32 %27, i32 0, i32 0)
  %29 = zext i32 %28 to i64
  %30 = icmp sgt i32 %27, -1
  call void @llvm.assume(i1 %30)
  %31 = mul i32 %12, %14
  %localIdX21 = zext i16 %localIdX to i32
  %32 = add i32 %31, %localIdX21
  %33 = add i32 %32, %10
  %34 = zext i32 %33 to i64
  %35 = icmp sgt i32 %33, -1
  call void @llvm.assume(i1 %35)
  %36 = zext i16 %localIdY to i64
  %37 = call i64 @llvm.genx.GenISA.WaveShuffleIndex.i64.i32.i32(i64 %36, i32 0, i32 0)
  %38 = sub nsw i64 %29, %37, !spirv.Decorations !437
  %39 = zext i16 %localIdX to i64
  %40 = sub nsw i64 %34, %39, !spirv.Decorations !437
  %41 = add i64 %16, %const_reg_qword3
  %42 = sub i64 0, %41
  %43 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %18, i64 %42
  %44 = udiv i64 %40, %3
  %45 = shl i64 %44, 4
  %46 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %43, i64 %45
  %47 = add i64 %19, %const_reg_qword7
  %48 = sub i64 0, %47
  %49 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %21, i64 %48
  %50 = shl nsw i64 %38, 9, !spirv.Decorations !437
  %51 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46 to i32 addrspace(1)*
  %52 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %51)
  %53 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 32
  %54 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %53 to i32 addrspace(1)*
  %55 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %54)
  %56 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 64
  %57 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %56 to i32 addrspace(1)*
  %58 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %57)
  %59 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 96
  %60 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %59 to i32 addrspace(1)*
  %61 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %60)
  %62 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 128
  %63 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %62 to i32 addrspace(1)*
  %64 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %63)
  %65 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 160
  %66 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %65 to i32 addrspace(1)*
  %67 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %66)
  %68 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 192
  %69 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %68 to i32 addrspace(1)*
  %70 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %69)
  %71 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 224
  %72 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %71 to i32 addrspace(1)*
  %73 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %72)
  %74 = insertelement <8 x i32> undef, i32 %52, i64 0
  %75 = insertelement <8 x i32> %74, i32 %55, i64 1
  %76 = insertelement <8 x i32> %75, i32 %58, i64 2
  %77 = insertelement <8 x i32> %76, i32 %61, i64 3
  %78 = insertelement <8 x i32> %77, i32 %64, i64 4
  %79 = insertelement <8 x i32> %78, i32 %67, i64 5
  %80 = insertelement <8 x i32> %79, i32 %70, i64 6
  %81 = insertelement <8 x i32> %80, i32 %73, i64 7
  %82 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %49, i64 %50
  %83 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82 to i32 addrspace(1)*
  %84 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %83)
  %85 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 32
  %86 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %85 to i32 addrspace(1)*
  %87 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %86)
  %88 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 64
  %89 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %88 to i32 addrspace(1)*
  %90 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %89)
  %91 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 96
  %92 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %91 to i32 addrspace(1)*
  %93 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %92)
  %94 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 128
  %95 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %94 to i32 addrspace(1)*
  %96 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %95)
  %97 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 160
  %98 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %97 to i32 addrspace(1)*
  %99 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %98)
  %100 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 192
  %101 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %100 to i32 addrspace(1)*
  %102 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %101)
  %103 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 224
  %104 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %103 to i32 addrspace(1)*
  %105 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %104)
  %106 = insertelement <8 x i32> undef, i32 %84, i64 0
  %107 = insertelement <8 x i32> %106, i32 %87, i64 1
  %108 = insertelement <8 x i32> %107, i32 %90, i64 2
  %109 = insertelement <8 x i32> %108, i32 %93, i64 3
  %110 = insertelement <8 x i32> %109, i32 %96, i64 4
  %111 = insertelement <8 x i32> %110, i32 %99, i64 5
  %112 = insertelement <8 x i32> %111, i32 %102, i64 6
  %113 = insertelement <8 x i32> %112, i32 %105, i64 7
  %114 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x i32> %113, <8 x i32> %81, i32 9, i32 9, i32 8, i32 8, i1 false)
  %115 = or i64 %50, 256
  %116 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %49, i64 %115
  %117 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116 to i32 addrspace(1)*
  %118 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %117)
  %119 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 32
  %120 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %119 to i32 addrspace(1)*
  %121 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %120)
  %122 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 64
  %123 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %122 to i32 addrspace(1)*
  %124 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %123)
  %125 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 96
  %126 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %125 to i32 addrspace(1)*
  %127 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %126)
  %128 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 128
  %129 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %128 to i32 addrspace(1)*
  %130 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %129)
  %131 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 160
  %132 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %131 to i32 addrspace(1)*
  %133 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %132)
  %134 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 192
  %135 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %134 to i32 addrspace(1)*
  %136 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %135)
  %137 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %116, i64 224
  %138 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %137 to i32 addrspace(1)*
  %139 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %138)
  %140 = insertelement <8 x i32> undef, i32 %118, i64 0
  %141 = insertelement <8 x i32> %140, i32 %121, i64 1
  %142 = insertelement <8 x i32> %141, i32 %124, i64 2
  %143 = insertelement <8 x i32> %142, i32 %127, i64 3
  %144 = insertelement <8 x i32> %143, i32 %130, i64 4
  %145 = insertelement <8 x i32> %144, i32 %133, i64 5
  %146 = insertelement <8 x i32> %145, i32 %136, i64 6
  %147 = insertelement <8 x i32> %146, i32 %139, i64 7
  %148 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x i32> %147, <8 x i32> %81, i32 9, i32 9, i32 8, i32 8, i1 false)
  %149 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %46, i64 256
  %150 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149 to i32 addrspace(1)*
  %151 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %150)
  %152 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 32
  %153 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %152 to i32 addrspace(1)*
  %154 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %153)
  %155 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 64
  %156 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %155 to i32 addrspace(1)*
  %157 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %156)
  %158 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 96
  %159 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %158 to i32 addrspace(1)*
  %160 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %159)
  %161 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 128
  %162 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %161 to i32 addrspace(1)*
  %163 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %162)
  %164 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 160
  %165 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %164 to i32 addrspace(1)*
  %166 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %165)
  %167 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 192
  %168 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %167 to i32 addrspace(1)*
  %169 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %168)
  %170 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %149, i64 224
  %171 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %170 to i32 addrspace(1)*
  %172 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %171)
  %173 = insertelement <8 x i32> undef, i32 %151, i64 0
  %174 = insertelement <8 x i32> %173, i32 %154, i64 1
  %175 = insertelement <8 x i32> %174, i32 %157, i64 2
  %176 = insertelement <8 x i32> %175, i32 %160, i64 3
  %177 = insertelement <8 x i32> %176, i32 %163, i64 4
  %178 = insertelement <8 x i32> %177, i32 %166, i64 5
  %179 = insertelement <8 x i32> %178, i32 %169, i64 6
  %180 = insertelement <8 x i32> %179, i32 %172, i64 7
  %181 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %49, i64 16
  %182 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %181, i64 %50
  %183 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182 to i32 addrspace(1)*
  %184 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %183)
  %185 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 32
  %186 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %185 to i32 addrspace(1)*
  %187 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %186)
  %188 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 64
  %189 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %188 to i32 addrspace(1)*
  %190 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %189)
  %191 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 96
  %192 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %191 to i32 addrspace(1)*
  %193 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %192)
  %194 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 128
  %195 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %194 to i32 addrspace(1)*
  %196 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %195)
  %197 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 160
  %198 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %197 to i32 addrspace(1)*
  %199 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %198)
  %200 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 192
  %201 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %200 to i32 addrspace(1)*
  %202 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %201)
  %203 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %182, i64 224
  %204 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %203 to i32 addrspace(1)*
  %205 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %204)
  %206 = insertelement <8 x i32> undef, i32 %184, i64 0
  %207 = insertelement <8 x i32> %206, i32 %187, i64 1
  %208 = insertelement <8 x i32> %207, i32 %190, i64 2
  %209 = insertelement <8 x i32> %208, i32 %193, i64 3
  %210 = insertelement <8 x i32> %209, i32 %196, i64 4
  %211 = insertelement <8 x i32> %210, i32 %199, i64 5
  %212 = insertelement <8 x i32> %211, i32 %202, i64 6
  %213 = insertelement <8 x i32> %212, i32 %205, i64 7
  %214 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %114, <8 x i32> %213, <8 x i32> %180, i32 9, i32 9, i32 8, i32 8, i1 false)
  %215 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %181, i64 %115
  %216 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215 to i32 addrspace(1)*
  %217 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %216)
  %218 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 32
  %219 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %218 to i32 addrspace(1)*
  %220 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %219)
  %221 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 64
  %222 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %221 to i32 addrspace(1)*
  %223 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %222)
  %224 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 96
  %225 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %224 to i32 addrspace(1)*
  %226 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %225)
  %227 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 128
  %228 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %227 to i32 addrspace(1)*
  %229 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %228)
  %230 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 160
  %231 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %230 to i32 addrspace(1)*
  %232 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %231)
  %233 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 192
  %234 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %233 to i32 addrspace(1)*
  %235 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %234)
  %236 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %215, i64 224
  %237 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %236 to i32 addrspace(1)*
  %238 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %237)
  %239 = insertelement <8 x i32> undef, i32 %217, i64 0
  %240 = insertelement <8 x i32> %239, i32 %220, i64 1
  %241 = insertelement <8 x i32> %240, i32 %223, i64 2
  %242 = insertelement <8 x i32> %241, i32 %226, i64 3
  %243 = insertelement <8 x i32> %242, i32 %229, i64 4
  %244 = insertelement <8 x i32> %243, i32 %232, i64 5
  %245 = insertelement <8 x i32> %244, i32 %235, i64 6
  %246 = insertelement <8 x i32> %245, i32 %238, i64 7
  %247 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %148, <8 x i32> %246, <8 x i32> %180, i32 9, i32 9, i32 8, i32 8, i1 false)
  %248 = add i64 %22, %const_reg_qword11
  %249 = sub i64 0, %248
  %250 = getelementptr inbounds float, float addrspace(1)* %24, i64 %249
  %251 = shl nsw i64 %38, 8, !spirv.Decorations !437
  %252 = shl i64 %44, 3
  %253 = getelementptr float, float addrspace(1)* %250, i64 %252
  %254 = getelementptr float, float addrspace(1)* %253, i64 %251
  %255 = bitcast float addrspace(1)* %254 to i32 addrspace(1)*
  %bc = bitcast <8 x float> %214 to <8 x i32>
  %bc71 = extractelement <8 x i32> %bc, i64 0
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %255, i32 %bc71)
  %256 = getelementptr inbounds float, float addrspace(1)* %254, i64 16
  %257 = bitcast float addrspace(1)* %256 to i32 addrspace(1)*
  %bc6480 = extractelement <8 x i32> %bc, i64 1
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %257, i32 %bc6480)
  %258 = getelementptr inbounds float, float addrspace(1)* %254, i64 32
  %259 = bitcast float addrspace(1)* %258 to i32 addrspace(1)*
  %bc6589 = extractelement <8 x i32> %bc, i64 2
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %259, i32 %bc6589)
  %260 = getelementptr inbounds float, float addrspace(1)* %254, i64 48
  %261 = bitcast float addrspace(1)* %260 to i32 addrspace(1)*
  %bc6698 = extractelement <8 x i32> %bc, i64 3
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %261, i32 %bc6698)
  %262 = getelementptr inbounds float, float addrspace(1)* %254, i64 64
  %263 = bitcast float addrspace(1)* %262 to i32 addrspace(1)*
  %bc67107 = extractelement <8 x i32> %bc, i64 4
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %263, i32 %bc67107)
  %264 = getelementptr inbounds float, float addrspace(1)* %254, i64 80
  %265 = bitcast float addrspace(1)* %264 to i32 addrspace(1)*
  %bc68116 = extractelement <8 x i32> %bc, i64 5
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %265, i32 %bc68116)
  %266 = getelementptr inbounds float, float addrspace(1)* %254, i64 96
  %267 = bitcast float addrspace(1)* %266 to i32 addrspace(1)*
  %bc69125 = extractelement <8 x i32> %bc, i64 6
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %267, i32 %bc69125)
  %268 = getelementptr inbounds float, float addrspace(1)* %254, i64 112
  %269 = bitcast float addrspace(1)* %268 to i32 addrspace(1)*
  %bc70134 = extractelement <8 x i32> %bc, i64 7
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %269, i32 %bc70134)
  %270 = or i64 %251, 128
  %271 = getelementptr float, float addrspace(1)* %253, i64 %270
  %272 = bitcast float addrspace(1)* %271 to i32 addrspace(1)*
  %bc.1 = bitcast <8 x float> %247 to <8 x i32>
  %bc71.1 = extractelement <8 x i32> %bc.1, i64 0
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %272, i32 %bc71.1)
  %273 = getelementptr inbounds float, float addrspace(1)* %271, i64 16
  %274 = bitcast float addrspace(1)* %273 to i32 addrspace(1)*
  %bc6480.1 = extractelement <8 x i32> %bc.1, i64 1
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %274, i32 %bc6480.1)
  %275 = getelementptr inbounds float, float addrspace(1)* %271, i64 32
  %276 = bitcast float addrspace(1)* %275 to i32 addrspace(1)*
  %bc6589.1 = extractelement <8 x i32> %bc.1, i64 2
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %276, i32 %bc6589.1)
  %277 = getelementptr inbounds float, float addrspace(1)* %271, i64 48
  %278 = bitcast float addrspace(1)* %277 to i32 addrspace(1)*
  %bc6698.1 = extractelement <8 x i32> %bc.1, i64 3
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %278, i32 %bc6698.1)
  %279 = getelementptr inbounds float, float addrspace(1)* %271, i64 64
  %280 = bitcast float addrspace(1)* %279 to i32 addrspace(1)*
  %bc67107.1 = extractelement <8 x i32> %bc.1, i64 4
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %280, i32 %bc67107.1)
  %281 = getelementptr inbounds float, float addrspace(1)* %271, i64 80
  %282 = bitcast float addrspace(1)* %281 to i32 addrspace(1)*
  %bc68116.1 = extractelement <8 x i32> %bc.1, i64 5
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %282, i32 %bc68116.1)
  %283 = getelementptr inbounds float, float addrspace(1)* %271, i64 96
  %284 = bitcast float addrspace(1)* %283 to i32 addrspace(1)*
  %bc69125.1 = extractelement <8 x i32> %bc.1, i64 6
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %284, i32 %bc69125.1)
  %285 = getelementptr inbounds float, float addrspace(1)* %271, i64 112
  %286 = bitcast float addrspace(1)* %285 to i32 addrspace(1)*
  %bc70134.1 = extractelement <8 x i32> %bc.1, i64 7
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %286, i32 %bc70134.1)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Function Attrs: convergent nounwind
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float>, <8 x i32>, <8 x i32>, i32, i32, i32, i32, i1) #3

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_get_simd_size() local_unnamed_addr #4

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef) local_unnamed_addr #4

; Function Attrs: convergent
declare spir_func void @__builtin_IB_simd_block_write_1_global(i32 addrspace(1)* noundef, i32 noundef) local_unnamed_addr #4

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef) local_unnamed_addr #5

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_global_offset(i32 noundef) local_unnamed_addr #5

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)*) #6

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind
declare void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)*, i32) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: convergent nounwind readnone
declare i32 @llvm.genx.GenISA.WaveShuffleIndex.i32.i32.i32(i32, i32, i32) #8

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: convergent nounwind readnone
declare i64 @llvm.genx.GenISA.WaveShuffleIndex.i64.i32.i32(i64, i32, i32) #8

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }
attributes #8 = { convergent nounwind readnone }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!434, !434, !434, !434, !434}
!opencl.spir.version = !{!434, !434, !434, !434, !434}
!llvm.ident = !{!435, !435, !435, !435, !435}
!llvm.module.flags = !{!436}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE, !4}
!4 = !{!5, !6, !41}
!5 = !{!"function_type", i32 0}
!6 = !{!"implicit_arg_desc", !7, !8, !9, !10, !11, !12, !13, !14, !15, !18, !20, !22, !23, !25, !26, !28, !29, !31, !32, !34, !35, !37, !39}
!7 = !{i32 0}
!8 = !{i32 1}
!9 = !{i32 5}
!10 = !{i32 6}
!11 = !{i32 7}
!12 = !{i32 8}
!13 = !{i32 9}
!14 = !{i32 12}
!15 = !{i32 16, !16, !17}
!16 = !{!"explicit_arg_num", i32 1}
!17 = !{!"struct_arg_offset", i32 0}
!18 = !{i32 16, !16, !19}
!19 = !{!"struct_arg_offset", i32 8}
!20 = !{i32 16, !21, !17}
!21 = !{!"explicit_arg_num", i32 2}
!22 = !{i32 16, !21, !19}
!23 = !{i32 16, !24, !17}
!24 = !{!"explicit_arg_num", i32 5}
!25 = !{i32 16, !24, !19}
!26 = !{i32 16, !27, !17}
!27 = !{!"explicit_arg_num", i32 6}
!28 = !{i32 16, !27, !19}
!29 = !{i32 16, !30, !17}
!30 = !{!"explicit_arg_num", i32 8}
!31 = !{i32 16, !30, !19}
!32 = !{i32 16, !33, !17}
!33 = !{!"explicit_arg_num", i32 9}
!34 = !{i32 16, !33, !19}
!35 = !{i32 14, !36}
!36 = !{!"explicit_arg_num", i32 0}
!37 = !{i32 14, !38}
!38 = !{!"explicit_arg_num", i32 4}
!39 = !{i32 14, !40}
!40 = !{!"explicit_arg_num", i32 7}
!41 = !{!"sub_group_size", i32 8}
!42 = !{!"ModuleMD", !43, !44, !121, !286, !317, !333, !354, !364, !366, !367, !381, !382, !383, !384, !388, !389, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !406, !408, !412, !413, !414, !415, !416, !417, !418, !419, !420, !421, !422, !207, !423, !424, !425, !427, !430, !431, !432}
!43 = !{!"isPrecise", i1 false}
!44 = !{!"compOpt", !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120}
!45 = !{!"DenormsAreZero", i1 false}
!46 = !{!"BFTFDenormsAreZero", i1 false}
!47 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!48 = !{!"OptDisable", i1 false}
!49 = !{!"MadEnable", i1 true}
!50 = !{!"NoSignedZeros", i1 false}
!51 = !{!"NoNaNs", i1 false}
!52 = !{!"FloatRoundingMode", i32 0}
!53 = !{!"FloatCvtIntRoundingMode", i32 3}
!54 = !{!"LoadCacheDefault", i32 4}
!55 = !{!"StoreCacheDefault", i32 7}
!56 = !{!"VISAPreSchedRPThreshold", i32 0}
!57 = !{!"SetLoopUnrollThreshold", i32 0}
!58 = !{!"UnsafeMathOptimizations", i1 false}
!59 = !{!"disableCustomUnsafeOpts", i1 false}
!60 = !{!"disableReducePow", i1 false}
!61 = !{!"disableSqrtOpt", i1 false}
!62 = !{!"FiniteMathOnly", i1 false}
!63 = !{!"FastRelaxedMath", i1 false}
!64 = !{!"DashGSpecified", i1 false}
!65 = !{!"FastCompilation", i1 false}
!66 = !{!"UseScratchSpacePrivateMemory", i1 true}
!67 = !{!"RelaxedBuiltins", i1 false}
!68 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!69 = !{!"GreaterThan2GBBufferRequired", i1 true}
!70 = !{!"GreaterThan4GBBufferRequired", i1 false}
!71 = !{!"DisableA64WA", i1 false}
!72 = !{!"ForceEnableA64WA", i1 false}
!73 = !{!"PushConstantsEnable", i1 true}
!74 = !{!"HasPositivePointerOffset", i1 false}
!75 = !{!"HasBufferOffsetArg", i1 true}
!76 = !{!"BufferOffsetArgOptional", i1 true}
!77 = !{!"replaceGlobalOffsetsByZero", i1 false}
!78 = !{!"forcePixelShaderSIMDMode", i32 0}
!79 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!80 = !{!"UniformWGS", i1 false}
!81 = !{!"disableVertexComponentPacking", i1 false}
!82 = !{!"disablePartialVertexComponentPacking", i1 false}
!83 = !{!"PreferBindlessImages", i1 false}
!84 = !{!"UseBindlessMode", i1 false}
!85 = !{!"UseLegacyBindlessMode", i1 true}
!86 = !{!"disableMathRefactoring", i1 false}
!87 = !{!"atomicBranch", i1 false}
!88 = !{!"spillCompression", i1 false}
!89 = !{!"ForceInt32DivRemEmu", i1 false}
!90 = !{!"ForceInt32DivRemEmuSP", i1 false}
!91 = !{!"DisableFastestSingleCSSIMD", i1 false}
!92 = !{!"DisableFastestLinearScan", i1 false}
!93 = !{!"UseStatelessforPrivateMemory", i1 false}
!94 = !{!"EnableTakeGlobalAddress", i1 false}
!95 = !{!"IsLibraryCompilation", i1 false}
!96 = !{!"LibraryCompileSIMDSize", i32 0}
!97 = !{!"FastVISACompile", i1 false}
!98 = !{!"MatchSinCosPi", i1 false}
!99 = !{!"ExcludeIRFromZEBinary", i1 false}
!100 = !{!"EmitZeBinVISASections", i1 false}
!101 = !{!"FP64GenEmulationEnabled", i1 false}
!102 = !{!"FP64GenConvEmulationEnabled", i1 false}
!103 = !{!"allowDisableRematforCS", i1 false}
!104 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!105 = !{!"DisableCPSOmaskWA", i1 false}
!106 = !{!"DisableFastestGopt", i1 false}
!107 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!108 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!109 = !{!"DisableConstantCoalescing", i1 false}
!110 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!111 = !{!"WaEnableALTModeVisaWA", i1 false}
!112 = !{!"NewSpillCostFunction", i1 false}
!113 = !{!"ForceLargeGRFNum4RQ", i1 false}
!114 = !{!"DisableEUFusion", i1 false}
!115 = !{!"DisableFDivToFMulInvOpt", i1 false}
!116 = !{!"initializePhiSampleSourceWA", i1 false}
!117 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!118 = !{!"DisableLoosenSimd32Occu", i1 false}
!119 = !{!"FastestS1Options", i32 0}
!120 = !{!"EnableFastestForWaveIntrinsicsCS", i1 false}
!121 = !{!"FuncMD", !122, !123}
!122 = !{!"FuncMDMap[0]", void (%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE}
!123 = !{!"FuncMDValue[0]", !124, !125, !129, !130, !131, !152, !199, !200, !201, !202, !203, !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215, !226, !237, !248, !259, !270, !281, !282}
!124 = !{!"localOffsets"}
!125 = !{!"workGroupWalkOrder", !126, !127, !128}
!126 = !{!"dim0", i32 0}
!127 = !{!"dim1", i32 1}
!128 = !{!"dim2", i32 2}
!129 = !{!"funcArgs"}
!130 = !{!"functionType", !"KernelFunction"}
!131 = !{!"rtInfo", !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !151}
!132 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!133 = !{!"isContinuation", i1 false}
!134 = !{!"hasTraceRayPayload", i1 false}
!135 = !{!"hasHitAttributes", i1 false}
!136 = !{!"hasCallableData", i1 false}
!137 = !{!"ShaderStackSize", i32 0}
!138 = !{!"ShaderHash", i64 0}
!139 = !{!"ShaderName", !""}
!140 = !{!"ParentName", !""}
!141 = !{!"SlotNum", i1* null}
!142 = !{!"NOSSize", i32 0}
!143 = !{!"globalRootSignatureSize", i32 0}
!144 = !{!"Entries"}
!145 = !{!"SpillUnions"}
!146 = !{!"CustomHitAttrSizeInBytes", i32 0}
!147 = !{!"Types", !148, !149, !150}
!148 = !{!"FrameStartTys"}
!149 = !{!"ArgumentTys"}
!150 = !{!"FullFrameTys"}
!151 = !{!"Aliases"}
!152 = !{!"resAllocMD", !153, !154, !155, !156, !198}
!153 = !{!"uavsNumType", i32 4}
!154 = !{!"srvsNumType", i32 0}
!155 = !{!"samplersNumType", i32 0}
!156 = !{!"argAllocMDList", !157, !161, !164, !165, !166, !168, !169, !170, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197}
!157 = !{!"argAllocMDListVec[0]", !158, !159, !160}
!158 = !{!"type", i32 1}
!159 = !{!"extensionType", i32 -1}
!160 = !{!"indexType", i32 0}
!161 = !{!"argAllocMDListVec[1]", !162, !159, !163}
!162 = !{!"type", i32 0}
!163 = !{!"indexType", i32 -1}
!164 = !{!"argAllocMDListVec[2]", !162, !159, !163}
!165 = !{!"argAllocMDListVec[3]", !162, !159, !163}
!166 = !{!"argAllocMDListVec[4]", !158, !159, !167}
!167 = !{!"indexType", i32 1}
!168 = !{!"argAllocMDListVec[5]", !162, !159, !163}
!169 = !{!"argAllocMDListVec[6]", !162, !159, !163}
!170 = !{!"argAllocMDListVec[7]", !158, !159, !171}
!171 = !{!"indexType", i32 2}
!172 = !{!"argAllocMDListVec[8]", !162, !159, !163}
!173 = !{!"argAllocMDListVec[9]", !162, !159, !163}
!174 = !{!"argAllocMDListVec[10]", !162, !159, !163}
!175 = !{!"argAllocMDListVec[11]", !162, !159, !163}
!176 = !{!"argAllocMDListVec[12]", !162, !159, !163}
!177 = !{!"argAllocMDListVec[13]", !162, !159, !163}
!178 = !{!"argAllocMDListVec[14]", !162, !159, !163}
!179 = !{!"argAllocMDListVec[15]", !162, !159, !163}
!180 = !{!"argAllocMDListVec[16]", !162, !159, !163}
!181 = !{!"argAllocMDListVec[17]", !158, !159, !182}
!182 = !{!"indexType", i32 3}
!183 = !{!"argAllocMDListVec[18]", !162, !159, !163}
!184 = !{!"argAllocMDListVec[19]", !162, !159, !163}
!185 = !{!"argAllocMDListVec[20]", !162, !159, !163}
!186 = !{!"argAllocMDListVec[21]", !162, !159, !163}
!187 = !{!"argAllocMDListVec[22]", !162, !159, !163}
!188 = !{!"argAllocMDListVec[23]", !162, !159, !163}
!189 = !{!"argAllocMDListVec[24]", !162, !159, !163}
!190 = !{!"argAllocMDListVec[25]", !162, !159, !163}
!191 = !{!"argAllocMDListVec[26]", !162, !159, !163}
!192 = !{!"argAllocMDListVec[27]", !162, !159, !163}
!193 = !{!"argAllocMDListVec[28]", !162, !159, !163}
!194 = !{!"argAllocMDListVec[29]", !162, !159, !163}
!195 = !{!"argAllocMDListVec[30]", !162, !159, !163}
!196 = !{!"argAllocMDListVec[31]", !162, !159, !163}
!197 = !{!"argAllocMDListVec[32]", !162, !159, !163}
!198 = !{!"inlineSamplersMD"}
!199 = !{!"maxByteOffsets"}
!200 = !{!"IsInitializer", i1 false}
!201 = !{!"IsFinalizer", i1 false}
!202 = !{!"CompiledSubGroupsNumber", i32 0}
!203 = !{!"hasInlineVmeSamplers", i1 false}
!204 = !{!"localSize", i32 0}
!205 = !{!"localIDPresent", i1 false}
!206 = !{!"groupIDPresent", i1 false}
!207 = !{!"privateMemoryPerWI", i32 0}
!208 = !{!"prevFPOffset", i32 0}
!209 = !{!"globalIDPresent", i1 false}
!210 = !{!"hasSyncRTCalls", i1 false}
!211 = !{!"hasNonKernelArgLoad", i1 false}
!212 = !{!"hasNonKernelArgStore", i1 false}
!213 = !{!"hasNonKernelArgAtomic", i1 false}
!214 = !{!"UserAnnotations"}
!215 = !{!"m_OpenCLArgAddressSpaces", !216, !217, !218, !219, !220, !221, !222, !223, !224, !225}
!216 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!217 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!218 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!219 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!220 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!221 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!222 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!223 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!224 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!225 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!226 = !{!"m_OpenCLArgAccessQualifiers", !227, !228, !229, !230, !231, !232, !233, !234, !235, !236}
!227 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!228 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!229 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!230 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!231 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!232 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!233 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!234 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!235 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!236 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!237 = !{!"m_OpenCLArgTypes", !238, !239, !240, !241, !242, !243, !244, !245, !246, !247}
!238 = !{!"m_OpenCLArgTypesVec[0]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!239 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!240 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!241 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!242 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!243 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!244 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!245 = !{!"m_OpenCLArgTypesVec[7]", !"float*"}
!246 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!247 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!248 = !{!"m_OpenCLArgBaseTypes", !249, !250, !251, !252, !253, !254, !255, !256, !257, !258}
!249 = !{!"m_OpenCLArgBaseTypesVec[0]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!250 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!251 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!252 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!253 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!254 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!255 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!256 = !{!"m_OpenCLArgBaseTypesVec[7]", !"float*"}
!257 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!258 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!259 = !{!"m_OpenCLArgTypeQualifiers", !260, !261, !262, !263, !264, !265, !266, !267, !268, !269}
!260 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!261 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!262 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!263 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!264 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!265 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!266 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!267 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!268 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!269 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!270 = !{!"m_OpenCLArgNames", !271, !272, !273, !274, !275, !276, !277, !278, !279, !280}
!271 = !{!"m_OpenCLArgNamesVec[0]", !""}
!272 = !{!"m_OpenCLArgNamesVec[1]", !""}
!273 = !{!"m_OpenCLArgNamesVec[2]", !""}
!274 = !{!"m_OpenCLArgNamesVec[3]", !""}
!275 = !{!"m_OpenCLArgNamesVec[4]", !""}
!276 = !{!"m_OpenCLArgNamesVec[5]", !""}
!277 = !{!"m_OpenCLArgNamesVec[6]", !""}
!278 = !{!"m_OpenCLArgNamesVec[7]", !""}
!279 = !{!"m_OpenCLArgNamesVec[8]", !""}
!280 = !{!"m_OpenCLArgNamesVec[9]", !""}
!281 = !{!"m_OpenCLArgScalarAsPointers"}
!282 = !{!"m_OptsToDisablePerFunc", !283, !284, !285}
!283 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!284 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!285 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!286 = !{!"pushInfo", !287, !288, !289, !293, !294, !295, !296, !297, !298, !299, !300, !313, !314, !315, !316}
!287 = !{!"pushableAddresses"}
!288 = !{!"bindlessPushInfo"}
!289 = !{!"dynamicBufferInfo", !290, !291, !292}
!290 = !{!"firstIndex", i32 0}
!291 = !{!"numOffsets", i32 0}
!292 = !{!"forceDisabled", i1 false}
!293 = !{!"MaxNumberOfPushedBuffers", i32 0}
!294 = !{!"inlineConstantBufferSlot", i32 -1}
!295 = !{!"inlineConstantBufferOffset", i32 -1}
!296 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!297 = !{!"constants"}
!298 = !{!"inputs"}
!299 = !{!"constantReg"}
!300 = !{!"simplePushInfoArr", !301, !310, !311, !312}
!301 = !{!"simplePushInfoArrVec[0]", !302, !303, !304, !305, !306, !307, !308, !309}
!302 = !{!"cbIdx", i32 0}
!303 = !{!"pushableAddressGrfOffset", i32 -1}
!304 = !{!"pushableOffsetGrfOffset", i32 -1}
!305 = !{!"offset", i32 0}
!306 = !{!"size", i32 0}
!307 = !{!"isStateless", i1 false}
!308 = !{!"isBindless", i1 false}
!309 = !{!"simplePushLoads"}
!310 = !{!"simplePushInfoArrVec[1]", !302, !303, !304, !305, !306, !307, !308, !309}
!311 = !{!"simplePushInfoArrVec[2]", !302, !303, !304, !305, !306, !307, !308, !309}
!312 = !{!"simplePushInfoArrVec[3]", !302, !303, !304, !305, !306, !307, !308, !309}
!313 = !{!"simplePushBufferUsed", i32 0}
!314 = !{!"pushAnalysisWIInfos"}
!315 = !{!"inlineRTGlobalPtrOffset", i32 0}
!316 = !{!"rtSyncSurfPtrOffset", i32 0}
!317 = !{!"psInfo", !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332}
!318 = !{!"BlendStateDisabledMask", i8 0}
!319 = !{!"SkipSrc0Alpha", i1 false}
!320 = !{!"DualSourceBlendingDisabled", i1 false}
!321 = !{!"ForceEnableSimd32", i1 false}
!322 = !{!"outputDepth", i1 false}
!323 = !{!"outputStencil", i1 false}
!324 = !{!"outputMask", i1 false}
!325 = !{!"blendToFillEnabled", i1 false}
!326 = !{!"forceEarlyZ", i1 false}
!327 = !{!"hasVersionedLoop", i1 false}
!328 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!329 = !{!"NumSamples", i8 0}
!330 = !{!"blendOptimizationMode"}
!331 = !{!"colorOutputMask"}
!332 = !{!"WaDisableVRS", i1 false}
!333 = !{!"csInfo", !334, !335, !336, !337, !338, !56, !57, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !87, !88, !351, !352, !353}
!334 = !{!"maxWorkGroupSize", i32 0}
!335 = !{!"waveSize", i32 0}
!336 = !{!"ComputeShaderSecondCompile"}
!337 = !{!"forcedSIMDSize", i8 0}
!338 = !{!"forceTotalGRFNum", i32 0}
!339 = !{!"forceSpillCompression", i1 false}
!340 = !{!"allowLowerSimd", i1 false}
!341 = !{!"disableSimd32Slicing", i1 false}
!342 = !{!"disableSplitOnSpill", i1 false}
!343 = !{!"enableNewSpillCostFunction", i1 false}
!344 = !{!"forcedVISAPreRAScheduler", i1 false}
!345 = !{!"forceUniformBuffer", i1 false}
!346 = !{!"forceUniformSurfaceSampler", i1 false}
!347 = !{!"disableLocalIdOrderOptimizations", i1 false}
!348 = !{!"disableDispatchAlongY", i1 false}
!349 = !{!"neededThreadIdLayout", i1* null}
!350 = !{!"forceTileYWalk", i1 false}
!351 = !{!"walkOrderEnabled", i1 false}
!352 = !{!"walkOrderOverride", i32 0}
!353 = !{!"ResForHfPacking"}
!354 = !{!"msInfo", !355, !356, !357, !358, !359, !360, !361, !362, !363}
!355 = !{!"PrimitiveTopology", i32 3}
!356 = !{!"MaxNumOfPrimitives", i32 0}
!357 = !{!"MaxNumOfVertices", i32 0}
!358 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!359 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!360 = !{!"WorkGroupSize", i32 0}
!361 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!362 = !{!"IndexFormat", i32 6}
!363 = !{!"SubgroupSize", i32 0}
!364 = !{!"taskInfo", !365, !360, !361, !363}
!365 = !{!"MaxNumOfOutputs", i32 0}
!366 = !{!"NBarrierCnt", i32 0}
!367 = !{!"rtInfo", !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380}
!368 = !{!"RayQueryAllocSizeInBytes", i32 0}
!369 = !{!"NumContinuations", i32 0}
!370 = !{!"RTAsyncStackAddrspace", i32 -1}
!371 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!372 = !{!"SWHotZoneAddrspace", i32 -1}
!373 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!374 = !{!"SWStackAddrspace", i32 -1}
!375 = !{!"SWStackSurfaceStateOffset", i1* null}
!376 = !{!"RTSyncStackAddrspace", i32 -1}
!377 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!378 = !{!"doSyncDispatchRays", i1 false}
!379 = !{!"MemStyle", !"Xe"}
!380 = !{!"GlobalDataStyle", !"Xe"}
!381 = !{!"CurUniqueIndirectIdx", i32 0}
!382 = !{!"inlineDynTextures"}
!383 = !{!"inlineResInfoData"}
!384 = !{!"immConstant", !385, !386, !387}
!385 = !{!"data"}
!386 = !{!"sizes"}
!387 = !{!"zeroIdxs"}
!388 = !{!"stringConstants"}
!389 = !{!"inlineBuffers", !390, !394, !395}
!390 = !{!"inlineBuffersVec[0]", !391, !392, !393}
!391 = !{!"alignment", i32 0}
!392 = !{!"allocSize", i64 0}
!393 = !{!"Buffer"}
!394 = !{!"inlineBuffersVec[1]", !391, !392, !393}
!395 = !{!"inlineBuffersVec[2]", !391, !392, !393}
!396 = !{!"GlobalPointerProgramBinaryInfos"}
!397 = !{!"ConstantPointerProgramBinaryInfos"}
!398 = !{!"GlobalBufferAddressRelocInfo"}
!399 = !{!"ConstantBufferAddressRelocInfo"}
!400 = !{!"forceLscCacheList"}
!401 = !{!"SrvMap"}
!402 = !{!"RasterizerOrderedByteAddressBuffer"}
!403 = !{!"RasterizerOrderedViews"}
!404 = !{!"MinNOSPushConstantSize", i32 0}
!405 = !{!"inlineProgramScopeOffsets"}
!406 = !{!"shaderData", !407}
!407 = !{!"numReplicas", i32 0}
!408 = !{!"URBInfo", !409, !410, !411}
!409 = !{!"has64BVertexHeaderInput", i1 false}
!410 = !{!"has64BVertexHeaderOutput", i1 false}
!411 = !{!"hasVertexHeader", i1 true}
!412 = !{!"UseBindlessImage", i1 false}
!413 = !{!"enableRangeReduce", i1 false}
!414 = !{!"allowMatchMadOptimizationforVS", i1 false}
!415 = !{!"disableMatchMadOptimizationForCS", i1 false}
!416 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!417 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!418 = !{!"statefulResourcesNotAliased", i1 false}
!419 = !{!"disableMixMode", i1 false}
!420 = !{!"genericAccessesResolved", i1 false}
!421 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!422 = !{!"disableSeparateScratchWA", i1 false}
!423 = !{!"PrivateMemoryPerFG"}
!424 = !{!"m_OptsToDisable"}
!425 = !{!"capabilities", !426}
!426 = !{!"globalVariableDecorationsINTEL", i1 false}
!427 = !{!"m_ShaderResourceViewMcsMask", !428, !429}
!428 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!429 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!430 = !{!"computedDepthMode", i32 0}
!431 = !{!"isHDCFastClearShader", i1 false}
!432 = !{!"argRegisterReservations", !433}
!433 = !{!"argRegisterReservationsVec[0]", i32 0}
!434 = !{i32 2, i32 0}
!435 = !{!"clang version 14.0.5"}
!436 = !{i32 1, !"wchar_size", i32 4}
!437 = !{!438}
!438 = !{i32 4469}
