; ------------------------------------------------
; OCL_asm14db04af8499574b_codegen.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32-p131072:32:32:32-p131073:32:32:32-p131074:32:32:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm8ELm8ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_8x8_i32_8_global_v8i8_pi32_i32.24.exit:
  %10 = bitcast i64 %3 to <2 x i32>
  %11 = extractelement <2 x i32> %10, i32 0
  %12 = extractelement <2 x i32> %10, i32 1
  %13 = bitcast i64 %const_reg_qword1 to <2 x i32>
  %14 = extractelement <2 x i32> %13, i32 0
  %15 = extractelement <2 x i32> %13, i32 1
  %16 = bitcast i64 %const_reg_qword2 to <2 x i32>
  %17 = extractelement <2 x i32> %16, i32 0
  %18 = extractelement <2 x i32> %16, i32 1
  %19 = bitcast i64 %const_reg_qword3 to <2 x i32>
  %20 = extractelement <2 x i32> %19, i32 0
  %21 = extractelement <2 x i32> %19, i32 1
  %22 = bitcast i64 %const_reg_qword5 to <2 x i32>
  %23 = extractelement <2 x i32> %22, i32 0
  %24 = extractelement <2 x i32> %22, i32 1
  %25 = bitcast i64 %const_reg_qword6 to <2 x i32>
  %26 = extractelement <2 x i32> %25, i32 0
  %27 = extractelement <2 x i32> %25, i32 1
  %28 = bitcast i64 %const_reg_qword7 to <2 x i32>
  %29 = extractelement <2 x i32> %28, i32 0
  %30 = extractelement <2 x i32> %28, i32 1
  %31 = bitcast i64 %const_reg_qword9 to <2 x i32>
  %32 = extractelement <2 x i32> %31, i32 0
  %33 = extractelement <2 x i32> %31, i32 1
  %34 = bitcast i64 %const_reg_qword10 to <2 x i32>
  %35 = extractelement <2 x i32> %34, i32 0
  %36 = extractelement <2 x i32> %34, i32 1
  %37 = bitcast i64 %const_reg_qword11 to <2 x i32>
  %38 = extractelement <2 x i32> %37, i32 0
  %39 = extractelement <2 x i32> %37, i32 1
  %40 = extractelement <8 x i32> %payloadHeader, i32 0
  %41 = extractelement <8 x i32> %payloadHeader, i32 1
  %42 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %43 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %44 = extractelement <8 x i32> %r0, i32 1
  %45 = extractelement <8 x i32> %r0, i32 6
  %46 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %17, i32 %18, i32 %14, i32 %15)
  %47 = extractvalue { i32, i32 } %46, 0
  %48 = extractvalue { i32, i32 } %46, 1
  %49 = shl i32 %47, 2
  %50 = add i32 %49, %bufferOffset
  %51 = shl i32 %20, 2
  %52 = add i32 %50, %51
  %53 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %26, i32 %27, i32 %23, i32 %24)
  %54 = extractvalue { i32, i32 } %53, 0
  %55 = extractvalue { i32, i32 } %53, 1
  %56 = shl i32 %54, 1
  %57 = add i32 %56, %bufferOffset12
  %58 = shl i32 %29, 1
  %59 = add i32 %57, %58
  %60 = call { i32, i32 } @llvm.genx.GenISA.mul.pair(i32 %35, i32 %36, i32 %32, i32 %33)
  %61 = extractvalue { i32, i32 } %60, 0
  %62 = extractvalue { i32, i32 } %60, 1
  %63 = shl i32 %61, 1
  %64 = add i32 %63, %bufferOffset13
  %65 = shl i32 %38, 1
  %66 = add i32 %64, %65
  %67 = mul i32 %43, %45
  %localIdY15 = zext i16 %localIdY to i32
  %68 = add i32 %67, %localIdY15
  %69 = add i32 %68, %41
  %70 = call i32 @llvm.genx.GenISA.WaveShuffleIndex.i32.i32.i32(i32 %69, i32 0, i32 0)
  %71 = mul i32 %42, %44
  %localIdX21 = zext i16 %localIdX to i32
  %72 = add i32 %71, %localIdX21
  %73 = add i32 %72, %40
  %74 = zext i16 %localIdY to i32
  %75 = insertelement <2 x i32> undef, i32 %74, i32 0
  %76 = insertelement <2 x i32> %75, i32 0, i32 1
  %77 = bitcast <2 x i32> %76 to i64
  %78 = call i64 @llvm.genx.GenISA.WaveShuffleIndex.i64.i32.i32(i64 %77, i32 0, i32 0)
  %79 = bitcast i64 %78 to <2 x i32>
  %80 = extractelement <2 x i32> %79, i32 0
  %81 = extractelement <2 x i32> %79, i32 1
  %82 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 %70, i32 0, i32 %80, i32 %81)
  %83 = extractvalue { i32, i32 } %82, 0
  %84 = zext i16 %localIdX to i32
  %85 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 %73, i32 0, i32 %84, i32 0)
  %86 = extractvalue { i32, i32 } %85, 0
  %87 = extractvalue { i32, i32 } %85, 1
  %88 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %47, i32 %48, i32 %20, i32 %21)
  %89 = extractvalue { i32, i32 } %88, 0
  %90 = extractvalue { i32, i32 } %88, 1
  %91 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %89, i32 %90)
  %92 = extractvalue { i32, i32 } %91, 0
  %93 = shl i32 %92, 2
  %94 = add i32 %52, %93
  %95 = shl i32 %83, 7
  %96 = shl i32 %95, 2
  %97 = add i32 %94, %96
  %98 = or i32 %87, %12
  %99 = icmp ult i32 %98, 1
  br i1 %99, label %214, label %100, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !524

100:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_8x8_i32_8_global_v8i8_pi32_i32.24.exit
  %101 = lshr i32 %11, 20
  %102 = shl i32 %12, 12
  %103 = lshr i32 %11, 20
  %104 = or i32 %103, %102
  %105 = lshr i32 %12, 8
  %106 = uitofp i32 %105 to float
  %107 = and i32 %104, 1044480
  %108 = or i32 %107, %101
  %109 = uitofp i32 %108 to float
  %110 = and i32 %11, 1048575
  %111 = uitofp i32 %110 to float
  %112 = call float @llvm.fma.f32(float %109, float 0x4130000000000000, float %111) #4
  %113 = call float @llvm.fma.f32(float %106, float 0x4270000000000000, float %112) #4
  %114 = lshr i32 %87, 8
  %115 = uitofp i32 %114 to float
  %116 = fdiv float 1.000000e+00, %113
  %117 = call float @llvm.fma.f32(float %116, float 0xBE9C000000000000, float %116) #4
  %118 = fmul float %117, %115
  %119 = call float @llvm.floor.f32(float %118) #4
  %120 = fsub float 0.000000e+00, %111
  %121 = call float @llvm.fma.f32(float %120, float %119, float %115) #4
  %122 = fmul float %109, 0xC130000000000000
  %123 = call float @llvm.fma.f32(float %122, float %119, float %121) #4
  %124 = lshr i32 %86, 20
  %125 = shl i32 %87, 12
  %126 = lshr i32 %86, 20
  %127 = or i32 %126, %125
  %128 = and i32 %127, 1044480
  %129 = or i32 %128, %124
  %130 = uitofp i32 %129 to float
  %131 = call float @llvm.fma.f32(float %123, float 0x4130000000000000, float %130) #4
  %132 = fmul float %117, %131
  %133 = call float @llvm.floor.f32(float %132) #4
  %134 = fsub float 0.000000e+00, %109
  %135 = call float @llvm.fma.f32(float %134, float %133, float %123) #4
  %136 = fmul float %106, 0xC130000000000000
  %137 = call float @llvm.fma.f32(float %136, float %133, float %135) #4
  %138 = fmul float %111, 0x3EB0000000000000
  %139 = fmul float %138, %133
  %140 = call float @llvm.floor.f32(float %139) #4
  %141 = fsub float 0.000000e+00, %140
  %142 = call float @llvm.fma.f32(float %133, float %138, float %141) #4
  %143 = call float @llvm.fma.f32(float %140, float -1.000000e+00, float %137) #4
  %144 = call float @llvm.fma.f32(float %142, float 0xC130000000000000, float %130) #4
  %145 = and i32 %86, 1048575
  %146 = uitofp i32 %145 to float
  %147 = call float @llvm.fma.f32(float %144, float 0x4130000000000000, float %146) #4
  %148 = call float @llvm.fma.f32(float %143, float 0x4270000000000000, float %147) #4
  %149 = fmul float %117, %148
  %150 = call float @llvm.floor.f32(float %149) #4
  %151 = fsub float 0.000000e+00, %106
  %152 = call float @llvm.fma.f32(float %151, float %150, float %143) #4
  %153 = fmul float %138, %150
  %154 = call float @llvm.floor.f32(float %153) #4
  %155 = fsub float 0.000000e+00, %154
  %156 = call float @llvm.fma.f32(float %150, float %138, float %155) #4
  %157 = call float @llvm.fma.f32(float %154, float -1.000000e+00, float %144) #4
  %158 = call float @llvm.fma.f32(float %156, float 0xC130000000000000, float %146) #4
  %159 = fmul float %109, 0x3EB0000000000000
  %160 = fmul float %159, %150
  %161 = call float @llvm.floor.f32(float %160) #4
  %162 = fsub float 0.000000e+00, %161
  %163 = call float @llvm.fma.f32(float %150, float %159, float %162) #4
  %164 = call float @llvm.fma.f32(float %161, float -1.000000e+00, float %152) #4
  %165 = call float @llvm.fma.f32(float %163, float 0xC130000000000000, float %157) #4
  %166 = fptosi float %119 to i32
  %167 = fptosi float %133 to i32
  %168 = fptosi float %150 to i32
  %169 = call float @llvm.fma.f32(float %164, float 0x4130000000000000, float %165) #4
  %170 = call float @llvm.fma.f32(float %169, float 0x4130000000000000, float %158) #4
  %171 = fmul float %117, %170
  %172 = call float @llvm.floor.f32(float %171) #4
  %173 = fsub float 0.000000e+00, %172
  %174 = call float @llvm.fma.f32(float %173, float %106, float %164) #4
  %175 = call float @llvm.fma.f32(float %173, float %109, float %165) #4
  %176 = call float @llvm.fma.f32(float %173, float %111, float %158) #4
  %177 = fptosi float %172 to i32
  %178 = add i32 %177, %168
  %179 = fptosi float %175 to i32
  %180 = fptosi float %176 to i32
  %181 = ashr i32 %179, 31
  %182 = shl i32 %179, 20
  %183 = lshr i32 %179, 12
  %184 = shl i32 %181, 20
  %185 = or i32 %184, %183
  %186 = ashr i32 %180, 31
  %187 = fmul float %174, 0x3DF0000000000000
  %188 = call float @llvm.trunc.f32(float %187)
  %189 = call float @llvm.fma.f32(float %188, float 0xC1F0000000000000, float %174)
  %190 = fptoui float %189 to i32
  %191 = shl i32 %190, 8
  %192 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 0, i32 %191, i32 %180, i32 %186)
  %193 = extractvalue { i32, i32 } %192, 0
  %194 = extractvalue { i32, i32 } %192, 1
  %195 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %193, i32 %194, i32 %182, i32 %185)
  %196 = extractvalue { i32, i32 } %195, 0
  %197 = extractvalue { i32, i32 } %195, 1
  %198 = shl i32 %166, 8
  %199 = shl i32 %167, 20
  %200 = lshr i32 %167, 12
  %201 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %199, i32 %200, i32 0, i32 %198)
  %202 = extractvalue { i32, i32 } %201, 0
  %203 = extractvalue { i32, i32 } %201, 1
  %204 = icmp eq i32 %197, %12
  %205 = icmp uge i32 %196, %11
  %206 = and i1 %204, %205
  %207 = icmp ugt i32 %197, %12
  %208 = or i1 %206, %207
  %209 = sext i1 %208 to i32
  %210 = sub i32 0, %209
  %211 = add i32 %178, %210
  %212 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %202, i32 %203, i32 %211, i32 0)
  %213 = extractvalue { i32, i32 } %212, 0
  br label %__igcbuiltin_u64_udiv_sp.exit, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !525

214:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_Accumulator_RowMajor_8x8_i32_8_global_v8i8_pi32_i32.24.exit
  %tobool.i.i = icmp eq i32 %11, 0
  br i1 %tobool.i.i, label %.precompiled_u32divrem_sp.exit.i_crit_edge, label %if.end.i.i, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !525

if.end.i.i:                                       ; preds = %214
  %215 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %11) #4
  %conv.i.i = fptoui float %215 to i32
  %sub.i.i = sub i32 %11, %conv.i.i
  %216 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %86) #4
  %div.i.i = fdiv float 1.000000e+00, %215, !fpmath !526
  %217 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %div.i.i, float 0xBE98000000000000, float %div.i.i) #4
  %218 = call float @llvm.genx.GenISA.mul.rtz.f32.f32.f32(float %216, float %217) #4
  %conv3.i.i = fptoui float %216 to i32
  %sub4.i.i = sub i32 %86, %conv3.i.i
  %conv8.i.i = fptoui float %218 to i32
  %219 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %sub.i.i) #4
  %220 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %sub4.i.i) #4
  %221 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %conv8.i.i) #4
  %222 = fsub float 0.000000e+00, %215
  %223 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %222, float %221, float %216) #4
  %224 = fsub float 0.000000e+00, %219
  %225 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %224, float %221, float %220) #4
  %226 = call float @llvm.genx.GenISA.add.rtz.f32.f32.f32(float %223, float %225) #4
  %227 = call float @llvm.genx.GenISA.mul.rtz.f32.f32.f32(float %217, float %226) #4
  %conv16.i.i = fptoui float %227 to i32
  %add.i.i = add i32 %conv16.i.i, %conv8.i.i
  %mul.i.i = mul i32 %add.i.i, %11
  %sub17.i.i = sub i32 %86, %mul.i.i
  %cmp.i.i = icmp uge i32 %sub17.i.i, %11
  %228 = sext i1 %cmp.i.i to i32
  %229 = sub i32 0, %228
  %add19.i.i = add i32 %add.i.i, %229
  br label %precompiled_u32divrem_sp.exit.i, !stats.blockFrequency.digits !527, !stats.blockFrequency.scale !528

.precompiled_u32divrem_sp.exit.i_crit_edge:       ; preds = %214
  br label %precompiled_u32divrem_sp.exit.i, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !530

precompiled_u32divrem_sp.exit.i:                  ; preds = %.precompiled_u32divrem_sp.exit.i_crit_edge, %if.end.i.i
  %retval.0.i.i = phi i32 [ %add19.i.i, %if.end.i.i ], [ -1, %.precompiled_u32divrem_sp.exit.i_crit_edge ]
  br label %__igcbuiltin_u64_udiv_sp.exit, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !525

__igcbuiltin_u64_udiv_sp.exit:                    ; preds = %precompiled_u32divrem_sp.exit.i, %100
  %230 = phi i32 [ %retval.0.i.i, %precompiled_u32divrem_sp.exit.i ], [ %213, %100 ]
  %231 = shl i32 %230, 3
  %232 = shl i32 %231, 2
  %233 = add i32 %97, %232
  %234 = inttoptr i32 %233 to i32 addrspace(131072)*
  %235 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %234)
  %236 = add i32 %233, 64
  %237 = inttoptr i32 %236 to i32 addrspace(131072)*
  %238 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %237)
  %239 = add i32 %233, 128
  %240 = inttoptr i32 %239 to i32 addrspace(131072)*
  %241 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %240)
  %242 = add i32 %233, 192
  %243 = inttoptr i32 %242 to i32 addrspace(131072)*
  %244 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %243)
  %245 = add i32 %233, 256
  %246 = inttoptr i32 %245 to i32 addrspace(131072)*
  %247 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %246)
  %248 = add i32 %233, 320
  %249 = inttoptr i32 %248 to i32 addrspace(131072)*
  %250 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %249)
  %251 = add i32 %233, 384
  %252 = inttoptr i32 %251 to i32 addrspace(131072)*
  %253 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %252)
  %254 = add i32 %233, 448
  %255 = inttoptr i32 %254 to i32 addrspace(131072)*
  %256 = call float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)* %255)
  %.28.vec.insert.assembled.vect = insertelement <8 x float> undef, float %235, i32 0
  %.28.vec.insert.assembled.vect170 = insertelement <8 x float> %.28.vec.insert.assembled.vect, float %238, i32 1
  %.28.vec.insert.assembled.vect171 = insertelement <8 x float> %.28.vec.insert.assembled.vect170, float %241, i32 2
  %.28.vec.insert.assembled.vect172 = insertelement <8 x float> %.28.vec.insert.assembled.vect171, float %244, i32 3
  %.28.vec.insert.assembled.vect173 = insertelement <8 x float> %.28.vec.insert.assembled.vect172, float %247, i32 4
  %.28.vec.insert.assembled.vect174 = insertelement <8 x float> %.28.vec.insert.assembled.vect173, float %250, i32 5
  %.28.vec.insert.assembled.vect175 = insertelement <8 x float> %.28.vec.insert.assembled.vect174, float %253, i32 6
  %.28.vec.insert.assembled.vect176 = insertelement <8 x float> %.28.vec.insert.assembled.vect175, float %256, i32 7
  %257 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %54, i32 %55, i32 %29, i32 %30)
  %258 = extractvalue { i32, i32 } %257, 0
  %259 = extractvalue { i32, i32 } %257, 1
  %260 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %258, i32 %259)
  %261 = extractvalue { i32, i32 } %260, 0
  %262 = shl i32 %261, 1
  %263 = add i32 %59, %262
  %264 = shl i32 %83, 8
  %265 = shl i32 %264, 1
  %266 = add i32 %263, %265
  %267 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %61, i32 %62, i32 %38, i32 %39)
  %268 = extractvalue { i32, i32 } %267, 0
  %269 = extractvalue { i32, i32 } %267, 1
  %270 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %268, i32 %269)
  %271 = extractvalue { i32, i32 } %270, 0
  %272 = shl i32 %271, 1
  %273 = add i32 %66, %272
  %274 = shl i32 %230, 4
  %275 = shl i32 %274, 1
  %276 = add i32 %273, %275
  %277 = inttoptr i32 %266 to i32 addrspace(131073)*
  %278 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %277)
  %279 = add i32 %266, 64
  %280 = inttoptr i32 %279 to i32 addrspace(131073)*
  %281 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %280)
  %282 = add i32 %266, 128
  %283 = inttoptr i32 %282 to i32 addrspace(131073)*
  %284 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %283)
  %285 = add i32 %266, 192
  %286 = inttoptr i32 %285 to i32 addrspace(131073)*
  %287 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %286)
  %288 = add i32 %266, 256
  %289 = inttoptr i32 %288 to i32 addrspace(131073)*
  %290 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %289)
  %291 = add i32 %266, 320
  %292 = inttoptr i32 %291 to i32 addrspace(131073)*
  %293 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %292)
  %294 = add i32 %266, 384
  %295 = inttoptr i32 %294 to i32 addrspace(131073)*
  %296 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %295)
  %297 = add i32 %266, 448
  %298 = inttoptr i32 %297 to i32 addrspace(131073)*
  %299 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %298)
  %.28.vec.insert24.assembled.vect = insertelement <8 x i32> undef, i32 %278, i32 0
  %.28.vec.insert24.assembled.vect177 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect, i32 %281, i32 1
  %.28.vec.insert24.assembled.vect178 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect177, i32 %284, i32 2
  %.28.vec.insert24.assembled.vect179 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect178, i32 %287, i32 3
  %.28.vec.insert24.assembled.vect180 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect179, i32 %290, i32 4
  %.28.vec.insert24.assembled.vect181 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect180, i32 %293, i32 5
  %.28.vec.insert24.assembled.vect182 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect181, i32 %296, i32 6
  %.28.vec.insert24.assembled.vect183 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect182, i32 %299, i32 7
  %300 = inttoptr i32 %276 to i32 addrspace(131074)*
  %301 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %300)
  %302 = add i32 %276, 64
  %303 = inttoptr i32 %302 to i32 addrspace(131074)*
  %304 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %303)
  %305 = add i32 %276, 128
  %306 = inttoptr i32 %305 to i32 addrspace(131074)*
  %307 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %306)
  %308 = add i32 %276, 192
  %309 = inttoptr i32 %308 to i32 addrspace(131074)*
  %310 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %309)
  %311 = add i32 %276, 256
  %312 = inttoptr i32 %311 to i32 addrspace(131074)*
  %313 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %312)
  %314 = add i32 %276, 320
  %315 = inttoptr i32 %314 to i32 addrspace(131074)*
  %316 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %315)
  %317 = add i32 %276, 384
  %318 = inttoptr i32 %317 to i32 addrspace(131074)*
  %319 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %318)
  %320 = add i32 %276, 448
  %321 = inttoptr i32 %320 to i32 addrspace(131074)*
  %322 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %321)
  %.28.vec.insert41.assembled.vect = insertelement <8 x i32> undef, i32 %301, i32 0
  %.28.vec.insert41.assembled.vect184 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect, i32 %304, i32 1
  %.28.vec.insert41.assembled.vect185 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect184, i32 %307, i32 2
  %.28.vec.insert41.assembled.vect186 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect185, i32 %310, i32 3
  %.28.vec.insert41.assembled.vect187 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect186, i32 %313, i32 4
  %.28.vec.insert41.assembled.vect188 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect187, i32 %316, i32 5
  %.28.vec.insert41.assembled.vect189 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect188, i32 %319, i32 6
  %.28.vec.insert41.assembled.vect190 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect189, i32 %322, i32 7
  %dpas = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %.28.vec.insert.assembled.vect176, <8 x i32> %.28.vec.insert24.assembled.vect183, <8 x i32> %.28.vec.insert41.assembled.vect190, i32 11, i32 11, i32 8, i32 8, i1 false)
  %323 = add i32 %266, 32
  %324 = inttoptr i32 %323 to i32 addrspace(131073)*
  %325 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %324)
  %326 = add i32 %266, 96
  %327 = inttoptr i32 %326 to i32 addrspace(131073)*
  %328 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %327)
  %329 = add i32 %266, 160
  %330 = inttoptr i32 %329 to i32 addrspace(131073)*
  %331 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %330)
  %332 = add i32 %266, 224
  %333 = inttoptr i32 %332 to i32 addrspace(131073)*
  %334 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %333)
  %335 = add i32 %266, 288
  %336 = inttoptr i32 %335 to i32 addrspace(131073)*
  %337 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %336)
  %338 = add i32 %266, 352
  %339 = inttoptr i32 %338 to i32 addrspace(131073)*
  %340 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %339)
  %341 = add i32 %266, 416
  %342 = inttoptr i32 %341 to i32 addrspace(131073)*
  %343 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %342)
  %344 = add i32 %266, 480
  %345 = inttoptr i32 %344 to i32 addrspace(131073)*
  %346 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %345)
  %.28.vec.insert24.assembled.vect.1 = insertelement <8 x i32> undef, i32 %325, i32 0
  %.28.vec.insert24.assembled.vect177.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect.1, i32 %328, i32 1
  %.28.vec.insert24.assembled.vect178.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect177.1, i32 %331, i32 2
  %.28.vec.insert24.assembled.vect179.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect178.1, i32 %334, i32 3
  %.28.vec.insert24.assembled.vect180.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect179.1, i32 %337, i32 4
  %.28.vec.insert24.assembled.vect181.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect180.1, i32 %340, i32 5
  %.28.vec.insert24.assembled.vect182.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect181.1, i32 %343, i32 6
  %.28.vec.insert24.assembled.vect183.1 = insertelement <8 x i32> %.28.vec.insert24.assembled.vect182.1, i32 %346, i32 7
  %347 = add i32 %276, 512
  %348 = inttoptr i32 %347 to i32 addrspace(131074)*
  %349 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %348)
  %350 = add i32 %276, 576
  %351 = inttoptr i32 %350 to i32 addrspace(131074)*
  %352 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %351)
  %353 = add i32 %276, 640
  %354 = inttoptr i32 %353 to i32 addrspace(131074)*
  %355 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %354)
  %356 = add i32 %276, 704
  %357 = inttoptr i32 %356 to i32 addrspace(131074)*
  %358 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %357)
  %359 = add i32 %276, 768
  %360 = inttoptr i32 %359 to i32 addrspace(131074)*
  %361 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %360)
  %362 = add i32 %276, 832
  %363 = inttoptr i32 %362 to i32 addrspace(131074)*
  %364 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %363)
  %365 = add i32 %276, 896
  %366 = inttoptr i32 %365 to i32 addrspace(131074)*
  %367 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %366)
  %368 = add i32 %276, 960
  %369 = inttoptr i32 %368 to i32 addrspace(131074)*
  %370 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)* %369)
  %.28.vec.insert41.assembled.vect.1 = insertelement <8 x i32> undef, i32 %349, i32 0
  %.28.vec.insert41.assembled.vect184.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect.1, i32 %352, i32 1
  %.28.vec.insert41.assembled.vect185.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect184.1, i32 %355, i32 2
  %.28.vec.insert41.assembled.vect186.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect185.1, i32 %358, i32 3
  %.28.vec.insert41.assembled.vect187.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect186.1, i32 %361, i32 4
  %.28.vec.insert41.assembled.vect188.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect187.1, i32 %364, i32 5
  %.28.vec.insert41.assembled.vect189.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect188.1, i32 %367, i32 6
  %.28.vec.insert41.assembled.vect190.1 = insertelement <8 x i32> %.28.vec.insert41.assembled.vect189.1, i32 %370, i32 7
  %dpas.1 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %dpas, <8 x i32> %.28.vec.insert24.assembled.vect183.1, <8 x i32> %.28.vec.insert41.assembled.vect190.1, i32 11, i32 11, i32 8, i32 8, i1 false)
  %371 = extractelement <8 x float> %dpas.1, i32 0
  %372 = bitcast float %371 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %234, i32 %372)
  %373 = extractelement <8 x float> %dpas.1, i32 1
  %374 = bitcast float %373 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %237, i32 %374)
  %375 = extractelement <8 x float> %dpas.1, i32 2
  %376 = bitcast float %375 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %240, i32 %376)
  %377 = extractelement <8 x float> %dpas.1, i32 3
  %378 = bitcast float %377 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %243, i32 %378)
  %379 = extractelement <8 x float> %dpas.1, i32 4
  %380 = bitcast float %379 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %246, i32 %380)
  %381 = extractelement <8 x float> %dpas.1, i32 5
  %382 = bitcast float %381 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %249, i32 %382)
  %383 = extractelement <8 x float> %dpas.1, i32 6
  %384 = bitcast float %383 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %252, i32 %384)
  %385 = extractelement <8 x float> %dpas.1, i32 7
  %386 = bitcast float %385 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)* %255, i32 %386)
  ret void, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !524
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Desc: XeHP SDV: dot product accumulate systolic
; Output: dst
; Arg 0: src0(acc)
; Arg 1: src1
; Arg 2: src2
; Arg 3: src1's precision
; Arg 4: src2's precision
; Arg 5: systolic depth
; Arg 6: repeat count
; Arg 7: isDpasw
; Function Attrs: convergent nounwind
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float>, <8 x i32>, <8 x i32>, i32, i32, i32, i32, i1) #2

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare float @llvm.genx.GenISA.simdBlockRead.f32.p1i32(i32 addrspace(1)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind
declare void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)*, i32) #4

; Function Desc: Read from a specific lane
; Output: TODO: could be changed to anytype when support has been backported from llvm 3.7
; Arg 0: value
; Arg 1: lane
; Arg 2: helperLaneMode : 0: not used; 1: helper lanes participatein wave ops, 2: helper lanes do not participate in wave ops.
; Function Attrs: convergent nounwind readnone
declare i64 @llvm.genx.GenISA.WaveShuffleIndex.i64.i32.i32(i64, i32, i32) #5

; Function Desc: Read from a specific lane
; Output: TODO: could be changed to anytype when support has been backported from llvm 3.7
; Arg 0: value
; Arg 1: lane
; Arg 2: helperLaneMode : 0: not used; 1: helper lanes participatein wave ops, 2: helper lanes do not participate in wave ops.
; Function Attrs: convergent nounwind readnone
declare i32 @llvm.genx.GenISA.WaveShuffleIndex.i32.i32.i32(i32, i32, i32) #5

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fma.f32(float, float, float) #6

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.floor.f32(float) #6

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: nounwind readnone
declare float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float, float, float) #7

; Function Desc: 
; Output: result
; Arg 0: input
; Function Attrs: nounwind readnone
declare float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind readnone
declare float @llvm.genx.GenISA.mul.rtz.f32.f32.f32(float, float) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind readnone
declare float @llvm.genx.GenISA.add.rtz.f32.f32.f32(float, float) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare float @llvm.genx.GenISA.simdBlockRead.f32.p131072i32(i32 addrspace(131072)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind
declare void @llvm.genx.GenISA.simdBlockWrite.p131072i32.i32(i32 addrspace(131072)*, i32) #4

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p131074i32(i32 addrspace(131074)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Function Attrs: nounwind readnone
declare { i32, i32 } @llvm.genx.GenISA.mul.pair(i32, i32, i32, i32) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Function Attrs: nounwind readnone
declare { i32, i32 } @llvm.genx.GenISA.sub.pair(i32, i32, i32, i32) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Function Attrs: nounwind readnone
declare { i32, i32 } @llvm.genx.GenISA.add.pair(i32, i32, i32, i32) #7

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.trunc.f32(float) #6

attributes #0 = { convergent nounwind "less-precise-fpmad"="false" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind readonly }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind readnone }
attributes #6 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #7 = { nounwind readnone }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!518, !518, !518, !518, !518, !518, !518}
!opencl.spir.version = !{!518, !518, !518, !518, !518}
!llvm.ident = !{!519, !519, !519, !519, !519, !520, !520, !521, !521}
!llvm.module.flags = !{!522}
!printf.strings = !{}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm8ELm8ELm16EE, !4}
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
!42 = !{!"ModuleMD", !43, !44, !141, !311, !342, !343, !347, !350, !351, !352, !387, !413, !426, !427, !428, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !456, !457, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !477, !481, !482, !483, !484, !485, !486, !487, !488, !489, !490, !491, !492, !493, !494, !495, !496, !497, !232, !498, !501, !502, !504, !507, !508, !509, !511, !512, !513}
!43 = !{!"isPrecise", i1 false}
!44 = !{!"compOpt", !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140}
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
!66 = !{!"UseScratchSpacePrivateMemory", i1 false}
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
!79 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!80 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!81 = !{!"UniformWGS", i1 false}
!82 = !{!"disableVertexComponentPacking", i1 false}
!83 = !{!"disablePartialVertexComponentPacking", i1 false}
!84 = !{!"PreferBindlessImages", i1 false}
!85 = !{!"UseBindlessMode", i1 false}
!86 = !{!"UseLegacyBindlessMode", i1 true}
!87 = !{!"disableMathRefactoring", i1 false}
!88 = !{!"atomicBranch", i1 false}
!89 = !{!"spillCompression", i1 false}
!90 = !{!"DisableEarlyOut", i1 false}
!91 = !{!"ForceInt32DivRemEmu", i1 false}
!92 = !{!"ForceInt32DivRemEmuSP", i1 false}
!93 = !{!"WaveIntrinsicUsed", i1 false}
!94 = !{!"DisableMultiPolyPS", i1 false}
!95 = !{!"NeedTexture3DLODWA", i1 false}
!96 = !{!"DisableFastestSingleCSSIMD", i1 false}
!97 = !{!"DisableFastestLinearScan", i1 false}
!98 = !{!"UseStatelessforPrivateMemory", i1 false}
!99 = !{!"EnableTakeGlobalAddress", i1 false}
!100 = !{!"IsLibraryCompilation", i1 false}
!101 = !{!"LibraryCompileSIMDSize", i32 0}
!102 = !{!"FastVISACompile", i1 false}
!103 = !{!"MatchSinCosPi", i1 false}
!104 = !{!"ExcludeIRFromZEBinary", i1 false}
!105 = !{!"EmitZeBinVISASections", i1 false}
!106 = !{!"FP64GenEmulationEnabled", i1 false}
!107 = !{!"FP64GenConvEmulationEnabled", i1 false}
!108 = !{!"allowDisableRematforCS", i1 false}
!109 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!110 = !{!"DisableCPSOmaskWA", i1 false}
!111 = !{!"DisableFastestGopt", i1 false}
!112 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!113 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!114 = !{!"DisableConstantCoalescing", i1 false}
!115 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!116 = !{!"WaEnableALTModeVisaWA", i1 false}
!117 = !{!"WaEnableAtomicWaveFusion", i1 false}
!118 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!119 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!120 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!121 = !{!"ForceCBThroughSampler3D", i1 false}
!122 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!123 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!124 = !{!"WaZeroSLMBeforeUse", i1 false}
!125 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!126 = !{!"NewSpillCostFunction", i1 false}
!127 = !{!"EnableVRT", i1 false}
!128 = !{!"ForceLargeGRFNum4RQ", i1 false}
!129 = !{!"Enable2xGRFRetry", i1 false}
!130 = !{!"Detect2xGRFCandidate", i1 false}
!131 = !{!"EnableURBWritesMerging", i1 true}
!132 = !{!"DisableEUFusion", i1 false}
!133 = !{!"DisableFDivToFMulInvOpt", i1 false}
!134 = !{!"initializePhiSampleSourceWA", i1 false}
!135 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!136 = !{!"DisableLoosenSimd32Occu", i1 false}
!137 = !{!"FastestS1Options", i32 0}
!138 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!139 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!140 = !{!"DisableLscSamplerRouting", i1 false}
!141 = !{!"FuncMD", !142, !143}
!142 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm8ELm8ELm16EE}
!143 = !{!"FuncMDValue[0]", !144, !145, !149, !150, !151, !152, !153, !154, !155, !177, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !251, !262, !273, !284, !295, !306, !307}
!144 = !{!"localOffsets"}
!145 = !{!"workGroupWalkOrder", !146, !147, !148}
!146 = !{!"dim0", i32 0}
!147 = !{!"dim1", i32 1}
!148 = !{!"dim2", i32 2}
!149 = !{!"funcArgs"}
!150 = !{!"functionType", !"KernelFunction"}
!151 = !{!"inlineDynConstants"}
!152 = !{!"inlineDynRootConstant"}
!153 = !{!"inlineDynConstantDescTable"}
!154 = !{!"m_pInterestingConstants"}
!155 = !{!"rtInfo", !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !175, !176, !127}
!156 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!157 = !{!"isContinuation", i1 false}
!158 = !{!"hasTraceRayPayload", i1 false}
!159 = !{!"hasHitAttributes", i1 false}
!160 = !{!"hasCallableData", i1 false}
!161 = !{!"ShaderStackSize", i32 0}
!162 = !{!"ShaderHash", i64 0}
!163 = !{!"ShaderName", !""}
!164 = !{!"ParentName", !""}
!165 = !{!"SlotNum", i1* null}
!166 = !{!"NOSSize", i32 0}
!167 = !{!"globalRootSignatureSize", i32 0}
!168 = !{!"Entries"}
!169 = !{!"SpillUnions"}
!170 = !{!"CustomHitAttrSizeInBytes", i32 0}
!171 = !{!"Types", !172, !173, !174}
!172 = !{!"FrameStartTys"}
!173 = !{!"ArgumentTys"}
!174 = !{!"FullFrameTys"}
!175 = !{!"Aliases"}
!176 = !{!"NumGRF", i32 0}
!177 = !{!"resAllocMD", !178, !179, !180, !181, !223}
!178 = !{!"uavsNumType", i32 7}
!179 = !{!"srvsNumType", i32 0}
!180 = !{!"samplersNumType", i32 0}
!181 = !{!"argAllocMDList", !182, !186, !189, !190, !191, !193, !194, !195, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222}
!182 = !{!"argAllocMDListVec[0]", !183, !184, !185}
!183 = !{!"type", i32 1}
!184 = !{!"extensionType", i32 -1}
!185 = !{!"indexType", i32 0}
!186 = !{!"argAllocMDListVec[1]", !187, !184, !188}
!187 = !{!"type", i32 0}
!188 = !{!"indexType", i32 -1}
!189 = !{!"argAllocMDListVec[2]", !187, !184, !188}
!190 = !{!"argAllocMDListVec[3]", !187, !184, !188}
!191 = !{!"argAllocMDListVec[4]", !183, !184, !192}
!192 = !{!"indexType", i32 1}
!193 = !{!"argAllocMDListVec[5]", !187, !184, !188}
!194 = !{!"argAllocMDListVec[6]", !187, !184, !188}
!195 = !{!"argAllocMDListVec[7]", !183, !184, !196}
!196 = !{!"indexType", i32 2}
!197 = !{!"argAllocMDListVec[8]", !187, !184, !188}
!198 = !{!"argAllocMDListVec[9]", !187, !184, !188}
!199 = !{!"argAllocMDListVec[10]", !187, !184, !188}
!200 = !{!"argAllocMDListVec[11]", !187, !184, !188}
!201 = !{!"argAllocMDListVec[12]", !187, !184, !188}
!202 = !{!"argAllocMDListVec[13]", !187, !184, !188}
!203 = !{!"argAllocMDListVec[14]", !187, !184, !188}
!204 = !{!"argAllocMDListVec[15]", !187, !184, !188}
!205 = !{!"argAllocMDListVec[16]", !187, !184, !188}
!206 = !{!"argAllocMDListVec[17]", !183, !184, !207}
!207 = !{!"indexType", i32 3}
!208 = !{!"argAllocMDListVec[18]", !187, !184, !188}
!209 = !{!"argAllocMDListVec[19]", !187, !184, !188}
!210 = !{!"argAllocMDListVec[20]", !187, !184, !188}
!211 = !{!"argAllocMDListVec[21]", !187, !184, !188}
!212 = !{!"argAllocMDListVec[22]", !187, !184, !188}
!213 = !{!"argAllocMDListVec[23]", !187, !184, !188}
!214 = !{!"argAllocMDListVec[24]", !187, !184, !188}
!215 = !{!"argAllocMDListVec[25]", !187, !184, !188}
!216 = !{!"argAllocMDListVec[26]", !187, !184, !188}
!217 = !{!"argAllocMDListVec[27]", !187, !184, !188}
!218 = !{!"argAllocMDListVec[28]", !187, !184, !188}
!219 = !{!"argAllocMDListVec[29]", !187, !184, !188}
!220 = !{!"argAllocMDListVec[30]", !187, !184, !188}
!221 = !{!"argAllocMDListVec[31]", !187, !184, !188}
!222 = !{!"argAllocMDListVec[32]", !187, !184, !188}
!223 = !{!"inlineSamplersMD"}
!224 = !{!"maxByteOffsets"}
!225 = !{!"IsInitializer", i1 false}
!226 = !{!"IsFinalizer", i1 false}
!227 = !{!"CompiledSubGroupsNumber", i32 0}
!228 = !{!"hasInlineVmeSamplers", i1 false}
!229 = !{!"localSize", i32 0}
!230 = !{!"localIDPresent", i1 false}
!231 = !{!"groupIDPresent", i1 false}
!232 = !{!"privateMemoryPerWI", i32 0}
!233 = !{!"prevFPOffset", i32 0}
!234 = !{!"globalIDPresent", i1 false}
!235 = !{!"hasSyncRTCalls", i1 false}
!236 = !{!"hasNonKernelArgLoad", i1 false}
!237 = !{!"hasNonKernelArgStore", i1 false}
!238 = !{!"hasNonKernelArgAtomic", i1 false}
!239 = !{!"UserAnnotations"}
!240 = !{!"m_OpenCLArgAddressSpaces", !241, !242, !243, !244, !245, !246, !247, !248, !249, !250}
!241 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!242 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 0}
!243 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 0}
!244 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 0}
!245 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!246 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 0}
!247 = !{!"m_OpenCLArgAddressSpacesVec[6]", i32 0}
!248 = !{!"m_OpenCLArgAddressSpacesVec[7]", i32 1}
!249 = !{!"m_OpenCLArgAddressSpacesVec[8]", i32 0}
!250 = !{!"m_OpenCLArgAddressSpacesVec[9]", i32 0}
!251 = !{!"m_OpenCLArgAccessQualifiers", !252, !253, !254, !255, !256, !257, !258, !259, !260, !261}
!252 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!253 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!254 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!255 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!256 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!257 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!258 = !{!"m_OpenCLArgAccessQualifiersVec[6]", !"none"}
!259 = !{!"m_OpenCLArgAccessQualifiersVec[7]", !"none"}
!260 = !{!"m_OpenCLArgAccessQualifiersVec[8]", !"none"}
!261 = !{!"m_OpenCLArgAccessQualifiersVec[9]", !"none"}
!262 = !{!"m_OpenCLArgTypes", !263, !264, !265, !266, !267, !268, !269, !270, !271, !272}
!263 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!264 = !{!"m_OpenCLArgTypesVec[1]", !"class.sycl::_V1::range"}
!265 = !{!"m_OpenCLArgTypesVec[2]", !"class.sycl::_V1::range"}
!266 = !{!"m_OpenCLArgTypesVec[3]", !"long"}
!267 = !{!"m_OpenCLArgTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!268 = !{!"m_OpenCLArgTypesVec[5]", !"class.sycl::_V1::range"}
!269 = !{!"m_OpenCLArgTypesVec[6]", !"class.sycl::_V1::range"}
!270 = !{!"m_OpenCLArgTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!271 = !{!"m_OpenCLArgTypesVec[8]", !"class.sycl::_V1::range"}
!272 = !{!"m_OpenCLArgTypesVec[9]", !"class.sycl::_V1::range"}
!273 = !{!"m_OpenCLArgBaseTypes", !274, !275, !276, !277, !278, !279, !280, !281, !282, !283}
!274 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!275 = !{!"m_OpenCLArgBaseTypesVec[1]", !"class.sycl::_V1::range"}
!276 = !{!"m_OpenCLArgBaseTypesVec[2]", !"class.sycl::_V1::range"}
!277 = !{!"m_OpenCLArgBaseTypesVec[3]", !"long"}
!278 = !{!"m_OpenCLArgBaseTypesVec[4]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!279 = !{!"m_OpenCLArgBaseTypesVec[5]", !"class.sycl::_V1::range"}
!280 = !{!"m_OpenCLArgBaseTypesVec[6]", !"class.sycl::_V1::range"}
!281 = !{!"m_OpenCLArgBaseTypesVec[7]", !"class.sycl::_V1::ext::oneapi::bfloat16*"}
!282 = !{!"m_OpenCLArgBaseTypesVec[8]", !"class.sycl::_V1::range"}
!283 = !{!"m_OpenCLArgBaseTypesVec[9]", !"class.sycl::_V1::range"}
!284 = !{!"m_OpenCLArgTypeQualifiers", !285, !286, !287, !288, !289, !290, !291, !292, !293, !294}
!285 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!286 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!287 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!288 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!289 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!290 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!291 = !{!"m_OpenCLArgTypeQualifiersVec[6]", !""}
!292 = !{!"m_OpenCLArgTypeQualifiersVec[7]", !""}
!293 = !{!"m_OpenCLArgTypeQualifiersVec[8]", !""}
!294 = !{!"m_OpenCLArgTypeQualifiersVec[9]", !""}
!295 = !{!"m_OpenCLArgNames", !296, !297, !298, !299, !300, !301, !302, !303, !304, !305}
!296 = !{!"m_OpenCLArgNamesVec[0]", !""}
!297 = !{!"m_OpenCLArgNamesVec[1]", !""}
!298 = !{!"m_OpenCLArgNamesVec[2]", !""}
!299 = !{!"m_OpenCLArgNamesVec[3]", !""}
!300 = !{!"m_OpenCLArgNamesVec[4]", !""}
!301 = !{!"m_OpenCLArgNamesVec[5]", !""}
!302 = !{!"m_OpenCLArgNamesVec[6]", !""}
!303 = !{!"m_OpenCLArgNamesVec[7]", !""}
!304 = !{!"m_OpenCLArgNamesVec[8]", !""}
!305 = !{!"m_OpenCLArgNamesVec[9]", !""}
!306 = !{!"m_OpenCLArgScalarAsPointers"}
!307 = !{!"m_OptsToDisablePerFunc", !308, !309, !310}
!308 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!309 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!310 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!311 = !{!"pushInfo", !312, !313, !314, !318, !319, !320, !321, !322, !323, !324, !325, !338, !339, !340, !341}
!312 = !{!"pushableAddresses"}
!313 = !{!"bindlessPushInfo"}
!314 = !{!"dynamicBufferInfo", !315, !316, !317}
!315 = !{!"firstIndex", i32 0}
!316 = !{!"numOffsets", i32 0}
!317 = !{!"forceDisabled", i1 false}
!318 = !{!"MaxNumberOfPushedBuffers", i32 0}
!319 = !{!"inlineConstantBufferSlot", i32 -1}
!320 = !{!"inlineConstantBufferOffset", i32 -1}
!321 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!322 = !{!"constants"}
!323 = !{!"inputs"}
!324 = !{!"constantReg"}
!325 = !{!"simplePushInfoArr", !326, !335, !336, !337}
!326 = !{!"simplePushInfoArrVec[0]", !327, !328, !329, !330, !331, !332, !333, !334}
!327 = !{!"cbIdx", i32 0}
!328 = !{!"pushableAddressGrfOffset", i32 -1}
!329 = !{!"pushableOffsetGrfOffset", i32 -1}
!330 = !{!"offset", i32 0}
!331 = !{!"size", i32 0}
!332 = !{!"isStateless", i1 false}
!333 = !{!"isBindless", i1 false}
!334 = !{!"simplePushLoads"}
!335 = !{!"simplePushInfoArrVec[1]", !327, !328, !329, !330, !331, !332, !333, !334}
!336 = !{!"simplePushInfoArrVec[2]", !327, !328, !329, !330, !331, !332, !333, !334}
!337 = !{!"simplePushInfoArrVec[3]", !327, !328, !329, !330, !331, !332, !333, !334}
!338 = !{!"simplePushBufferUsed", i32 0}
!339 = !{!"pushAnalysisWIInfos"}
!340 = !{!"inlineRTGlobalPtrOffset", i32 0}
!341 = !{!"rtSyncSurfPtrOffset", i32 0}
!342 = !{!"WaEnableICBPromotion", i1 false}
!343 = !{!"vsInfo", !344, !345, !346}
!344 = !{!"DrawIndirectBufferIndex", i32 -1}
!345 = !{!"vertexReordering", i32 -1}
!346 = !{!"MaxNumOfOutputs", i32 0}
!347 = !{!"hsInfo", !348, !349}
!348 = !{!"numPatchAttributesPatchBaseName", !""}
!349 = !{!"numVertexAttributesPatchBaseName", !""}
!350 = !{!"dsInfo", !346}
!351 = !{!"gsInfo", !346}
!352 = !{!"psInfo", !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !364, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !382, !383, !384, !385, !386}
!353 = !{!"BlendStateDisabledMask", i8 0}
!354 = !{!"SkipSrc0Alpha", i1 false}
!355 = !{!"DualSourceBlendingDisabled", i1 false}
!356 = !{!"ForceEnableSimd32", i1 false}
!357 = !{!"outputDepth", i1 false}
!358 = !{!"outputStencil", i1 false}
!359 = !{!"outputMask", i1 false}
!360 = !{!"blendToFillEnabled", i1 false}
!361 = !{!"forceEarlyZ", i1 false}
!362 = !{!"hasVersionedLoop", i1 false}
!363 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!364 = !{!"requestCPSizeRelevant", i1 false}
!365 = !{!"requestCPSize", i1 false}
!366 = !{!"texelMaskFastClearMode", !"Disabled"}
!367 = !{!"NumSamples", i8 0}
!368 = !{!"blendOptimizationMode"}
!369 = !{!"colorOutputMask"}
!370 = !{!"ProvokingVertexModeNosIndex", i32 0}
!371 = !{!"ProvokingVertexModeNosPatch", !""}
!372 = !{!"ProvokingVertexModeLast", !"Negative"}
!373 = !{!"VertexAttributesBypass", i1 false}
!374 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!375 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!376 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!377 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!378 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!379 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!380 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!381 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!382 = !{!"generatePatchesForRTWriteSends", i1 false}
!383 = !{!"forceVMask", i1 false}
!384 = !{!"WaDisableVRS", i1 false}
!385 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!386 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!387 = !{!"csInfo", !388, !389, !390, !391, !392, !56, !57, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !89, !406, !407, !408, !409, !410, !411, !412}
!388 = !{!"maxWorkGroupSize", i32 0}
!389 = !{!"waveSize", i32 0}
!390 = !{!"ComputeShaderSecondCompile"}
!391 = !{!"forcedSIMDSize", i8 0}
!392 = !{!"forceTotalGRFNum", i32 0}
!393 = !{!"forceSpillCompression", i1 false}
!394 = !{!"allowLowerSimd", i1 false}
!395 = !{!"disableSimd32Slicing", i1 false}
!396 = !{!"disableSplitOnSpill", i1 false}
!397 = !{!"enableNewSpillCostFunction", i1 false}
!398 = !{!"forceVISAPreSched", i1 false}
!399 = !{!"forceUniformBuffer", i1 false}
!400 = !{!"forceUniformSurfaceSampler", i1 false}
!401 = !{!"disableLocalIdOrderOptimizations", i1 false}
!402 = !{!"disableDispatchAlongY", i1 false}
!403 = !{!"neededThreadIdLayout", i1* null}
!404 = !{!"forceTileYWalk", i1 false}
!405 = !{!"atomicBranch", i32 0}
!406 = !{!"disableEarlyOut", i1 false}
!407 = !{!"walkOrderEnabled", i1 false}
!408 = !{!"walkOrderOverride", i32 0}
!409 = !{!"ResForHfPacking"}
!410 = !{!"hasWaveMatrix", i1 false}
!411 = !{!"constantFoldSimdSize", i1 false}
!412 = !{!"isNodeShader", i1 false}
!413 = !{!"msInfo", !414, !415, !416, !417, !418, !419, !420, !421, !422, !423, !424, !372, !370, !425}
!414 = !{!"PrimitiveTopology", i32 3}
!415 = !{!"MaxNumOfPrimitives", i32 0}
!416 = !{!"MaxNumOfVertices", i32 0}
!417 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!418 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!419 = !{!"WorkGroupSize", i32 0}
!420 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!421 = !{!"IndexFormat", i32 6}
!422 = !{!"SubgroupSize", i32 0}
!423 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!424 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!425 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!426 = !{!"taskInfo", !346, !419, !420, !422}
!427 = !{!"NBarrierCnt", i32 0}
!428 = !{!"rtInfo", !429, !430, !431, !432, !433, !434, !435, !436, !437, !438, !439, !440, !441, !442}
!429 = !{!"RayQueryAllocSizeInBytes", i32 0}
!430 = !{!"NumContinuations", i32 0}
!431 = !{!"RTAsyncStackAddrspace", i32 -1}
!432 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!433 = !{!"SWHotZoneAddrspace", i32 -1}
!434 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!435 = !{!"SWStackAddrspace", i32 -1}
!436 = !{!"SWStackSurfaceStateOffset", i1* null}
!437 = !{!"RTSyncStackAddrspace", i32 -1}
!438 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!439 = !{!"doSyncDispatchRays", i1 false}
!440 = !{!"MemStyle", !"Xe"}
!441 = !{!"GlobalDataStyle", !"Xe"}
!442 = !{!"NeedsBTD", i1 true}
!443 = !{!"EnableTextureIndirection", i1 false}
!444 = !{!"EnableSamplerIndirection", i1 false}
!445 = !{!"samplerStateStride", i32 0}
!446 = !{!"samplerStateOffset", i32 0}
!447 = !{!"textureStateStride", i32 0}
!448 = !{!"textureStateOffset", i32 0}
!449 = !{!"CurUniqueIndirectIdx", i32 0}
!450 = !{!"inlineDynTextures"}
!451 = !{!"inlineResInfoData"}
!452 = !{!"immConstant", !453, !454, !455}
!453 = !{!"data"}
!454 = !{!"sizes"}
!455 = !{!"zeroIdxs"}
!456 = !{!"stringConstants"}
!457 = !{!"inlineBuffers", !458, !462, !463}
!458 = !{!"inlineBuffersVec[0]", !459, !460, !461}
!459 = !{!"alignment", i32 0}
!460 = !{!"allocSize", i64 0}
!461 = !{!"Buffer"}
!462 = !{!"inlineBuffersVec[1]", !459, !460, !461}
!463 = !{!"inlineBuffersVec[2]", !459, !460, !461}
!464 = !{!"GlobalPointerProgramBinaryInfos"}
!465 = !{!"ConstantPointerProgramBinaryInfos"}
!466 = !{!"GlobalBufferAddressRelocInfo"}
!467 = !{!"ConstantBufferAddressRelocInfo"}
!468 = !{!"forceLscCacheList"}
!469 = !{!"SrvMap"}
!470 = !{!"RootConstantBufferOffsetInBytes"}
!471 = !{!"RasterizerOrderedByteAddressBuffer"}
!472 = !{!"RasterizerOrderedViews"}
!473 = !{!"MinNOSPushConstantSize", i32 2}
!474 = !{!"inlineProgramScopeOffsets"}
!475 = !{!"shaderData", !476}
!476 = !{!"numReplicas", i32 0}
!477 = !{!"URBInfo", !478, !479, !480}
!478 = !{!"has64BVertexHeaderInput", i1 false}
!479 = !{!"has64BVertexHeaderOutput", i1 false}
!480 = !{!"hasVertexHeader", i1 true}
!481 = !{!"m_ForcePullModel", i1 false}
!482 = !{!"UseBindlessImage", i1 false}
!483 = !{!"enableRangeReduce", i1 false}
!484 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!485 = !{!"enableFRemToSRemOpt", i1 false}
!486 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!487 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!488 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!489 = !{!"allowMatchMadOptimizationforVS", i1 false}
!490 = !{!"disableMatchMadOptimizationForCS", i1 false}
!491 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!492 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!493 = !{!"statefulResourcesNotAliased", i1 false}
!494 = !{!"disableMixMode", i1 false}
!495 = !{!"genericAccessesResolved", i1 false}
!496 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!497 = !{!"disableSeparateScratchWA", i1 false}
!498 = !{!"PrivateMemoryPerFG", !499, !500}
!499 = !{!"PrivateMemoryPerFGMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm8ELm8ELm16EE}
!500 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!501 = !{!"m_OptsToDisable"}
!502 = !{!"capabilities", !503}
!503 = !{!"globalVariableDecorationsINTEL", i1 false}
!504 = !{!"m_ShaderResourceViewMcsMask", !505, !506}
!505 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!506 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!507 = !{!"computedDepthMode", i32 0}
!508 = !{!"isHDCFastClearShader", i1 false}
!509 = !{!"argRegisterReservations", !510}
!510 = !{!"argRegisterReservationsVec[0]", i32 0}
!511 = !{!"SIMD16_SpillThreshold", i8 0}
!512 = !{!"SIMD32_SpillThreshold", i8 0}
!513 = !{!"m_CacheControlOption", !514, !515, !516, !517}
!514 = !{!"LscLoadCacheControlOverride", i8 0}
!515 = !{!"LscStoreCacheControlOverride", i8 0}
!516 = !{!"TgmLoadCacheControlOverride", i8 0}
!517 = !{!"TgmStoreCacheControlOverride", i8 0}
!518 = !{i32 2, i32 0}
!519 = !{!"clang version 14.0.5"}
!520 = !{!"clang version 9.0.0 (2dbee6917918354f14026a637370445d50bdaf6a)"}
!521 = !{!"clang version 9.0.0 (c68f557a081b1b2339a42d7cd6af3c2ab18c6061)"}
!522 = !{i32 1, !"wchar_size", i32 4}
!523 = !{!"5764607523034234880"}
!524 = !{!"-59"}
!525 = !{!"-60"}
!526 = !{float 2.500000e+00}
!527 = !{!"14274266247513343513"}
!528 = !{!"-62"}
!529 = !{!"17568327689247192015"}
!530 = !{!"-63"}
