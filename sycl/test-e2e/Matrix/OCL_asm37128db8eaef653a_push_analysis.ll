; ------------------------------------------------
; OCL_asm37128db8eaef653a_push_analysis.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32-p131072:32:32:32-p131073:32:32:32-p131074:32:32:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, float addrspace(1)* align 4 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
.preheader.preheader:
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
  %49 = shl i32 %47, 1
  %50 = add i32 %49, %bufferOffset
  %51 = shl i32 %20, 1
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
  %63 = shl i32 %61, 2
  %64 = add i32 %63, %bufferOffset13
  %65 = shl i32 %38, 2
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
  %84 = extractvalue { i32, i32 } %82, 1
  %85 = zext i16 %localIdX to i32
  %86 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 %73, i32 0, i32 %85, i32 0)
  %87 = extractvalue { i32, i32 } %86, 0
  %88 = extractvalue { i32, i32 } %86, 1
  %89 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %47, i32 %48, i32 %20, i32 %21)
  %90 = extractvalue { i32, i32 } %89, 0
  %91 = extractvalue { i32, i32 } %89, 1
  %92 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %90, i32 %91)
  %93 = extractvalue { i32, i32 } %92, 0
  %94 = shl i32 %93, 1
  %95 = add i32 %52, %94
  %96 = or i32 %88, %12
  %97 = icmp ult i32 %96, 1
  br i1 %97, label %98, label %114

98:                                               ; preds = %.preheader.preheader
  %tobool.i.i = icmp eq i32 %11, 0
  br i1 %tobool.i.i, label %.precompiled_u32divrem_sp.exit.i_crit_edge, label %if.end.i.i

.precompiled_u32divrem_sp.exit.i_crit_edge:       ; preds = %98
  br label %precompiled_u32divrem_sp.exit.i

if.end.i.i:                                       ; preds = %98
  %99 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %11) #4
  %conv.i.i = fptoui float %99 to i32
  %sub.i.i = sub i32 %11, %conv.i.i
  %100 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %sub.i.i) #4
  %101 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %87) #4
  %conv3.i.i = fptoui float %101 to i32
  %sub4.i.i = sub i32 %87, %conv3.i.i
  %102 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %sub4.i.i) #4
  %div.i.i = fdiv float 1.000000e+00, %99, !fpmath !441
  %103 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %div.i.i, float 0xBE98000000000000, float %div.i.i) #4
  %104 = call float @llvm.genx.GenISA.mul.rtz.f32.f32.f32(float %101, float %103) #4
  %conv8.i.i = fptoui float %104 to i32
  %105 = call float @llvm.genx.GenISA.uitof.rtz.f32.i32(i32 %conv8.i.i) #4
  %106 = fsub float 0.000000e+00, %99
  %107 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %106, float %105, float %101) #4
  %108 = fsub float 0.000000e+00, %100
  %109 = call float @llvm.genx.GenISA.fma.rtz.f32.f32.f32.f32(float %108, float %105, float %102) #4
  %110 = call float @llvm.genx.GenISA.add.rtz.f32.f32.f32(float %107, float %109) #4
  %111 = call float @llvm.genx.GenISA.mul.rtz.f32.f32.f32(float %103, float %110) #4
  %conv16.i.i = fptoui float %111 to i32
  %add.i.i = add i32 %conv16.i.i, %conv8.i.i
  %mul.i.i = mul i32 %add.i.i, %11
  %sub17.i.i = sub i32 %87, %mul.i.i
  %cmp.i.i = icmp uge i32 %sub17.i.i, %11
  %112 = sext i1 %cmp.i.i to i32
  %113 = sub i32 0, %112
  %add19.i.i = add i32 %add.i.i, %113
  br label %precompiled_u32divrem_sp.exit.i

precompiled_u32divrem_sp.exit.i:                  ; preds = %.precompiled_u32divrem_sp.exit.i_crit_edge, %if.end.i.i
  %retval.0.i.i = phi i32 [ %add19.i.i, %if.end.i.i ], [ -1, %.precompiled_u32divrem_sp.exit.i_crit_edge ]
  br label %__igcbuiltin_u64_udiv_sp.exit

114:                                              ; preds = %.preheader.preheader
  %115 = lshr i32 %11, 20
  %116 = shl i32 %12, 12
  %117 = lshr i32 %11, 20
  %118 = or i32 %117, %116
  %119 = lshr i32 %12, 8
  %120 = uitofp i32 %119 to float
  %121 = and i32 %118, 1044480
  %122 = or i32 %121, %115
  %123 = uitofp i32 %122 to float
  %124 = and i32 %11, 1048575
  %125 = uitofp i32 %124 to float
  %126 = call float @llvm.fma.f32(float %123, float 0x4130000000000000, float %125) #4
  %127 = call float @llvm.fma.f32(float %120, float 0x4270000000000000, float %126) #4
  %128 = lshr i32 %88, 8
  %129 = uitofp i32 %128 to float
  %130 = fdiv float 1.000000e+00, %127
  %131 = call float @llvm.fma.f32(float %130, float 0xBE9C000000000000, float %130) #4
  %132 = fmul float %131, %129
  %133 = call float @llvm.floor.f32(float %132) #4
  %134 = fsub float 0.000000e+00, %125
  %135 = call float @llvm.fma.f32(float %134, float %133, float %129) #4
  %136 = fmul float %123, 0xC130000000000000
  %137 = call float @llvm.fma.f32(float %136, float %133, float %135) #4
  %138 = lshr i32 %87, 20
  %139 = shl i32 %88, 12
  %140 = lshr i32 %87, 20
  %141 = or i32 %140, %139
  %142 = and i32 %141, 1044480
  %143 = or i32 %142, %138
  %144 = uitofp i32 %143 to float
  %145 = call float @llvm.fma.f32(float %137, float 0x4130000000000000, float %144) #4
  %146 = fmul float %131, %145
  %147 = call float @llvm.floor.f32(float %146) #4
  %148 = fsub float 0.000000e+00, %123
  %149 = call float @llvm.fma.f32(float %148, float %147, float %137) #4
  %150 = fmul float %120, 0xC130000000000000
  %151 = call float @llvm.fma.f32(float %150, float %147, float %149) #4
  %152 = fmul float %125, 0x3EB0000000000000
  %153 = fmul float %152, %147
  %154 = call float @llvm.floor.f32(float %153) #4
  %155 = fsub float 0.000000e+00, %154
  %156 = call float @llvm.fma.f32(float %147, float %152, float %155) #4
  %157 = call float @llvm.fma.f32(float %154, float -1.000000e+00, float %151) #4
  %158 = call float @llvm.fma.f32(float %156, float 0xC130000000000000, float %144) #4
  %159 = and i32 %87, 1048575
  %160 = uitofp i32 %159 to float
  %161 = call float @llvm.fma.f32(float %158, float 0x4130000000000000, float %160) #4
  %162 = call float @llvm.fma.f32(float %157, float 0x4270000000000000, float %161) #4
  %163 = fmul float %131, %162
  %164 = call float @llvm.floor.f32(float %163) #4
  %165 = fsub float 0.000000e+00, %120
  %166 = call float @llvm.fma.f32(float %165, float %164, float %157) #4
  %167 = fmul float %152, %164
  %168 = call float @llvm.floor.f32(float %167) #4
  %169 = fsub float 0.000000e+00, %168
  %170 = call float @llvm.fma.f32(float %164, float %152, float %169) #4
  %171 = call float @llvm.fma.f32(float %168, float -1.000000e+00, float %158) #4
  %172 = call float @llvm.fma.f32(float %170, float 0xC130000000000000, float %160) #4
  %173 = fmul float %123, 0x3EB0000000000000
  %174 = fmul float %173, %164
  %175 = call float @llvm.floor.f32(float %174) #4
  %176 = fsub float 0.000000e+00, %175
  %177 = call float @llvm.fma.f32(float %164, float %173, float %176) #4
  %178 = call float @llvm.fma.f32(float %175, float -1.000000e+00, float %166) #4
  %179 = call float @llvm.fma.f32(float %177, float 0xC130000000000000, float %171) #4
  %180 = fptosi float %133 to i32
  %181 = fptosi float %147 to i32
  %182 = fptosi float %164 to i32
  %183 = call float @llvm.fma.f32(float %178, float 0x4130000000000000, float %179) #4
  %184 = call float @llvm.fma.f32(float %183, float 0x4130000000000000, float %172) #4
  %185 = fmul float %131, %184
  %186 = call float @llvm.floor.f32(float %185) #4
  %187 = fsub float 0.000000e+00, %186
  %188 = call float @llvm.fma.f32(float %187, float %120, float %178) #4
  %189 = call float @llvm.fma.f32(float %187, float %123, float %179) #4
  %190 = call float @llvm.fma.f32(float %187, float %125, float %172) #4
  %191 = fptosi float %186 to i32
  %192 = add i32 %191, %182
  %193 = fptosi float %189 to i32
  %194 = fptosi float %190 to i32
  %195 = ashr i32 %193, 31
  %196 = shl i32 %193, 20
  %197 = lshr i32 %193, 12
  %198 = shl i32 %195, 20
  %199 = or i32 %198, %197
  %200 = ashr i32 %194, 31
  %201 = fmul float %188, 0x3DF0000000000000
  %202 = call float @llvm.trunc.f32(float %201)
  %203 = call float @llvm.fma.f32(float %202, float 0xC1F0000000000000, float %188)
  %204 = fptoui float %203 to i32
  %205 = shl i32 %204, 8
  %206 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 0, i32 %205, i32 %194, i32 %200)
  %207 = extractvalue { i32, i32 } %206, 0
  %208 = extractvalue { i32, i32 } %206, 1
  %209 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %207, i32 %208, i32 %196, i32 %199)
  %210 = extractvalue { i32, i32 } %209, 0
  %211 = extractvalue { i32, i32 } %209, 1
  %212 = shl i32 %180, 8
  %213 = shl i32 %181, 20
  %214 = lshr i32 %181, 12
  %215 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %213, i32 %214, i32 0, i32 %212)
  %216 = extractvalue { i32, i32 } %215, 0
  %217 = extractvalue { i32, i32 } %215, 1
  %218 = icmp uge i32 %210, %11
  %219 = icmp eq i32 %211, %12
  %220 = and i1 %219, %218
  %221 = icmp ugt i32 %211, %12
  %222 = or i1 %220, %221
  %223 = sext i1 %222 to i32
  %224 = sub i32 0, %223
  %225 = add i32 %192, %224
  %226 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %216, i32 %217, i32 %225, i32 0)
  %227 = extractvalue { i32, i32 } %226, 0
  br label %__igcbuiltin_u64_udiv_sp.exit

__igcbuiltin_u64_udiv_sp.exit:                    ; preds = %precompiled_u32divrem_sp.exit.i, %114
  %228 = phi i32 [ %retval.0.i.i, %precompiled_u32divrem_sp.exit.i ], [ %227, %114 ]
  %229 = shl i32 %228, 4
  %230 = shl i32 %229, 1
  %231 = add i32 %95, %230
  %232 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %54, i32 %55, i32 %29, i32 %30)
  %233 = extractvalue { i32, i32 } %232, 0
  %234 = extractvalue { i32, i32 } %232, 1
  %235 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %233, i32 %234)
  %236 = extractvalue { i32, i32 } %235, 0
  %237 = shl i32 %236, 1
  %238 = add i32 %59, %237
  %239 = shl i32 %83, 9
  %240 = lshr i32 %83, 23
  %241 = shl i32 %84, 9
  %242 = or i32 %241, %240
  %243 = inttoptr i32 %231 to i32 addrspace(131072)*
  %244 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %243)
  %245 = add i32 %231, 64
  %246 = inttoptr i32 %245 to i32 addrspace(131072)*
  %247 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %246)
  %248 = add i32 %231, 128
  %249 = inttoptr i32 %248 to i32 addrspace(131072)*
  %250 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %249)
  %251 = add i32 %231, 192
  %252 = inttoptr i32 %251 to i32 addrspace(131072)*
  %253 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %252)
  %254 = add i32 %231, 256
  %255 = inttoptr i32 %254 to i32 addrspace(131072)*
  %256 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %255)
  %257 = add i32 %231, 320
  %258 = inttoptr i32 %257 to i32 addrspace(131072)*
  %259 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %258)
  %260 = add i32 %231, 384
  %261 = inttoptr i32 %260 to i32 addrspace(131072)*
  %262 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %261)
  %263 = add i32 %231, 448
  %264 = inttoptr i32 %263 to i32 addrspace(131072)*
  %265 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %264)
  %266 = insertelement <8 x i32> undef, i32 %244, i32 0
  %267 = insertelement <8 x i32> %266, i32 %247, i32 1
  %268 = insertelement <8 x i32> %267, i32 %250, i32 2
  %269 = insertelement <8 x i32> %268, i32 %253, i32 3
  %270 = insertelement <8 x i32> %269, i32 %256, i32 4
  %271 = insertelement <8 x i32> %270, i32 %259, i32 5
  %272 = insertelement <8 x i32> %271, i32 %262, i32 6
  %273 = insertelement <8 x i32> %272, i32 %265, i32 7
  %274 = shl i32 %239, 1
  %275 = add i32 %238, %274
  %276 = inttoptr i32 %275 to i32 addrspace(131073)*
  %277 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %276)
  %278 = add i32 %275, 64
  %279 = inttoptr i32 %278 to i32 addrspace(131073)*
  %280 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %279)
  %281 = add i32 %275, 128
  %282 = inttoptr i32 %281 to i32 addrspace(131073)*
  %283 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %282)
  %284 = add i32 %275, 192
  %285 = inttoptr i32 %284 to i32 addrspace(131073)*
  %286 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %285)
  %287 = add i32 %275, 256
  %288 = inttoptr i32 %287 to i32 addrspace(131073)*
  %289 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %288)
  %290 = add i32 %275, 320
  %291 = inttoptr i32 %290 to i32 addrspace(131073)*
  %292 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %291)
  %293 = add i32 %275, 384
  %294 = inttoptr i32 %293 to i32 addrspace(131073)*
  %295 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %294)
  %296 = add i32 %275, 448
  %297 = inttoptr i32 %296 to i32 addrspace(131073)*
  %298 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %297)
  %299 = insertelement <8 x i32> undef, i32 %277, i32 0
  %300 = insertelement <8 x i32> %299, i32 %280, i32 1
  %301 = insertelement <8 x i32> %300, i32 %283, i32 2
  %302 = insertelement <8 x i32> %301, i32 %286, i32 3
  %303 = insertelement <8 x i32> %302, i32 %289, i32 4
  %304 = insertelement <8 x i32> %303, i32 %292, i32 5
  %305 = insertelement <8 x i32> %304, i32 %295, i32 6
  %306 = insertelement <8 x i32> %305, i32 %298, i32 7
  %307 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x i32> %306, <8 x i32> %273, i32 9, i32 9, i32 8, i32 8, i1 false)
  %308 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %239, i32 %242, i32 256, i32 0)
  %309 = extractvalue { i32, i32 } %308, 0
  %310 = shl i32 %309, 1
  %311 = add i32 %238, %310
  %312 = inttoptr i32 %311 to i32 addrspace(131073)*
  %313 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %312)
  %314 = add i32 %311, 64
  %315 = inttoptr i32 %314 to i32 addrspace(131073)*
  %316 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %315)
  %317 = add i32 %311, 128
  %318 = inttoptr i32 %317 to i32 addrspace(131073)*
  %319 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %318)
  %320 = add i32 %311, 192
  %321 = inttoptr i32 %320 to i32 addrspace(131073)*
  %322 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %321)
  %323 = add i32 %311, 256
  %324 = inttoptr i32 %323 to i32 addrspace(131073)*
  %325 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %324)
  %326 = add i32 %311, 320
  %327 = inttoptr i32 %326 to i32 addrspace(131073)*
  %328 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %327)
  %329 = add i32 %311, 384
  %330 = inttoptr i32 %329 to i32 addrspace(131073)*
  %331 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %330)
  %332 = add i32 %311, 448
  %333 = inttoptr i32 %332 to i32 addrspace(131073)*
  %334 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %333)
  %335 = insertelement <8 x i32> undef, i32 %313, i32 0
  %336 = insertelement <8 x i32> %335, i32 %316, i32 1
  %337 = insertelement <8 x i32> %336, i32 %319, i32 2
  %338 = insertelement <8 x i32> %337, i32 %322, i32 3
  %339 = insertelement <8 x i32> %338, i32 %325, i32 4
  %340 = insertelement <8 x i32> %339, i32 %328, i32 5
  %341 = insertelement <8 x i32> %340, i32 %331, i32 6
  %342 = insertelement <8 x i32> %341, i32 %334, i32 7
  %343 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x i32> %342, <8 x i32> %273, i32 9, i32 9, i32 8, i32 8, i1 false)
  %344 = add i32 %231, 512
  %345 = inttoptr i32 %344 to i32 addrspace(131072)*
  %346 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %345)
  %347 = add i32 %231, 576
  %348 = inttoptr i32 %347 to i32 addrspace(131072)*
  %349 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %348)
  %350 = add i32 %231, 640
  %351 = inttoptr i32 %350 to i32 addrspace(131072)*
  %352 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %351)
  %353 = add i32 %231, 704
  %354 = inttoptr i32 %353 to i32 addrspace(131072)*
  %355 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %354)
  %356 = add i32 %231, 768
  %357 = inttoptr i32 %356 to i32 addrspace(131072)*
  %358 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %357)
  %359 = add i32 %231, 832
  %360 = inttoptr i32 %359 to i32 addrspace(131072)*
  %361 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %360)
  %362 = add i32 %231, 896
  %363 = inttoptr i32 %362 to i32 addrspace(131072)*
  %364 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %363)
  %365 = add i32 %231, 960
  %366 = inttoptr i32 %365 to i32 addrspace(131072)*
  %367 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)* %366)
  %368 = insertelement <8 x i32> undef, i32 %346, i32 0
  %369 = insertelement <8 x i32> %368, i32 %349, i32 1
  %370 = insertelement <8 x i32> %369, i32 %352, i32 2
  %371 = insertelement <8 x i32> %370, i32 %355, i32 3
  %372 = insertelement <8 x i32> %371, i32 %358, i32 4
  %373 = insertelement <8 x i32> %372, i32 %361, i32 5
  %374 = insertelement <8 x i32> %373, i32 %364, i32 6
  %375 = insertelement <8 x i32> %374, i32 %367, i32 7
  %376 = add i32 %238, 32
  %377 = add i32 %376, %274
  %378 = inttoptr i32 %377 to i32 addrspace(131073)*
  %379 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %378)
  %380 = add i32 %377, 64
  %381 = inttoptr i32 %380 to i32 addrspace(131073)*
  %382 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %381)
  %383 = add i32 %377, 128
  %384 = inttoptr i32 %383 to i32 addrspace(131073)*
  %385 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %384)
  %386 = add i32 %377, 192
  %387 = inttoptr i32 %386 to i32 addrspace(131073)*
  %388 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %387)
  %389 = add i32 %377, 256
  %390 = inttoptr i32 %389 to i32 addrspace(131073)*
  %391 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %390)
  %392 = add i32 %377, 320
  %393 = inttoptr i32 %392 to i32 addrspace(131073)*
  %394 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %393)
  %395 = add i32 %377, 384
  %396 = inttoptr i32 %395 to i32 addrspace(131073)*
  %397 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %396)
  %398 = add i32 %377, 448
  %399 = inttoptr i32 %398 to i32 addrspace(131073)*
  %400 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %399)
  %401 = insertelement <8 x i32> undef, i32 %379, i32 0
  %402 = insertelement <8 x i32> %401, i32 %382, i32 1
  %403 = insertelement <8 x i32> %402, i32 %385, i32 2
  %404 = insertelement <8 x i32> %403, i32 %388, i32 3
  %405 = insertelement <8 x i32> %404, i32 %391, i32 4
  %406 = insertelement <8 x i32> %405, i32 %394, i32 5
  %407 = insertelement <8 x i32> %406, i32 %397, i32 6
  %408 = insertelement <8 x i32> %407, i32 %400, i32 7
  %409 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %307, <8 x i32> %408, <8 x i32> %375, i32 9, i32 9, i32 8, i32 8, i1 false)
  %410 = add i32 %376, %310
  %411 = inttoptr i32 %410 to i32 addrspace(131073)*
  %412 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %411)
  %413 = add i32 %410, 64
  %414 = inttoptr i32 %413 to i32 addrspace(131073)*
  %415 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %414)
  %416 = add i32 %410, 128
  %417 = inttoptr i32 %416 to i32 addrspace(131073)*
  %418 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %417)
  %419 = add i32 %410, 192
  %420 = inttoptr i32 %419 to i32 addrspace(131073)*
  %421 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %420)
  %422 = add i32 %410, 256
  %423 = inttoptr i32 %422 to i32 addrspace(131073)*
  %424 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %423)
  %425 = add i32 %410, 320
  %426 = inttoptr i32 %425 to i32 addrspace(131073)*
  %427 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %426)
  %428 = add i32 %410, 384
  %429 = inttoptr i32 %428 to i32 addrspace(131073)*
  %430 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %429)
  %431 = add i32 %410, 448
  %432 = inttoptr i32 %431 to i32 addrspace(131073)*
  %433 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)* %432)
  %434 = insertelement <8 x i32> undef, i32 %412, i32 0
  %435 = insertelement <8 x i32> %434, i32 %415, i32 1
  %436 = insertelement <8 x i32> %435, i32 %418, i32 2
  %437 = insertelement <8 x i32> %436, i32 %421, i32 3
  %438 = insertelement <8 x i32> %437, i32 %424, i32 4
  %439 = insertelement <8 x i32> %438, i32 %427, i32 5
  %440 = insertelement <8 x i32> %439, i32 %430, i32 6
  %441 = insertelement <8 x i32> %440, i32 %433, i32 7
  %442 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %343, <8 x i32> %441, <8 x i32> %375, i32 9, i32 9, i32 8, i32 8, i1 false)
  %443 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %61, i32 %62, i32 %38, i32 %39)
  %444 = extractvalue { i32, i32 } %443, 0
  %445 = extractvalue { i32, i32 } %443, 1
  %446 = call { i32, i32 } @llvm.genx.GenISA.sub.pair(i32 0, i32 0, i32 %444, i32 %445)
  %447 = extractvalue { i32, i32 } %446, 0
  %448 = shl i32 %447, 2
  %449 = add i32 %66, %448
  %450 = shl i32 %83, 8
  %451 = lshr i32 %83, 24
  %452 = shl i32 %84, 8
  %453 = or i32 %452, %451
  %454 = shl i32 %228, 3
  %455 = shl i32 %454, 2
  %456 = add i32 %449, %455
  %457 = shl i32 %450, 2
  %458 = add i32 %456, %457
  %459 = extractelement <8 x float> %409, i32 0
  %460 = bitcast float %459 to i32
  %461 = inttoptr i32 %458 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %461, i32 %460)
  %462 = add i32 %458, 64
  %463 = extractelement <8 x float> %409, i32 1
  %464 = bitcast float %463 to i32
  %465 = inttoptr i32 %462 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %465, i32 %464)
  %466 = add i32 %458, 128
  %467 = extractelement <8 x float> %409, i32 2
  %468 = bitcast float %467 to i32
  %469 = inttoptr i32 %466 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %469, i32 %468)
  %470 = add i32 %458, 192
  %471 = extractelement <8 x float> %409, i32 3
  %472 = bitcast float %471 to i32
  %473 = inttoptr i32 %470 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %473, i32 %472)
  %474 = add i32 %458, 256
  %475 = extractelement <8 x float> %409, i32 4
  %476 = bitcast float %475 to i32
  %477 = inttoptr i32 %474 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %477, i32 %476)
  %478 = add i32 %458, 320
  %479 = extractelement <8 x float> %409, i32 5
  %480 = bitcast float %479 to i32
  %481 = inttoptr i32 %478 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %481, i32 %480)
  %482 = add i32 %458, 384
  %483 = extractelement <8 x float> %409, i32 6
  %484 = bitcast float %483 to i32
  %485 = inttoptr i32 %482 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %485, i32 %484)
  %486 = add i32 %458, 448
  %487 = extractelement <8 x float> %409, i32 7
  %488 = bitcast float %487 to i32
  %489 = inttoptr i32 %486 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %489, i32 %488)
  %490 = call { i32, i32 } @llvm.genx.GenISA.add.pair(i32 %450, i32 %453, i32 128, i32 0)
  %491 = extractvalue { i32, i32 } %490, 0
  %492 = shl i32 %491, 2
  %493 = add i32 %456, %492
  %494 = extractelement <8 x float> %442, i32 0
  %495 = bitcast float %494 to i32
  %496 = inttoptr i32 %493 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %496, i32 %495)
  %497 = add i32 %493, 64
  %498 = extractelement <8 x float> %442, i32 1
  %499 = bitcast float %498 to i32
  %500 = inttoptr i32 %497 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %500, i32 %499)
  %501 = add i32 %493, 128
  %502 = extractelement <8 x float> %442, i32 2
  %503 = bitcast float %502 to i32
  %504 = inttoptr i32 %501 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %504, i32 %503)
  %505 = add i32 %493, 192
  %506 = extractelement <8 x float> %442, i32 3
  %507 = bitcast float %506 to i32
  %508 = inttoptr i32 %505 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %508, i32 %507)
  %509 = add i32 %493, 256
  %510 = extractelement <8 x float> %442, i32 4
  %511 = bitcast float %510 to i32
  %512 = inttoptr i32 %509 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %512, i32 %511)
  %513 = add i32 %493, 320
  %514 = extractelement <8 x float> %442, i32 5
  %515 = bitcast float %514 to i32
  %516 = inttoptr i32 %513 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %516, i32 %515)
  %517 = add i32 %493, 384
  %518 = extractelement <8 x float> %442, i32 6
  %519 = bitcast float %518 to i32
  %520 = inttoptr i32 %517 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %520, i32 %519)
  %521 = add i32 %493, 448
  %522 = extractelement <8 x float> %442, i32 7
  %523 = bitcast float %522 to i32
  %524 = inttoptr i32 %521 to i32 addrspace(131074)*
  call void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)* %524, i32 %523)
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

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
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float>, <8 x i32>, <8 x i32>, i32, i32, i32, i32, i1) #2

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

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: convergent nounwind readnone
declare i32 @llvm.genx.GenISA.WaveShuffleIndex.i32.i32.i32(i32, i32, i32) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: convergent nounwind readnone
declare i64 @llvm.genx.GenISA.WaveShuffleIndex.i64.i32.i32(i64, i32, i32) #5

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
; Output: 
; Arg 0: 
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
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p131072i32(i32 addrspace(131072)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p131073i32(i32 addrspace(131073)*) #3

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind
declare void @llvm.genx.GenISA.simdBlockWrite.p131074i32.i32(i32 addrspace(131074)*, i32) #4

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
!opencl.ocl.version = !{!436, !436, !436, !436, !436, !436, !436}
!opencl.spir.version = !{!436, !436, !436, !436, !436}
!llvm.ident = !{!437, !437, !437, !437, !437, !438, !438, !439, !439}
!llvm.module.flags = !{!440}

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
!42 = !{!"ModuleMD", !43, !44, !121, !286, !317, !333, !354, !364, !366, !367, !381, !382, !383, !384, !388, !389, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !406, !408, !412, !413, !414, !415, !416, !417, !418, !419, !420, !421, !422, !207, !423, !426, !427, !429, !432, !433, !434}
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
!153 = !{!"uavsNumType", i32 7}
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
!404 = !{!"MinNOSPushConstantSize", i32 2}
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
!423 = !{!"PrivateMemoryPerFG", !424, !425}
!424 = !{!"PrivateMemoryPerFGMap[0]", void (%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE}
!425 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!426 = !{!"m_OptsToDisable"}
!427 = !{!"capabilities", !428}
!428 = !{!"globalVariableDecorationsINTEL", i1 false}
!429 = !{!"m_ShaderResourceViewMcsMask", !430, !431}
!430 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!431 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!432 = !{!"computedDepthMode", i32 0}
!433 = !{!"isHDCFastClearShader", i1 false}
!434 = !{!"argRegisterReservations", !435}
!435 = !{!"argRegisterReservationsVec[0]", i32 0}
!436 = !{i32 2, i32 0}
!437 = !{!"clang version 14.0.5"}
!438 = !{!"clang version 9.0.0 (2dbee6917918354f14026a637370445d50bdaf6a)"}
!439 = !{!"clang version 9.0.0 (c68f557a081b1b2339a42d7cd6af3c2ab18c6061)"}
!440 = !{i32 1, !"wchar_size", i32 4}
!441 = !{float 2.500000e+00}
