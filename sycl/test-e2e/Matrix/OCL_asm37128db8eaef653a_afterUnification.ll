; ------------------------------------------------
; OCL_asm37128db8eaef653a_afterUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved" = type { <8 x i32> }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved" = type { <8 x float> }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm8ELm8ELm16EE(%"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, float addrspace(1)* align 4 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
  %11 = extractelement <3 x i32> %localSize, i32 0
  %12 = extractelement <3 x i32> %localSize, i32 1
  %13 = extractelement <3 x i32> %localSize, i32 2
  %14 = extractelement <8 x i32> %payloadHeader, i32 0
  %15 = extractelement <8 x i32> %payloadHeader, i32 1
  %16 = extractelement <8 x i32> %payloadHeader, i32 2
  %17 = extractelement <8 x i32> %payloadHeader, i32 3
  %18 = extractelement <8 x i32> %payloadHeader, i32 4
  %19 = extractelement <8 x i32> %payloadHeader, i32 5
  %20 = extractelement <8 x i32> %payloadHeader, i32 6
  %21 = extractelement <8 x i32> %payloadHeader, i32 7
  %22 = extractelement <3 x i32> %enqueuedLocalSize, i32 0
  %23 = extractelement <3 x i32> %enqueuedLocalSize, i32 1
  %24 = extractelement <3 x i32> %enqueuedLocalSize, i32 2
  %25 = extractelement <8 x i32> %r0, i32 0
  %26 = extractelement <8 x i32> %r0, i32 1
  %27 = extractelement <8 x i32> %r0, i32 2
  %28 = extractelement <8 x i32> %r0, i32 3
  %29 = extractelement <8 x i32> %r0, i32 4
  %30 = extractelement <8 x i32> %r0, i32 5
  %31 = extractelement <8 x i32> %r0, i32 6
  %32 = extractelement <8 x i32> %r0, i32 7
  %33 = alloca [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"], align 8, !spirv.Decorations !434
  %34 = alloca [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], align 8, !spirv.Decorations !434
  %35 = mul i64 %const_reg_qword2, %const_reg_qword1
  %36 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %0, i64 %35
  %37 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %36, i64 %const_reg_qword3
  %38 = mul i64 %const_reg_qword6, %const_reg_qword5
  %39 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %4, i64 %38
  %40 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %39, i64 %const_reg_qword7
  %41 = mul i64 %const_reg_qword10, %const_reg_qword9
  %42 = getelementptr float, float addrspace(1)* %7, i64 %41
  %43 = getelementptr float, float addrspace(1)* %42, i64 %const_reg_qword11
  %44 = mul i32 %23, %31
  %localIdY15 = zext i16 %localIdY to i32
  %45 = add i32 %44, %localIdY15
  %46 = add i32 %45, %15
  %47 = zext i32 %46 to i64
  %48 = icmp sgt i32 %46, -1
  call void @llvm.assume(i1 %48)
  %49 = mul i32 %22, %26
  %localIdX21 = zext i16 %localIdX to i32
  %50 = add i32 %49, %localIdX21
  %51 = add i32 %50, %14
  %52 = zext i32 %51 to i64
  %53 = icmp sgt i32 %51, -1
  call void @llvm.assume(i1 %53)
  %54 = zext i16 %localIdY to i64
  %55 = sub nsw i64 %47, %54, !spirv.Decorations !436
  %56 = zext i16 %localIdX to i64
  %57 = sub nsw i64 %52, %56, !spirv.Decorations !436
  %58 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"]* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %58)
  %59 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"]* %33, i64 0, i64 0
  br label %60

60:                                               ; preds = %60, %10
  %61 = phi %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"* [ %59, %10 ], [ %62, %60 ]
  %62 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"* %61, i64 1
  %63 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"]* %33, i64 1, i64 0
  %64 = icmp eq %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"* %62, %63
  br i1 %64, label %65, label %60

65:                                               ; preds = %60
  %66 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %66)
  %67 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34, i64 0, i64 0
  br label %68

68:                                               ; preds = %68, %65
  %69 = phi %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"* [ %67, %65 ], [ %70, %68 ]
  %70 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"* %69, i64 1
  %71 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34, i64 1, i64 0
  %72 = icmp eq %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"* %70, %71
  br i1 %72, label %.preheader, label %68

.preheader:                                       ; preds = %68
  br label %73

73:                                               ; preds = %87, %.preheader
  %74 = phi i32 [ %90, %87 ], [ 0, %.preheader ]
  %75 = icmp ult i32 %74, 2
  br i1 %75, label %87, label %76

76:                                               ; preds = %73
  %77 = add i64 %35, %const_reg_qword3
  %78 = sub i64 0, %77
  %79 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %37, i64 %78
  %80 = udiv i64 %57, %3
  %freeze = freeze i64 %80
  %81 = shl i64 %freeze, 4
  %82 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %79, i64 %81
  %83 = add i64 %38, %const_reg_qword7
  %84 = sub i64 0, %83
  %85 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %40, i64 %84
  %86 = shl nsw i64 %55, 9, !spirv.Decorations !436
  br label %91

87:                                               ; preds = %73
  %88 = zext i32 %74 to i64
  %89 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34, i64 0, i64 %88, i32 0
  store <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x float>* %89, align 8
  %90 = add nuw nsw i32 %74, 1, !spirv.Decorations !438
  br label %73

91:                                               ; preds = %159, %76
  %92 = phi i32 [ 0, %76 ], [ %160, %159 ]
  %93 = icmp ult i32 %92, 2
  br i1 %93, label %102, label %94

94:                                               ; preds = %91
  %95 = add i64 %41, %const_reg_qword11
  %96 = sub i64 0, %95
  %97 = getelementptr inbounds float, float addrspace(1)* %43, i64 %96
  %98 = shl nsw i64 %55, 8, !spirv.Decorations !436
  %99 = udiv i64 %57, %3
  %freeze63 = freeze i64 %99
  %100 = shl i64 %freeze63, 3
  %101 = getelementptr float, float addrspace(1)* %97, i64 %100
  br label %221

102:                                              ; preds = %91
  %103 = shl nuw nsw i32 %92, 4, !spirv.Decorations !438
  %104 = zext i32 %103 to i64
  %105 = shl nuw nsw i64 %104, 4, !spirv.Decorations !438
  %106 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %82, i64 %105
  %107 = mul i32 %12, %11
  %108 = mul i32 %107, %13
  %109 = and i32 %108, 7
  %110 = icmp eq i32 %109, 0
  br i1 %110, label %123, label %111

111:                                              ; preds = %102
  %localIdZ28 = zext i16 %localIdZ to i32
  %112 = mul i32 %12, %localIdZ28
  %localIdY30 = zext i16 %localIdY to i32
  %113 = add i32 %112, %localIdY30
  %114 = mul i32 %113, %11
  %localIdX32 = zext i16 %localIdX to i32
  %115 = add i32 %114, %localIdX32
  %116 = lshr i32 %115, 3
  %117 = mul i32 %12, %11
  %118 = mul i32 %117, %13
  %119 = add i32 %118, 7
  %120 = lshr i32 %119, 3
  %121 = add nsw i32 %120, -1
  %122 = icmp ult i32 %116, %121
  br i1 %122, label %123, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x8_i16_8_global_v8i8_pi32_i32.22.exit

123:                                              ; preds = %111, %102
  br label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x8_i16_8_global_v8i8_pi32_i32.22.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x8_i16_8_global_v8i8_pi32_i32.22.exit: ; preds = %111, %123
  %124 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106 to i32 addrspace(1)*
  %125 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %124)
  %126 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 32
  %127 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %126 to i32 addrspace(1)*
  %128 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %127)
  %129 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 64
  %130 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %129 to i32 addrspace(1)*
  %131 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %130)
  %132 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 96
  %133 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %132 to i32 addrspace(1)*
  %134 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %133)
  %135 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 128
  %136 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %135 to i32 addrspace(1)*
  %137 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %136)
  %138 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 160
  %139 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %138 to i32 addrspace(1)*
  %140 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %139)
  %141 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 192
  %142 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %141 to i32 addrspace(1)*
  %143 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %142)
  %144 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %106, i64 224
  %145 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %144 to i32 addrspace(1)*
  %146 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %145)
  %147 = insertelement <8 x i32> undef, i32 %125, i32 0
  %148 = insertelement <8 x i32> %147, i32 %128, i32 1
  %149 = insertelement <8 x i32> %148, i32 %131, i32 2
  %150 = insertelement <8 x i32> %149, i32 %134, i32 3
  %151 = insertelement <8 x i32> %150, i32 %137, i32 4
  %152 = insertelement <8 x i32> %151, i32 %140, i32 5
  %153 = insertelement <8 x i32> %152, i32 %143, i32 6
  %154 = insertelement <8 x i32> %153, i32 %146, i32 7
  %155 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %85, i64 %104
  br label %156

156:                                              ; preds = %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_8x16_i16_8_global_v8i8_pi32_i32.23.exit, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x8_i16_8_global_v8i8_pi32_i32.22.exit
  %157 = phi i32 [ 0, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedB_PackedB_16x8_i16_8_global_v8i8_pi32_i32.22.exit ], [ %220, %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_8x16_i16_8_global_v8i8_pi32_i32.23.exit ]
  %158 = icmp ult i32 %157, 2
  br i1 %158, label %161, label %159

159:                                              ; preds = %156
  %160 = add nuw nsw i32 %92, 1, !spirv.Decorations !438
  br label %91

161:                                              ; preds = %156
  %162 = zext i32 %157 to i64
  %163 = shl nuw nsw i64 %162, 8, !spirv.Decorations !438
  %164 = or i64 %86, %163
  %165 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %155, i64 %164
  %166 = mul i32 %12, %11
  %167 = mul i32 %166, %13
  %168 = and i32 %167, 7
  %169 = icmp eq i32 %168, 0
  br i1 %169, label %182, label %170

170:                                              ; preds = %161
  %localIdZ39 = zext i16 %localIdZ to i32
  %171 = mul i32 %12, %localIdZ39
  %localIdY41 = zext i16 %localIdY to i32
  %172 = add i32 %171, %localIdY41
  %173 = mul i32 %172, %11
  %localIdX43 = zext i16 %localIdX to i32
  %174 = add i32 %173, %localIdX43
  %175 = lshr i32 %174, 3
  %176 = mul i32 %12, %11
  %177 = mul i32 %176, %13
  %178 = add i32 %177, 7
  %179 = lshr i32 %178, 3
  %180 = add nsw i32 %179, -1
  %181 = icmp ult i32 %175, %180
  br i1 %181, label %182, label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_8x16_i16_8_global_v8i8_pi32_i32.23.exit

182:                                              ; preds = %170, %161
  br label %__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_8x16_i16_8_global_v8i8_pi32_i32.23.exit

__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_8x16_i16_8_global_v8i8_pi32_i32.23.exit: ; preds = %170, %182
  %183 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165 to i32 addrspace(1)*
  %184 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %183)
  %185 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 32
  %186 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %185 to i32 addrspace(1)*
  %187 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %186)
  %188 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 64
  %189 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %188 to i32 addrspace(1)*
  %190 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %189)
  %191 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 96
  %192 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %191 to i32 addrspace(1)*
  %193 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %192)
  %194 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 128
  %195 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %194 to i32 addrspace(1)*
  %196 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %195)
  %197 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 160
  %198 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %197 to i32 addrspace(1)*
  %199 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %198)
  %200 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 192
  %201 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %200 to i32 addrspace(1)*
  %202 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %201)
  %203 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %165, i64 224
  %204 = bitcast %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* %203 to i32 addrspace(1)*
  %205 = call i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)* %204)
  %206 = insertelement <8 x i32> undef, i32 %184, i32 0
  %207 = insertelement <8 x i32> %206, i32 %187, i32 1
  %208 = insertelement <8 x i32> %207, i32 %190, i32 2
  %209 = insertelement <8 x i32> %208, i32 %193, i32 3
  %210 = insertelement <8 x i32> %209, i32 %196, i32 4
  %211 = insertelement <8 x i32> %210, i32 %199, i32 5
  %212 = insertelement <8 x i32> %211, i32 %202, i32 6
  %213 = insertelement <8 x i32> %212, i32 %205, i32 7
  %214 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"]* %33, i64 0, i64 %162, i32 0
  store <8 x i32> %213, <8 x i32>* %214, align 8
  %215 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34, i64 0, i64 %162
  %216 = getelementptr %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"* %215, i64 0, i32 0
  %217 = load <8 x float>, <8 x float>* %216, align 8
  %218 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i32.v8i32(<8 x float> %217, <8 x i32> %213, <8 x i32> %154, i32 9, i32 9, i32 8, i32 8, i1 false)
  %219 = getelementptr %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved", %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"* %215, i64 0, i32 0
  store <8 x float> %218, <8 x float>* %219, align 8
  %220 = add nuw nsw i32 %157, 1, !spirv.Decorations !438
  br label %156

221:                                              ; preds = %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_8x8_i32_8_global_pi64_v8i8.24.exit, %94
  %222 = phi i32 [ %271, %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_8x8_i32_8_global_pi64_v8i8.24.exit ], [ 0, %94 ]
  %223 = icmp ult i32 %222, 2
  br i1 %223, label %224, label %272

224:                                              ; preds = %221
  %225 = zext i32 %222 to i64
  %226 = shl nuw nsw i64 %225, 7, !spirv.Decorations !438
  %227 = or i64 %98, %226
  %228 = getelementptr float, float addrspace(1)* %101, i64 %227
  %229 = getelementptr inbounds [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"], [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34, i64 0, i64 %225, i32 0
  %230 = load <8 x float>, <8 x float>* %229, align 8
  %231 = extractelement <8 x float> %230, i32 0
  %232 = extractelement <8 x float> %230, i32 1
  %233 = extractelement <8 x float> %230, i32 2
  %234 = extractelement <8 x float> %230, i32 3
  %235 = extractelement <8 x float> %230, i32 4
  %236 = extractelement <8 x float> %230, i32 5
  %237 = extractelement <8 x float> %230, i32 6
  %238 = extractelement <8 x float> %230, i32 7
  %239 = mul i32 %12, %11
  %240 = mul i32 %239, %13
  %241 = and i32 %240, 7
  %242 = icmp eq i32 %241, 0
  br i1 %242, label %255, label %243

243:                                              ; preds = %224
  %localIdZ50 = zext i16 %localIdZ to i32
  %244 = mul i32 %12, %localIdZ50
  %localIdY52 = zext i16 %localIdY to i32
  %245 = add i32 %244, %localIdY52
  %246 = mul i32 %245, %11
  %localIdX54 = zext i16 %localIdX to i32
  %247 = add i32 %246, %localIdX54
  %248 = lshr i32 %247, 3
  %249 = mul i32 %12, %11
  %250 = mul i32 %249, %13
  %251 = add i32 %250, 7
  %252 = lshr i32 %251, 3
  %253 = add nsw i32 %252, -1
  %254 = icmp ult i32 %248, %253
  br i1 %254, label %255, label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_8x8_i32_8_global_pi64_v8i8.24.exit

255:                                              ; preds = %243, %224
  br label %__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_8x8_i32_8_global_pi64_v8i8.24.exit

__builtin_spriv_OpJointMatrixStoreINTEL_Accumulator_RowMajor_8x8_i32_8_global_pi64_v8i8.24.exit: ; preds = %243, %255
  %256 = bitcast float addrspace(1)* %228 to i32 addrspace(1)*
  %bc71 = bitcast float %231 to i32
  %bc72 = bitcast float %232 to i32
  %bc73 = bitcast float %233 to i32
  %bc74 = bitcast float %234 to i32
  %bc75 = bitcast float %235 to i32
  %bc76 = bitcast float %236 to i32
  %bc77 = bitcast float %237 to i32
  %bc78 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %256, i32 %bc71)
  %257 = getelementptr inbounds float, float addrspace(1)* %228, i64 16
  %258 = bitcast float addrspace(1)* %257 to i32 addrspace(1)*
  %bc6479 = bitcast float %231 to i32
  %bc6480 = bitcast float %232 to i32
  %bc6481 = bitcast float %233 to i32
  %bc6482 = bitcast float %234 to i32
  %bc6483 = bitcast float %235 to i32
  %bc6484 = bitcast float %236 to i32
  %bc6485 = bitcast float %237 to i32
  %bc6486 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %258, i32 %bc6480)
  %259 = getelementptr inbounds float, float addrspace(1)* %228, i64 32
  %260 = bitcast float addrspace(1)* %259 to i32 addrspace(1)*
  %bc6587 = bitcast float %231 to i32
  %bc6588 = bitcast float %232 to i32
  %bc6589 = bitcast float %233 to i32
  %bc6590 = bitcast float %234 to i32
  %bc6591 = bitcast float %235 to i32
  %bc6592 = bitcast float %236 to i32
  %bc6593 = bitcast float %237 to i32
  %bc6594 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %260, i32 %bc6589)
  %261 = getelementptr inbounds float, float addrspace(1)* %228, i64 48
  %262 = bitcast float addrspace(1)* %261 to i32 addrspace(1)*
  %bc6695 = bitcast float %231 to i32
  %bc6696 = bitcast float %232 to i32
  %bc6697 = bitcast float %233 to i32
  %bc6698 = bitcast float %234 to i32
  %bc6699 = bitcast float %235 to i32
  %bc66100 = bitcast float %236 to i32
  %bc66101 = bitcast float %237 to i32
  %bc66102 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %262, i32 %bc6698)
  %263 = getelementptr inbounds float, float addrspace(1)* %228, i64 64
  %264 = bitcast float addrspace(1)* %263 to i32 addrspace(1)*
  %bc67103 = bitcast float %231 to i32
  %bc67104 = bitcast float %232 to i32
  %bc67105 = bitcast float %233 to i32
  %bc67106 = bitcast float %234 to i32
  %bc67107 = bitcast float %235 to i32
  %bc67108 = bitcast float %236 to i32
  %bc67109 = bitcast float %237 to i32
  %bc67110 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %264, i32 %bc67107)
  %265 = getelementptr inbounds float, float addrspace(1)* %228, i64 80
  %266 = bitcast float addrspace(1)* %265 to i32 addrspace(1)*
  %bc68111 = bitcast float %231 to i32
  %bc68112 = bitcast float %232 to i32
  %bc68113 = bitcast float %233 to i32
  %bc68114 = bitcast float %234 to i32
  %bc68115 = bitcast float %235 to i32
  %bc68116 = bitcast float %236 to i32
  %bc68117 = bitcast float %237 to i32
  %bc68118 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %266, i32 %bc68116)
  %267 = getelementptr inbounds float, float addrspace(1)* %228, i64 96
  %268 = bitcast float addrspace(1)* %267 to i32 addrspace(1)*
  %bc69119 = bitcast float %231 to i32
  %bc69120 = bitcast float %232 to i32
  %bc69121 = bitcast float %233 to i32
  %bc69122 = bitcast float %234 to i32
  %bc69123 = bitcast float %235 to i32
  %bc69124 = bitcast float %236 to i32
  %bc69125 = bitcast float %237 to i32
  %bc69126 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %268, i32 %bc69125)
  %269 = getelementptr inbounds float, float addrspace(1)* %228, i64 112
  %270 = bitcast float addrspace(1)* %269 to i32 addrspace(1)*
  %bc70127 = bitcast float %231 to i32
  %bc70128 = bitcast float %232 to i32
  %bc70129 = bitcast float %233 to i32
  %bc70130 = bitcast float %234 to i32
  %bc70131 = bitcast float %235 to i32
  %bc70132 = bitcast float %236 to i32
  %bc70133 = bitcast float %237 to i32
  %bc70134 = bitcast float %238 to i32
  call void @llvm.genx.GenISA.simdBlockWrite.p1i32.i32(i32 addrspace(1)* %270, i32 %bc70134)
  %271 = add nuw nsw i32 %222, 1, !spirv.Decorations !438
  br label %221

272:                                              ; preds = %221
  %273 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.20.resolved"]* %34 to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %273)
  %274 = bitcast [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix.10.resolved"]* %33 to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %274)
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

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!431, !431, !431, !431, !431}
!opencl.spir.version = !{!431, !431, !431, !431, !431}
!llvm.ident = !{!432, !432, !432, !432, !432}
!llvm.module.flags = !{!433}

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
!42 = !{!"ModuleMD", !43, !44, !121, !283, !314, !330, !351, !361, !363, !364, !378, !379, !380, !381, !385, !386, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !403, !405, !409, !410, !411, !412, !413, !414, !415, !416, !417, !418, !419, !207, !420, !421, !422, !424, !427, !428, !429}
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
!282 = !{!"m_OptsToDisablePerFunc"}
!283 = !{!"pushInfo", !284, !285, !286, !290, !291, !292, !293, !294, !295, !296, !297, !310, !311, !312, !313}
!284 = !{!"pushableAddresses"}
!285 = !{!"bindlessPushInfo"}
!286 = !{!"dynamicBufferInfo", !287, !288, !289}
!287 = !{!"firstIndex", i32 0}
!288 = !{!"numOffsets", i32 0}
!289 = !{!"forceDisabled", i1 false}
!290 = !{!"MaxNumberOfPushedBuffers", i32 0}
!291 = !{!"inlineConstantBufferSlot", i32 -1}
!292 = !{!"inlineConstantBufferOffset", i32 -1}
!293 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!294 = !{!"constants"}
!295 = !{!"inputs"}
!296 = !{!"constantReg"}
!297 = !{!"simplePushInfoArr", !298, !307, !308, !309}
!298 = !{!"simplePushInfoArrVec[0]", !299, !300, !301, !302, !303, !304, !305, !306}
!299 = !{!"cbIdx", i32 0}
!300 = !{!"pushableAddressGrfOffset", i32 -1}
!301 = !{!"pushableOffsetGrfOffset", i32 -1}
!302 = !{!"offset", i32 0}
!303 = !{!"size", i32 0}
!304 = !{!"isStateless", i1 false}
!305 = !{!"isBindless", i1 false}
!306 = !{!"simplePushLoads"}
!307 = !{!"simplePushInfoArrVec[1]", !299, !300, !301, !302, !303, !304, !305, !306}
!308 = !{!"simplePushInfoArrVec[2]", !299, !300, !301, !302, !303, !304, !305, !306}
!309 = !{!"simplePushInfoArrVec[3]", !299, !300, !301, !302, !303, !304, !305, !306}
!310 = !{!"simplePushBufferUsed", i32 0}
!311 = !{!"pushAnalysisWIInfos"}
!312 = !{!"inlineRTGlobalPtrOffset", i32 0}
!313 = !{!"rtSyncSurfPtrOffset", i32 0}
!314 = !{!"psInfo", !315, !316, !317, !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !329}
!315 = !{!"BlendStateDisabledMask", i8 0}
!316 = !{!"SkipSrc0Alpha", i1 false}
!317 = !{!"DualSourceBlendingDisabled", i1 false}
!318 = !{!"ForceEnableSimd32", i1 false}
!319 = !{!"outputDepth", i1 false}
!320 = !{!"outputStencil", i1 false}
!321 = !{!"outputMask", i1 false}
!322 = !{!"blendToFillEnabled", i1 false}
!323 = !{!"forceEarlyZ", i1 false}
!324 = !{!"hasVersionedLoop", i1 false}
!325 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!326 = !{!"NumSamples", i8 0}
!327 = !{!"blendOptimizationMode"}
!328 = !{!"colorOutputMask"}
!329 = !{!"WaDisableVRS", i1 false}
!330 = !{!"csInfo", !331, !332, !333, !334, !335, !56, !57, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !87, !88, !348, !349, !350}
!331 = !{!"maxWorkGroupSize", i32 0}
!332 = !{!"waveSize", i32 0}
!333 = !{!"ComputeShaderSecondCompile"}
!334 = !{!"forcedSIMDSize", i8 0}
!335 = !{!"forceTotalGRFNum", i32 0}
!336 = !{!"forceSpillCompression", i1 false}
!337 = !{!"allowLowerSimd", i1 false}
!338 = !{!"disableSimd32Slicing", i1 false}
!339 = !{!"disableSplitOnSpill", i1 false}
!340 = !{!"enableNewSpillCostFunction", i1 false}
!341 = !{!"forcedVISAPreRAScheduler", i1 false}
!342 = !{!"forceUniformBuffer", i1 false}
!343 = !{!"forceUniformSurfaceSampler", i1 false}
!344 = !{!"disableLocalIdOrderOptimizations", i1 false}
!345 = !{!"disableDispatchAlongY", i1 false}
!346 = !{!"neededThreadIdLayout", i1* null}
!347 = !{!"forceTileYWalk", i1 false}
!348 = !{!"walkOrderEnabled", i1 false}
!349 = !{!"walkOrderOverride", i32 0}
!350 = !{!"ResForHfPacking"}
!351 = !{!"msInfo", !352, !353, !354, !355, !356, !357, !358, !359, !360}
!352 = !{!"PrimitiveTopology", i32 3}
!353 = !{!"MaxNumOfPrimitives", i32 0}
!354 = !{!"MaxNumOfVertices", i32 0}
!355 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!356 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!357 = !{!"WorkGroupSize", i32 0}
!358 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!359 = !{!"IndexFormat", i32 6}
!360 = !{!"SubgroupSize", i32 0}
!361 = !{!"taskInfo", !362, !357, !358, !360}
!362 = !{!"MaxNumOfOutputs", i32 0}
!363 = !{!"NBarrierCnt", i32 0}
!364 = !{!"rtInfo", !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377}
!365 = !{!"RayQueryAllocSizeInBytes", i32 0}
!366 = !{!"NumContinuations", i32 0}
!367 = !{!"RTAsyncStackAddrspace", i32 -1}
!368 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!369 = !{!"SWHotZoneAddrspace", i32 -1}
!370 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!371 = !{!"SWStackAddrspace", i32 -1}
!372 = !{!"SWStackSurfaceStateOffset", i1* null}
!373 = !{!"RTSyncStackAddrspace", i32 -1}
!374 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!375 = !{!"doSyncDispatchRays", i1 false}
!376 = !{!"MemStyle", !"Xe"}
!377 = !{!"GlobalDataStyle", !"Xe"}
!378 = !{!"CurUniqueIndirectIdx", i32 0}
!379 = !{!"inlineDynTextures"}
!380 = !{!"inlineResInfoData"}
!381 = !{!"immConstant", !382, !383, !384}
!382 = !{!"data"}
!383 = !{!"sizes"}
!384 = !{!"zeroIdxs"}
!385 = !{!"stringConstants"}
!386 = !{!"inlineBuffers", !387, !391, !392}
!387 = !{!"inlineBuffersVec[0]", !388, !389, !390}
!388 = !{!"alignment", i32 0}
!389 = !{!"allocSize", i64 0}
!390 = !{!"Buffer"}
!391 = !{!"inlineBuffersVec[1]", !388, !389, !390}
!392 = !{!"inlineBuffersVec[2]", !388, !389, !390}
!393 = !{!"GlobalPointerProgramBinaryInfos"}
!394 = !{!"ConstantPointerProgramBinaryInfos"}
!395 = !{!"GlobalBufferAddressRelocInfo"}
!396 = !{!"ConstantBufferAddressRelocInfo"}
!397 = !{!"forceLscCacheList"}
!398 = !{!"SrvMap"}
!399 = !{!"RasterizerOrderedByteAddressBuffer"}
!400 = !{!"RasterizerOrderedViews"}
!401 = !{!"MinNOSPushConstantSize", i32 0}
!402 = !{!"inlineProgramScopeOffsets"}
!403 = !{!"shaderData", !404}
!404 = !{!"numReplicas", i32 0}
!405 = !{!"URBInfo", !406, !407, !408}
!406 = !{!"has64BVertexHeaderInput", i1 false}
!407 = !{!"has64BVertexHeaderOutput", i1 false}
!408 = !{!"hasVertexHeader", i1 true}
!409 = !{!"UseBindlessImage", i1 false}
!410 = !{!"enableRangeReduce", i1 false}
!411 = !{!"allowMatchMadOptimizationforVS", i1 false}
!412 = !{!"disableMatchMadOptimizationForCS", i1 false}
!413 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!414 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!415 = !{!"statefulResourcesNotAliased", i1 false}
!416 = !{!"disableMixMode", i1 false}
!417 = !{!"genericAccessesResolved", i1 false}
!418 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!419 = !{!"disableSeparateScratchWA", i1 false}
!420 = !{!"PrivateMemoryPerFG"}
!421 = !{!"m_OptsToDisable"}
!422 = !{!"capabilities", !423}
!423 = !{!"globalVariableDecorationsINTEL", i1 false}
!424 = !{!"m_ShaderResourceViewMcsMask", !425, !426}
!425 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!426 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!427 = !{!"computedDepthMode", i32 0}
!428 = !{!"isHDCFastClearShader", i1 false}
!429 = !{!"argRegisterReservations", !430}
!430 = !{!"argRegisterReservationsVec[0]", i32 0}
!431 = !{i32 2, i32 0}
!432 = !{!"clang version 14.0.5"}
!433 = !{i32 1, !"wchar_size", i32 4}
!434 = !{!435}
!435 = !{i32 44, i32 8}
!436 = !{!437}
!437 = !{i32 4469}
!438 = !{!437, !439}
!439 = !{i32 4470}
