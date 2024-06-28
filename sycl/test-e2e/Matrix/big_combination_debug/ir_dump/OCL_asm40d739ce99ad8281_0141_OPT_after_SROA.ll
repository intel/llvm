; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0141_OPT_after_SROA.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTS7imatrixIfLm32ELm32ELm16EE(float addrspace(1)* align 4 %0, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %1, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %2, i64 %3, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %4, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %5, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %6, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)* align 2 %7, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %8, %"class.sycl::_V1::range"* byval(%"class.sycl::_V1::range") align 8 %9, <8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %localSize, <3 x i32> %enqueuedLocalSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i64 %const_reg_qword, i64 %const_reg_qword1, i64 %const_reg_qword2, i64 %const_reg_qword3, i64 %const_reg_qword4, i64 %const_reg_qword5, i64 %const_reg_qword6, i64 %const_reg_qword7, i64 %const_reg_qword8, i64 %const_reg_qword9, i64 %const_reg_qword10, i64 %const_reg_qword11, i32 %bufferOffset, i32 %bufferOffset12, i32 %bufferOffset13) #0 {
_Z18get_sub_group_sizev.exit.i4:
  %payloadHeader.scalar = extractelement <8 x i32> %payloadHeader, i64 0
  %payloadHeader.scalar87 = extractelement <8 x i32> %payloadHeader, i64 1
  %enqueuedLocalSize.scalar = extractelement <3 x i32> %enqueuedLocalSize, i64 0
  %enqueuedLocalSize.scalar85 = extractelement <3 x i32> %enqueuedLocalSize, i64 1
  %r0.scalar78 = extractelement <8 x i32> %r0, i64 1
  %r0.scalar83 = extractelement <8 x i32> %r0, i64 6
  %10 = mul i64 %const_reg_qword2, %const_reg_qword1
  %11 = getelementptr float, float addrspace(1)* %0, i64 %10
  %12 = getelementptr float, float addrspace(1)* %11, i64 %const_reg_qword3
  %13 = mul i32 %enqueuedLocalSize.scalar85, %r0.scalar83
  %localIdY15 = zext i16 %localIdY to i32
  %14 = add i32 %13, %localIdY15
  %15 = add i32 %14, %payloadHeader.scalar87
  %16 = zext i32 %15 to i64
  %17 = icmp sgt i32 %15, -1
  call void @llvm.assume(i1 %17)
  %18 = mul i32 %enqueuedLocalSize.scalar, %r0.scalar78
  %localIdX21 = zext i16 %localIdX to i32
  %19 = add i32 %18, %localIdX21
  %20 = add i32 %19, %payloadHeader.scalar
  %21 = zext i32 %20 to i64
  %22 = icmp sgt i32 %20, -1
  call void @llvm.assume(i1 %22)
  %23 = zext i16 %localIdY to i64
  %24 = sub nsw i64 %16, %23, !spirv.Decorations !519
  %25 = zext i16 %localIdX to i64
  %26 = sub nsw i64 %21, %25, !spirv.Decorations !519
  %27 = add i64 %10, %const_reg_qword3
  %28 = sub i64 0, %27
  %29 = getelementptr inbounds float, float addrspace(1)* %12, i64 %28
  %30 = shl nsw i64 %24, 11, !spirv.Decorations !519
  %31 = getelementptr inbounds float, float addrspace(1)* %29, i64 %30
  %32 = udiv i64 %26, %3
  %freeze = freeze i64 %32
  %33 = shl i64 %freeze, 5
  %34 = getelementptr inbounds float, float addrspace(1)* %31, i64 %33
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %simdLaneId16.fr = freeze i16 %simdLaneId16
  %simdLaneId = zext i16 %simdLaneId16.fr to i32
  %35 = and i32 %simdLaneId, 31
  %36 = icmp ult i16 %simdLaneId16.fr, 1024
  %37 = shl nuw nsw i32 %simdLaneId, 1
  %38 = and i32 %37, 131008
  %39 = or i32 %38, %35
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, float addrspace(1)* %34, i64 %40
  %42 = bitcast float addrspace(1)* %41 to i32 addrspace(1)*
  br label %43

43:                                               ; preds = %_Z18get_sub_group_sizev.exit.i4
  br i1 %36, label %44, label %._crit_edge

._crit_edge:                                      ; preds = %43
  br label %._crit_edge96

44:                                               ; preds = %43
  %45 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96

._crit_edge96:                                    ; preds = %._crit_edge, %44
  %46 = phi i32 [ %45, %44 ], [ 0, %._crit_edge ]
  %47 = bitcast i32 %46 to float
  %.sroa.0.0.vec.insert = insertelement <64 x float> undef, float %47, i32 0
  br i1 %36, label %48, label %._crit_edge.1

._crit_edge.1:                                    ; preds = %._crit_edge96
  br label %._crit_edge96.1

48:                                               ; preds = %._crit_edge96
  %49 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.1

._crit_edge96.1:                                  ; preds = %48, %._crit_edge.1
  %50 = phi i32 [ %49, %48 ], [ 0, %._crit_edge.1 ]
  %51 = bitcast i32 %50 to float
  %.sroa.0.4.vec.insert = insertelement <64 x float> %.sroa.0.0.vec.insert, float %51, i32 1
  br i1 %36, label %52, label %._crit_edge.2

._crit_edge.2:                                    ; preds = %._crit_edge96.1
  br label %._crit_edge96.2

52:                                               ; preds = %._crit_edge96.1
  %53 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.2

._crit_edge96.2:                                  ; preds = %52, %._crit_edge.2
  %54 = phi i32 [ %53, %52 ], [ 0, %._crit_edge.2 ]
  %55 = bitcast i32 %54 to float
  %.sroa.0.8.vec.insert = insertelement <64 x float> %.sroa.0.4.vec.insert, float %55, i32 2
  br i1 %36, label %56, label %._crit_edge.3

._crit_edge.3:                                    ; preds = %._crit_edge96.2
  br label %._crit_edge96.3

56:                                               ; preds = %._crit_edge96.2
  %57 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.3

._crit_edge96.3:                                  ; preds = %56, %._crit_edge.3
  %58 = phi i32 [ %57, %56 ], [ 0, %._crit_edge.3 ]
  %59 = bitcast i32 %58 to float
  %.sroa.0.12.vec.insert = insertelement <64 x float> %.sroa.0.8.vec.insert, float %59, i32 3
  br i1 %36, label %60, label %._crit_edge.4

._crit_edge.4:                                    ; preds = %._crit_edge96.3
  br label %._crit_edge96.4

60:                                               ; preds = %._crit_edge96.3
  %61 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.4

._crit_edge96.4:                                  ; preds = %60, %._crit_edge.4
  %62 = phi i32 [ %61, %60 ], [ 0, %._crit_edge.4 ]
  %63 = bitcast i32 %62 to float
  %.sroa.0.16.vec.insert = insertelement <64 x float> %.sroa.0.12.vec.insert, float %63, i32 4
  br i1 %36, label %64, label %._crit_edge.5

._crit_edge.5:                                    ; preds = %._crit_edge96.4
  br label %._crit_edge96.5

64:                                               ; preds = %._crit_edge96.4
  %65 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.5

._crit_edge96.5:                                  ; preds = %64, %._crit_edge.5
  %66 = phi i32 [ %65, %64 ], [ 0, %._crit_edge.5 ]
  %67 = bitcast i32 %66 to float
  %.sroa.0.20.vec.insert = insertelement <64 x float> %.sroa.0.16.vec.insert, float %67, i32 5
  br i1 %36, label %68, label %._crit_edge.6

._crit_edge.6:                                    ; preds = %._crit_edge96.5
  br label %._crit_edge96.6

68:                                               ; preds = %._crit_edge96.5
  %69 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.6

._crit_edge96.6:                                  ; preds = %68, %._crit_edge.6
  %70 = phi i32 [ %69, %68 ], [ 0, %._crit_edge.6 ]
  %71 = bitcast i32 %70 to float
  %.sroa.0.24.vec.insert = insertelement <64 x float> %.sroa.0.20.vec.insert, float %71, i32 6
  br i1 %36, label %72, label %._crit_edge.7

._crit_edge.7:                                    ; preds = %._crit_edge96.6
  br label %._crit_edge96.7

72:                                               ; preds = %._crit_edge96.6
  %73 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.7

._crit_edge96.7:                                  ; preds = %72, %._crit_edge.7
  %74 = phi i32 [ %73, %72 ], [ 0, %._crit_edge.7 ]
  %75 = bitcast i32 %74 to float
  %.sroa.0.28.vec.insert = insertelement <64 x float> %.sroa.0.24.vec.insert, float %75, i32 7
  br i1 %36, label %76, label %._crit_edge.8

._crit_edge.8:                                    ; preds = %._crit_edge96.7
  br label %._crit_edge96.8

76:                                               ; preds = %._crit_edge96.7
  %77 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.8

._crit_edge96.8:                                  ; preds = %76, %._crit_edge.8
  %78 = phi i32 [ %77, %76 ], [ 0, %._crit_edge.8 ]
  %79 = bitcast i32 %78 to float
  %.sroa.0.32.vec.insert = insertelement <64 x float> %.sroa.0.28.vec.insert, float %79, i32 8
  br i1 %36, label %80, label %._crit_edge.9

._crit_edge.9:                                    ; preds = %._crit_edge96.8
  br label %._crit_edge96.9

80:                                               ; preds = %._crit_edge96.8
  %81 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.9

._crit_edge96.9:                                  ; preds = %80, %._crit_edge.9
  %82 = phi i32 [ %81, %80 ], [ 0, %._crit_edge.9 ]
  %83 = bitcast i32 %82 to float
  %.sroa.0.36.vec.insert = insertelement <64 x float> %.sroa.0.32.vec.insert, float %83, i32 9
  br i1 %36, label %84, label %._crit_edge.10

._crit_edge.10:                                   ; preds = %._crit_edge96.9
  br label %._crit_edge96.10

84:                                               ; preds = %._crit_edge96.9
  %85 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.10

._crit_edge96.10:                                 ; preds = %84, %._crit_edge.10
  %86 = phi i32 [ %85, %84 ], [ 0, %._crit_edge.10 ]
  %87 = bitcast i32 %86 to float
  %.sroa.0.40.vec.insert = insertelement <64 x float> %.sroa.0.36.vec.insert, float %87, i32 10
  br i1 %36, label %88, label %._crit_edge.11

._crit_edge.11:                                   ; preds = %._crit_edge96.10
  br label %._crit_edge96.11

88:                                               ; preds = %._crit_edge96.10
  %89 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.11

._crit_edge96.11:                                 ; preds = %88, %._crit_edge.11
  %90 = phi i32 [ %89, %88 ], [ 0, %._crit_edge.11 ]
  %91 = bitcast i32 %90 to float
  %.sroa.0.44.vec.insert = insertelement <64 x float> %.sroa.0.40.vec.insert, float %91, i32 11
  br i1 %36, label %92, label %._crit_edge.12

._crit_edge.12:                                   ; preds = %._crit_edge96.11
  br label %._crit_edge96.12

92:                                               ; preds = %._crit_edge96.11
  %93 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.12

._crit_edge96.12:                                 ; preds = %92, %._crit_edge.12
  %94 = phi i32 [ %93, %92 ], [ 0, %._crit_edge.12 ]
  %95 = bitcast i32 %94 to float
  %.sroa.0.48.vec.insert = insertelement <64 x float> %.sroa.0.44.vec.insert, float %95, i32 12
  br i1 %36, label %96, label %._crit_edge.13

._crit_edge.13:                                   ; preds = %._crit_edge96.12
  br label %._crit_edge96.13

96:                                               ; preds = %._crit_edge96.12
  %97 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.13

._crit_edge96.13:                                 ; preds = %96, %._crit_edge.13
  %98 = phi i32 [ %97, %96 ], [ 0, %._crit_edge.13 ]
  %99 = bitcast i32 %98 to float
  %.sroa.0.52.vec.insert = insertelement <64 x float> %.sroa.0.48.vec.insert, float %99, i32 13
  br i1 %36, label %100, label %._crit_edge.14

._crit_edge.14:                                   ; preds = %._crit_edge96.13
  br label %._crit_edge96.14

100:                                              ; preds = %._crit_edge96.13
  %101 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.14

._crit_edge96.14:                                 ; preds = %100, %._crit_edge.14
  %102 = phi i32 [ %101, %100 ], [ 0, %._crit_edge.14 ]
  %103 = bitcast i32 %102 to float
  %.sroa.0.56.vec.insert = insertelement <64 x float> %.sroa.0.52.vec.insert, float %103, i32 14
  br i1 %36, label %104, label %._crit_edge.15

._crit_edge.15:                                   ; preds = %._crit_edge96.14
  br label %._crit_edge96.15

104:                                              ; preds = %._crit_edge96.14
  %105 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.15

._crit_edge96.15:                                 ; preds = %104, %._crit_edge.15
  %106 = phi i32 [ %105, %104 ], [ 0, %._crit_edge.15 ]
  %107 = bitcast i32 %106 to float
  %.sroa.0.60.vec.insert = insertelement <64 x float> %.sroa.0.56.vec.insert, float %107, i32 15
  br i1 %36, label %108, label %._crit_edge.16

._crit_edge.16:                                   ; preds = %._crit_edge96.15
  br label %._crit_edge96.16

108:                                              ; preds = %._crit_edge96.15
  %109 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.16

._crit_edge96.16:                                 ; preds = %108, %._crit_edge.16
  %110 = phi i32 [ %109, %108 ], [ 0, %._crit_edge.16 ]
  %111 = bitcast i32 %110 to float
  %.sroa.0.64.vec.insert = insertelement <64 x float> %.sroa.0.60.vec.insert, float %111, i32 16
  br i1 %36, label %112, label %._crit_edge.17

._crit_edge.17:                                   ; preds = %._crit_edge96.16
  br label %._crit_edge96.17

112:                                              ; preds = %._crit_edge96.16
  %113 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.17

._crit_edge96.17:                                 ; preds = %112, %._crit_edge.17
  %114 = phi i32 [ %113, %112 ], [ 0, %._crit_edge.17 ]
  %115 = bitcast i32 %114 to float
  %.sroa.0.68.vec.insert = insertelement <64 x float> %.sroa.0.64.vec.insert, float %115, i32 17
  br i1 %36, label %116, label %._crit_edge.18

._crit_edge.18:                                   ; preds = %._crit_edge96.17
  br label %._crit_edge96.18

116:                                              ; preds = %._crit_edge96.17
  %117 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.18

._crit_edge96.18:                                 ; preds = %116, %._crit_edge.18
  %118 = phi i32 [ %117, %116 ], [ 0, %._crit_edge.18 ]
  %119 = bitcast i32 %118 to float
  %.sroa.0.72.vec.insert = insertelement <64 x float> %.sroa.0.68.vec.insert, float %119, i32 18
  br i1 %36, label %120, label %._crit_edge.19

._crit_edge.19:                                   ; preds = %._crit_edge96.18
  br label %._crit_edge96.19

120:                                              ; preds = %._crit_edge96.18
  %121 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.19

._crit_edge96.19:                                 ; preds = %120, %._crit_edge.19
  %122 = phi i32 [ %121, %120 ], [ 0, %._crit_edge.19 ]
  %123 = bitcast i32 %122 to float
  %.sroa.0.76.vec.insert = insertelement <64 x float> %.sroa.0.72.vec.insert, float %123, i32 19
  br i1 %36, label %124, label %._crit_edge.20

._crit_edge.20:                                   ; preds = %._crit_edge96.19
  br label %._crit_edge96.20

124:                                              ; preds = %._crit_edge96.19
  %125 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.20

._crit_edge96.20:                                 ; preds = %124, %._crit_edge.20
  %126 = phi i32 [ %125, %124 ], [ 0, %._crit_edge.20 ]
  %127 = bitcast i32 %126 to float
  %.sroa.0.80.vec.insert = insertelement <64 x float> %.sroa.0.76.vec.insert, float %127, i32 20
  br i1 %36, label %128, label %._crit_edge.21

._crit_edge.21:                                   ; preds = %._crit_edge96.20
  br label %._crit_edge96.21

128:                                              ; preds = %._crit_edge96.20
  %129 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.21

._crit_edge96.21:                                 ; preds = %128, %._crit_edge.21
  %130 = phi i32 [ %129, %128 ], [ 0, %._crit_edge.21 ]
  %131 = bitcast i32 %130 to float
  %.sroa.0.84.vec.insert = insertelement <64 x float> %.sroa.0.80.vec.insert, float %131, i32 21
  br i1 %36, label %132, label %._crit_edge.22

._crit_edge.22:                                   ; preds = %._crit_edge96.21
  br label %._crit_edge96.22

132:                                              ; preds = %._crit_edge96.21
  %133 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.22

._crit_edge96.22:                                 ; preds = %132, %._crit_edge.22
  %134 = phi i32 [ %133, %132 ], [ 0, %._crit_edge.22 ]
  %135 = bitcast i32 %134 to float
  %.sroa.0.88.vec.insert = insertelement <64 x float> %.sroa.0.84.vec.insert, float %135, i32 22
  br i1 %36, label %136, label %._crit_edge.23

._crit_edge.23:                                   ; preds = %._crit_edge96.22
  br label %._crit_edge96.23

136:                                              ; preds = %._crit_edge96.22
  %137 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.23

._crit_edge96.23:                                 ; preds = %136, %._crit_edge.23
  %138 = phi i32 [ %137, %136 ], [ 0, %._crit_edge.23 ]
  %139 = bitcast i32 %138 to float
  %.sroa.0.92.vec.insert = insertelement <64 x float> %.sroa.0.88.vec.insert, float %139, i32 23
  br i1 %36, label %140, label %._crit_edge.24

._crit_edge.24:                                   ; preds = %._crit_edge96.23
  br label %._crit_edge96.24

140:                                              ; preds = %._crit_edge96.23
  %141 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.24

._crit_edge96.24:                                 ; preds = %140, %._crit_edge.24
  %142 = phi i32 [ %141, %140 ], [ 0, %._crit_edge.24 ]
  %143 = bitcast i32 %142 to float
  %.sroa.0.96.vec.insert = insertelement <64 x float> %.sroa.0.92.vec.insert, float %143, i32 24
  br i1 %36, label %144, label %._crit_edge.25

._crit_edge.25:                                   ; preds = %._crit_edge96.24
  br label %._crit_edge96.25

144:                                              ; preds = %._crit_edge96.24
  %145 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.25

._crit_edge96.25:                                 ; preds = %144, %._crit_edge.25
  %146 = phi i32 [ %145, %144 ], [ 0, %._crit_edge.25 ]
  %147 = bitcast i32 %146 to float
  %.sroa.0.100.vec.insert = insertelement <64 x float> %.sroa.0.96.vec.insert, float %147, i32 25
  br i1 %36, label %148, label %._crit_edge.26

._crit_edge.26:                                   ; preds = %._crit_edge96.25
  br label %._crit_edge96.26

148:                                              ; preds = %._crit_edge96.25
  %149 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.26

._crit_edge96.26:                                 ; preds = %148, %._crit_edge.26
  %150 = phi i32 [ %149, %148 ], [ 0, %._crit_edge.26 ]
  %151 = bitcast i32 %150 to float
  %.sroa.0.104.vec.insert = insertelement <64 x float> %.sroa.0.100.vec.insert, float %151, i32 26
  br i1 %36, label %152, label %._crit_edge.27

._crit_edge.27:                                   ; preds = %._crit_edge96.26
  br label %._crit_edge96.27

152:                                              ; preds = %._crit_edge96.26
  %153 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.27

._crit_edge96.27:                                 ; preds = %152, %._crit_edge.27
  %154 = phi i32 [ %153, %152 ], [ 0, %._crit_edge.27 ]
  %155 = bitcast i32 %154 to float
  %.sroa.0.108.vec.insert = insertelement <64 x float> %.sroa.0.104.vec.insert, float %155, i32 27
  br i1 %36, label %156, label %._crit_edge.28

._crit_edge.28:                                   ; preds = %._crit_edge96.27
  br label %._crit_edge96.28

156:                                              ; preds = %._crit_edge96.27
  %157 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.28

._crit_edge96.28:                                 ; preds = %156, %._crit_edge.28
  %158 = phi i32 [ %157, %156 ], [ 0, %._crit_edge.28 ]
  %159 = bitcast i32 %158 to float
  %.sroa.0.112.vec.insert = insertelement <64 x float> %.sroa.0.108.vec.insert, float %159, i32 28
  br i1 %36, label %160, label %._crit_edge.29

._crit_edge.29:                                   ; preds = %._crit_edge96.28
  br label %._crit_edge96.29

160:                                              ; preds = %._crit_edge96.28
  %161 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.29

._crit_edge96.29:                                 ; preds = %160, %._crit_edge.29
  %162 = phi i32 [ %161, %160 ], [ 0, %._crit_edge.29 ]
  %163 = bitcast i32 %162 to float
  %.sroa.0.116.vec.insert = insertelement <64 x float> %.sroa.0.112.vec.insert, float %163, i32 29
  br i1 %36, label %164, label %._crit_edge.30

._crit_edge.30:                                   ; preds = %._crit_edge96.29
  br label %._crit_edge96.30

164:                                              ; preds = %._crit_edge96.29
  %165 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.30

._crit_edge96.30:                                 ; preds = %164, %._crit_edge.30
  %166 = phi i32 [ %165, %164 ], [ 0, %._crit_edge.30 ]
  %167 = bitcast i32 %166 to float
  %.sroa.0.120.vec.insert = insertelement <64 x float> %.sroa.0.116.vec.insert, float %167, i32 30
  br i1 %36, label %168, label %._crit_edge.31

._crit_edge.31:                                   ; preds = %._crit_edge96.30
  br label %._crit_edge96.31

168:                                              ; preds = %._crit_edge96.30
  %169 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.31

._crit_edge96.31:                                 ; preds = %168, %._crit_edge.31
  %170 = phi i32 [ %169, %168 ], [ 0, %._crit_edge.31 ]
  %171 = bitcast i32 %170 to float
  %.sroa.0.124.vec.insert = insertelement <64 x float> %.sroa.0.120.vec.insert, float %171, i32 31
  br i1 %36, label %172, label %._crit_edge.32

._crit_edge.32:                                   ; preds = %._crit_edge96.31
  br label %._crit_edge96.32

172:                                              ; preds = %._crit_edge96.31
  %173 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.32

._crit_edge96.32:                                 ; preds = %172, %._crit_edge.32
  %174 = phi i32 [ %173, %172 ], [ 0, %._crit_edge.32 ]
  %175 = bitcast i32 %174 to float
  %.sroa.0.128.vec.insert = insertelement <64 x float> %.sroa.0.124.vec.insert, float %175, i32 32
  br i1 %36, label %176, label %._crit_edge.33

._crit_edge.33:                                   ; preds = %._crit_edge96.32
  br label %._crit_edge96.33

176:                                              ; preds = %._crit_edge96.32
  %177 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.33

._crit_edge96.33:                                 ; preds = %176, %._crit_edge.33
  %178 = phi i32 [ %177, %176 ], [ 0, %._crit_edge.33 ]
  %179 = bitcast i32 %178 to float
  %.sroa.0.132.vec.insert = insertelement <64 x float> %.sroa.0.128.vec.insert, float %179, i32 33
  br i1 %36, label %180, label %._crit_edge.34

._crit_edge.34:                                   ; preds = %._crit_edge96.33
  br label %._crit_edge96.34

180:                                              ; preds = %._crit_edge96.33
  %181 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.34

._crit_edge96.34:                                 ; preds = %180, %._crit_edge.34
  %182 = phi i32 [ %181, %180 ], [ 0, %._crit_edge.34 ]
  %183 = bitcast i32 %182 to float
  %.sroa.0.136.vec.insert = insertelement <64 x float> %.sroa.0.132.vec.insert, float %183, i32 34
  br i1 %36, label %184, label %._crit_edge.35

._crit_edge.35:                                   ; preds = %._crit_edge96.34
  br label %._crit_edge96.35

184:                                              ; preds = %._crit_edge96.34
  %185 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.35

._crit_edge96.35:                                 ; preds = %184, %._crit_edge.35
  %186 = phi i32 [ %185, %184 ], [ 0, %._crit_edge.35 ]
  %187 = bitcast i32 %186 to float
  %.sroa.0.140.vec.insert = insertelement <64 x float> %.sroa.0.136.vec.insert, float %187, i32 35
  br i1 %36, label %188, label %._crit_edge.36

._crit_edge.36:                                   ; preds = %._crit_edge96.35
  br label %._crit_edge96.36

188:                                              ; preds = %._crit_edge96.35
  %189 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.36

._crit_edge96.36:                                 ; preds = %188, %._crit_edge.36
  %190 = phi i32 [ %189, %188 ], [ 0, %._crit_edge.36 ]
  %191 = bitcast i32 %190 to float
  %.sroa.0.144.vec.insert = insertelement <64 x float> %.sroa.0.140.vec.insert, float %191, i32 36
  br i1 %36, label %192, label %._crit_edge.37

._crit_edge.37:                                   ; preds = %._crit_edge96.36
  br label %._crit_edge96.37

192:                                              ; preds = %._crit_edge96.36
  %193 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.37

._crit_edge96.37:                                 ; preds = %192, %._crit_edge.37
  %194 = phi i32 [ %193, %192 ], [ 0, %._crit_edge.37 ]
  %195 = bitcast i32 %194 to float
  %.sroa.0.148.vec.insert = insertelement <64 x float> %.sroa.0.144.vec.insert, float %195, i32 37
  br i1 %36, label %196, label %._crit_edge.38

._crit_edge.38:                                   ; preds = %._crit_edge96.37
  br label %._crit_edge96.38

196:                                              ; preds = %._crit_edge96.37
  %197 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.38

._crit_edge96.38:                                 ; preds = %196, %._crit_edge.38
  %198 = phi i32 [ %197, %196 ], [ 0, %._crit_edge.38 ]
  %199 = bitcast i32 %198 to float
  %.sroa.0.152.vec.insert = insertelement <64 x float> %.sroa.0.148.vec.insert, float %199, i32 38
  br i1 %36, label %200, label %._crit_edge.39

._crit_edge.39:                                   ; preds = %._crit_edge96.38
  br label %._crit_edge96.39

200:                                              ; preds = %._crit_edge96.38
  %201 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.39

._crit_edge96.39:                                 ; preds = %200, %._crit_edge.39
  %202 = phi i32 [ %201, %200 ], [ 0, %._crit_edge.39 ]
  %203 = bitcast i32 %202 to float
  %.sroa.0.156.vec.insert = insertelement <64 x float> %.sroa.0.152.vec.insert, float %203, i32 39
  br i1 %36, label %204, label %._crit_edge.40

._crit_edge.40:                                   ; preds = %._crit_edge96.39
  br label %._crit_edge96.40

204:                                              ; preds = %._crit_edge96.39
  %205 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.40

._crit_edge96.40:                                 ; preds = %204, %._crit_edge.40
  %206 = phi i32 [ %205, %204 ], [ 0, %._crit_edge.40 ]
  %207 = bitcast i32 %206 to float
  %.sroa.0.160.vec.insert = insertelement <64 x float> %.sroa.0.156.vec.insert, float %207, i32 40
  br i1 %36, label %208, label %._crit_edge.41

._crit_edge.41:                                   ; preds = %._crit_edge96.40
  br label %._crit_edge96.41

208:                                              ; preds = %._crit_edge96.40
  %209 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.41

._crit_edge96.41:                                 ; preds = %208, %._crit_edge.41
  %210 = phi i32 [ %209, %208 ], [ 0, %._crit_edge.41 ]
  %211 = bitcast i32 %210 to float
  %.sroa.0.164.vec.insert = insertelement <64 x float> %.sroa.0.160.vec.insert, float %211, i32 41
  br i1 %36, label %212, label %._crit_edge.42

._crit_edge.42:                                   ; preds = %._crit_edge96.41
  br label %._crit_edge96.42

212:                                              ; preds = %._crit_edge96.41
  %213 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.42

._crit_edge96.42:                                 ; preds = %212, %._crit_edge.42
  %214 = phi i32 [ %213, %212 ], [ 0, %._crit_edge.42 ]
  %215 = bitcast i32 %214 to float
  %.sroa.0.168.vec.insert = insertelement <64 x float> %.sroa.0.164.vec.insert, float %215, i32 42
  br i1 %36, label %216, label %._crit_edge.43

._crit_edge.43:                                   ; preds = %._crit_edge96.42
  br label %._crit_edge96.43

216:                                              ; preds = %._crit_edge96.42
  %217 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.43

._crit_edge96.43:                                 ; preds = %216, %._crit_edge.43
  %218 = phi i32 [ %217, %216 ], [ 0, %._crit_edge.43 ]
  %219 = bitcast i32 %218 to float
  %.sroa.0.172.vec.insert = insertelement <64 x float> %.sroa.0.168.vec.insert, float %219, i32 43
  br i1 %36, label %220, label %._crit_edge.44

._crit_edge.44:                                   ; preds = %._crit_edge96.43
  br label %._crit_edge96.44

220:                                              ; preds = %._crit_edge96.43
  %221 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.44

._crit_edge96.44:                                 ; preds = %220, %._crit_edge.44
  %222 = phi i32 [ %221, %220 ], [ 0, %._crit_edge.44 ]
  %223 = bitcast i32 %222 to float
  %.sroa.0.176.vec.insert = insertelement <64 x float> %.sroa.0.172.vec.insert, float %223, i32 44
  br i1 %36, label %224, label %._crit_edge.45

._crit_edge.45:                                   ; preds = %._crit_edge96.44
  br label %._crit_edge96.45

224:                                              ; preds = %._crit_edge96.44
  %225 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.45

._crit_edge96.45:                                 ; preds = %224, %._crit_edge.45
  %226 = phi i32 [ %225, %224 ], [ 0, %._crit_edge.45 ]
  %227 = bitcast i32 %226 to float
  %.sroa.0.180.vec.insert = insertelement <64 x float> %.sroa.0.176.vec.insert, float %227, i32 45
  br i1 %36, label %228, label %._crit_edge.46

._crit_edge.46:                                   ; preds = %._crit_edge96.45
  br label %._crit_edge96.46

228:                                              ; preds = %._crit_edge96.45
  %229 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.46

._crit_edge96.46:                                 ; preds = %228, %._crit_edge.46
  %230 = phi i32 [ %229, %228 ], [ 0, %._crit_edge.46 ]
  %231 = bitcast i32 %230 to float
  %.sroa.0.184.vec.insert = insertelement <64 x float> %.sroa.0.180.vec.insert, float %231, i32 46
  br i1 %36, label %232, label %._crit_edge.47

._crit_edge.47:                                   ; preds = %._crit_edge96.46
  br label %._crit_edge96.47

232:                                              ; preds = %._crit_edge96.46
  %233 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.47

._crit_edge96.47:                                 ; preds = %232, %._crit_edge.47
  %234 = phi i32 [ %233, %232 ], [ 0, %._crit_edge.47 ]
  %235 = bitcast i32 %234 to float
  %.sroa.0.188.vec.insert = insertelement <64 x float> %.sroa.0.184.vec.insert, float %235, i32 47
  br i1 %36, label %236, label %._crit_edge.48

._crit_edge.48:                                   ; preds = %._crit_edge96.47
  br label %._crit_edge96.48

236:                                              ; preds = %._crit_edge96.47
  %237 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.48

._crit_edge96.48:                                 ; preds = %236, %._crit_edge.48
  %238 = phi i32 [ %237, %236 ], [ 0, %._crit_edge.48 ]
  %239 = bitcast i32 %238 to float
  %.sroa.0.192.vec.insert = insertelement <64 x float> %.sroa.0.188.vec.insert, float %239, i32 48
  br i1 %36, label %240, label %._crit_edge.49

._crit_edge.49:                                   ; preds = %._crit_edge96.48
  br label %._crit_edge96.49

240:                                              ; preds = %._crit_edge96.48
  %241 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.49

._crit_edge96.49:                                 ; preds = %240, %._crit_edge.49
  %242 = phi i32 [ %241, %240 ], [ 0, %._crit_edge.49 ]
  %243 = bitcast i32 %242 to float
  %.sroa.0.196.vec.insert = insertelement <64 x float> %.sroa.0.192.vec.insert, float %243, i32 49
  br i1 %36, label %244, label %._crit_edge.50

._crit_edge.50:                                   ; preds = %._crit_edge96.49
  br label %._crit_edge96.50

244:                                              ; preds = %._crit_edge96.49
  %245 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.50

._crit_edge96.50:                                 ; preds = %244, %._crit_edge.50
  %246 = phi i32 [ %245, %244 ], [ 0, %._crit_edge.50 ]
  %247 = bitcast i32 %246 to float
  %.sroa.0.200.vec.insert = insertelement <64 x float> %.sroa.0.196.vec.insert, float %247, i32 50
  br i1 %36, label %248, label %._crit_edge.51

._crit_edge.51:                                   ; preds = %._crit_edge96.50
  br label %._crit_edge96.51

248:                                              ; preds = %._crit_edge96.50
  %249 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.51

._crit_edge96.51:                                 ; preds = %248, %._crit_edge.51
  %250 = phi i32 [ %249, %248 ], [ 0, %._crit_edge.51 ]
  %251 = bitcast i32 %250 to float
  %.sroa.0.204.vec.insert = insertelement <64 x float> %.sroa.0.200.vec.insert, float %251, i32 51
  br i1 %36, label %252, label %._crit_edge.52

._crit_edge.52:                                   ; preds = %._crit_edge96.51
  br label %._crit_edge96.52

252:                                              ; preds = %._crit_edge96.51
  %253 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.52

._crit_edge96.52:                                 ; preds = %252, %._crit_edge.52
  %254 = phi i32 [ %253, %252 ], [ 0, %._crit_edge.52 ]
  %255 = bitcast i32 %254 to float
  %.sroa.0.208.vec.insert = insertelement <64 x float> %.sroa.0.204.vec.insert, float %255, i32 52
  br i1 %36, label %256, label %._crit_edge.53

._crit_edge.53:                                   ; preds = %._crit_edge96.52
  br label %._crit_edge96.53

256:                                              ; preds = %._crit_edge96.52
  %257 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.53

._crit_edge96.53:                                 ; preds = %256, %._crit_edge.53
  %258 = phi i32 [ %257, %256 ], [ 0, %._crit_edge.53 ]
  %259 = bitcast i32 %258 to float
  %.sroa.0.212.vec.insert = insertelement <64 x float> %.sroa.0.208.vec.insert, float %259, i32 53
  br i1 %36, label %260, label %._crit_edge.54

._crit_edge.54:                                   ; preds = %._crit_edge96.53
  br label %._crit_edge96.54

260:                                              ; preds = %._crit_edge96.53
  %261 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.54

._crit_edge96.54:                                 ; preds = %260, %._crit_edge.54
  %262 = phi i32 [ %261, %260 ], [ 0, %._crit_edge.54 ]
  %263 = bitcast i32 %262 to float
  %.sroa.0.216.vec.insert = insertelement <64 x float> %.sroa.0.212.vec.insert, float %263, i32 54
  br i1 %36, label %264, label %._crit_edge.55

._crit_edge.55:                                   ; preds = %._crit_edge96.54
  br label %._crit_edge96.55

264:                                              ; preds = %._crit_edge96.54
  %265 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.55

._crit_edge96.55:                                 ; preds = %264, %._crit_edge.55
  %266 = phi i32 [ %265, %264 ], [ 0, %._crit_edge.55 ]
  %267 = bitcast i32 %266 to float
  %.sroa.0.220.vec.insert = insertelement <64 x float> %.sroa.0.216.vec.insert, float %267, i32 55
  br i1 %36, label %268, label %._crit_edge.56

._crit_edge.56:                                   ; preds = %._crit_edge96.55
  br label %._crit_edge96.56

268:                                              ; preds = %._crit_edge96.55
  %269 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.56

._crit_edge96.56:                                 ; preds = %268, %._crit_edge.56
  %270 = phi i32 [ %269, %268 ], [ 0, %._crit_edge.56 ]
  %271 = bitcast i32 %270 to float
  %.sroa.0.224.vec.insert = insertelement <64 x float> %.sroa.0.220.vec.insert, float %271, i32 56
  br i1 %36, label %272, label %._crit_edge.57

._crit_edge.57:                                   ; preds = %._crit_edge96.56
  br label %._crit_edge96.57

272:                                              ; preds = %._crit_edge96.56
  %273 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.57

._crit_edge96.57:                                 ; preds = %272, %._crit_edge.57
  %274 = phi i32 [ %273, %272 ], [ 0, %._crit_edge.57 ]
  %275 = bitcast i32 %274 to float
  %.sroa.0.228.vec.insert = insertelement <64 x float> %.sroa.0.224.vec.insert, float %275, i32 57
  br i1 %36, label %276, label %._crit_edge.58

._crit_edge.58:                                   ; preds = %._crit_edge96.57
  br label %._crit_edge96.58

276:                                              ; preds = %._crit_edge96.57
  %277 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.58

._crit_edge96.58:                                 ; preds = %276, %._crit_edge.58
  %278 = phi i32 [ %277, %276 ], [ 0, %._crit_edge.58 ]
  %279 = bitcast i32 %278 to float
  %.sroa.0.232.vec.insert = insertelement <64 x float> %.sroa.0.228.vec.insert, float %279, i32 58
  br i1 %36, label %280, label %._crit_edge.59

._crit_edge.59:                                   ; preds = %._crit_edge96.58
  br label %._crit_edge96.59

280:                                              ; preds = %._crit_edge96.58
  %281 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.59

._crit_edge96.59:                                 ; preds = %280, %._crit_edge.59
  %282 = phi i32 [ %281, %280 ], [ 0, %._crit_edge.59 ]
  %283 = bitcast i32 %282 to float
  %.sroa.0.236.vec.insert = insertelement <64 x float> %.sroa.0.232.vec.insert, float %283, i32 59
  br i1 %36, label %284, label %._crit_edge.60

._crit_edge.60:                                   ; preds = %._crit_edge96.59
  br label %._crit_edge96.60

284:                                              ; preds = %._crit_edge96.59
  %285 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.60

._crit_edge96.60:                                 ; preds = %284, %._crit_edge.60
  %286 = phi i32 [ %285, %284 ], [ 0, %._crit_edge.60 ]
  %287 = bitcast i32 %286 to float
  %.sroa.0.240.vec.insert = insertelement <64 x float> %.sroa.0.236.vec.insert, float %287, i32 60
  br i1 %36, label %288, label %._crit_edge.61

._crit_edge.61:                                   ; preds = %._crit_edge96.60
  br label %._crit_edge96.61

288:                                              ; preds = %._crit_edge96.60
  %289 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.61

._crit_edge96.61:                                 ; preds = %288, %._crit_edge.61
  %290 = phi i32 [ %289, %288 ], [ 0, %._crit_edge.61 ]
  %291 = bitcast i32 %290 to float
  %.sroa.0.244.vec.insert = insertelement <64 x float> %.sroa.0.240.vec.insert, float %291, i32 61
  br i1 %36, label %292, label %._crit_edge.62

._crit_edge.62:                                   ; preds = %._crit_edge96.61
  br label %._crit_edge96.62

292:                                              ; preds = %._crit_edge96.61
  %293 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.62

._crit_edge96.62:                                 ; preds = %292, %._crit_edge.62
  %294 = phi i32 [ %293, %292 ], [ 0, %._crit_edge.62 ]
  %295 = bitcast i32 %294 to float
  %.sroa.0.248.vec.insert = insertelement <64 x float> %.sroa.0.244.vec.insert, float %295, i32 62
  br i1 %36, label %296, label %._crit_edge.63

._crit_edge.63:                                   ; preds = %._crit_edge96.62
  br label %._crit_edge96.63

296:                                              ; preds = %._crit_edge96.62
  %297 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.63

._crit_edge96.63:                                 ; preds = %296, %._crit_edge.63
  %298 = phi i32 [ %297, %296 ], [ 0, %._crit_edge.63 ]
  %299 = bitcast i32 %298 to float
  %.sroa.0.252.vec.insert = insertelement <64 x float> %.sroa.0.248.vec.insert, float %299, i32 63
  br i1 %36, label %300, label %._crit_edge.64

._crit_edge.64:                                   ; preds = %._crit_edge96.63
  br label %._crit_edge96.64

300:                                              ; preds = %._crit_edge96.63
  %301 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.64

._crit_edge96.64:                                 ; preds = %300, %._crit_edge.64
  %302 = phi i32 [ %301, %300 ], [ 0, %._crit_edge.64 ]
  %303 = bitcast i32 %302 to float
  %.sroa.65.256.vec.insert = insertelement <64 x float> undef, float %303, i32 0
  br i1 %36, label %304, label %._crit_edge.65

._crit_edge.65:                                   ; preds = %._crit_edge96.64
  br label %._crit_edge96.65

304:                                              ; preds = %._crit_edge96.64
  %305 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.65

._crit_edge96.65:                                 ; preds = %304, %._crit_edge.65
  %306 = phi i32 [ %305, %304 ], [ 0, %._crit_edge.65 ]
  %307 = bitcast i32 %306 to float
  %.sroa.65.260.vec.insert = insertelement <64 x float> %.sroa.65.256.vec.insert, float %307, i32 1
  br i1 %36, label %308, label %._crit_edge.66

._crit_edge.66:                                   ; preds = %._crit_edge96.65
  br label %._crit_edge96.66

308:                                              ; preds = %._crit_edge96.65
  %309 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.66

._crit_edge96.66:                                 ; preds = %308, %._crit_edge.66
  %310 = phi i32 [ %309, %308 ], [ 0, %._crit_edge.66 ]
  %311 = bitcast i32 %310 to float
  %.sroa.65.264.vec.insert = insertelement <64 x float> %.sroa.65.260.vec.insert, float %311, i32 2
  br i1 %36, label %312, label %._crit_edge.67

._crit_edge.67:                                   ; preds = %._crit_edge96.66
  br label %._crit_edge96.67

312:                                              ; preds = %._crit_edge96.66
  %313 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.67

._crit_edge96.67:                                 ; preds = %312, %._crit_edge.67
  %314 = phi i32 [ %313, %312 ], [ 0, %._crit_edge.67 ]
  %315 = bitcast i32 %314 to float
  %.sroa.65.268.vec.insert = insertelement <64 x float> %.sroa.65.264.vec.insert, float %315, i32 3
  br i1 %36, label %316, label %._crit_edge.68

._crit_edge.68:                                   ; preds = %._crit_edge96.67
  br label %._crit_edge96.68

316:                                              ; preds = %._crit_edge96.67
  %317 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.68

._crit_edge96.68:                                 ; preds = %316, %._crit_edge.68
  %318 = phi i32 [ %317, %316 ], [ 0, %._crit_edge.68 ]
  %319 = bitcast i32 %318 to float
  %.sroa.65.272.vec.insert = insertelement <64 x float> %.sroa.65.268.vec.insert, float %319, i32 4
  br i1 %36, label %320, label %._crit_edge.69

._crit_edge.69:                                   ; preds = %._crit_edge96.68
  br label %._crit_edge96.69

320:                                              ; preds = %._crit_edge96.68
  %321 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.69

._crit_edge96.69:                                 ; preds = %320, %._crit_edge.69
  %322 = phi i32 [ %321, %320 ], [ 0, %._crit_edge.69 ]
  %323 = bitcast i32 %322 to float
  %.sroa.65.276.vec.insert = insertelement <64 x float> %.sroa.65.272.vec.insert, float %323, i32 5
  br i1 %36, label %324, label %._crit_edge.70

._crit_edge.70:                                   ; preds = %._crit_edge96.69
  br label %._crit_edge96.70

324:                                              ; preds = %._crit_edge96.69
  %325 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.70

._crit_edge96.70:                                 ; preds = %324, %._crit_edge.70
  %326 = phi i32 [ %325, %324 ], [ 0, %._crit_edge.70 ]
  %327 = bitcast i32 %326 to float
  %.sroa.65.280.vec.insert = insertelement <64 x float> %.sroa.65.276.vec.insert, float %327, i32 6
  br i1 %36, label %328, label %._crit_edge.71

._crit_edge.71:                                   ; preds = %._crit_edge96.70
  br label %._crit_edge96.71

328:                                              ; preds = %._crit_edge96.70
  %329 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.71

._crit_edge96.71:                                 ; preds = %328, %._crit_edge.71
  %330 = phi i32 [ %329, %328 ], [ 0, %._crit_edge.71 ]
  %331 = bitcast i32 %330 to float
  %.sroa.65.284.vec.insert = insertelement <64 x float> %.sroa.65.280.vec.insert, float %331, i32 7
  br i1 %36, label %332, label %._crit_edge.72

._crit_edge.72:                                   ; preds = %._crit_edge96.71
  br label %._crit_edge96.72

332:                                              ; preds = %._crit_edge96.71
  %333 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.72

._crit_edge96.72:                                 ; preds = %332, %._crit_edge.72
  %334 = phi i32 [ %333, %332 ], [ 0, %._crit_edge.72 ]
  %335 = bitcast i32 %334 to float
  %.sroa.65.288.vec.insert = insertelement <64 x float> %.sroa.65.284.vec.insert, float %335, i32 8
  br i1 %36, label %336, label %._crit_edge.73

._crit_edge.73:                                   ; preds = %._crit_edge96.72
  br label %._crit_edge96.73

336:                                              ; preds = %._crit_edge96.72
  %337 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.73

._crit_edge96.73:                                 ; preds = %336, %._crit_edge.73
  %338 = phi i32 [ %337, %336 ], [ 0, %._crit_edge.73 ]
  %339 = bitcast i32 %338 to float
  %.sroa.65.292.vec.insert = insertelement <64 x float> %.sroa.65.288.vec.insert, float %339, i32 9
  br i1 %36, label %340, label %._crit_edge.74

._crit_edge.74:                                   ; preds = %._crit_edge96.73
  br label %._crit_edge96.74

340:                                              ; preds = %._crit_edge96.73
  %341 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.74

._crit_edge96.74:                                 ; preds = %340, %._crit_edge.74
  %342 = phi i32 [ %341, %340 ], [ 0, %._crit_edge.74 ]
  %343 = bitcast i32 %342 to float
  %.sroa.65.296.vec.insert = insertelement <64 x float> %.sroa.65.292.vec.insert, float %343, i32 10
  br i1 %36, label %344, label %._crit_edge.75

._crit_edge.75:                                   ; preds = %._crit_edge96.74
  br label %._crit_edge96.75

344:                                              ; preds = %._crit_edge96.74
  %345 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.75

._crit_edge96.75:                                 ; preds = %344, %._crit_edge.75
  %346 = phi i32 [ %345, %344 ], [ 0, %._crit_edge.75 ]
  %347 = bitcast i32 %346 to float
  %.sroa.65.300.vec.insert = insertelement <64 x float> %.sroa.65.296.vec.insert, float %347, i32 11
  br i1 %36, label %348, label %._crit_edge.76

._crit_edge.76:                                   ; preds = %._crit_edge96.75
  br label %._crit_edge96.76

348:                                              ; preds = %._crit_edge96.75
  %349 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.76

._crit_edge96.76:                                 ; preds = %348, %._crit_edge.76
  %350 = phi i32 [ %349, %348 ], [ 0, %._crit_edge.76 ]
  %351 = bitcast i32 %350 to float
  %.sroa.65.304.vec.insert = insertelement <64 x float> %.sroa.65.300.vec.insert, float %351, i32 12
  br i1 %36, label %352, label %._crit_edge.77

._crit_edge.77:                                   ; preds = %._crit_edge96.76
  br label %._crit_edge96.77

352:                                              ; preds = %._crit_edge96.76
  %353 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.77

._crit_edge96.77:                                 ; preds = %352, %._crit_edge.77
  %354 = phi i32 [ %353, %352 ], [ 0, %._crit_edge.77 ]
  %355 = bitcast i32 %354 to float
  %.sroa.65.308.vec.insert = insertelement <64 x float> %.sroa.65.304.vec.insert, float %355, i32 13
  br i1 %36, label %356, label %._crit_edge.78

._crit_edge.78:                                   ; preds = %._crit_edge96.77
  br label %._crit_edge96.78

356:                                              ; preds = %._crit_edge96.77
  %357 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.78

._crit_edge96.78:                                 ; preds = %356, %._crit_edge.78
  %358 = phi i32 [ %357, %356 ], [ 0, %._crit_edge.78 ]
  %359 = bitcast i32 %358 to float
  %.sroa.65.312.vec.insert = insertelement <64 x float> %.sroa.65.308.vec.insert, float %359, i32 14
  br i1 %36, label %360, label %._crit_edge.79

._crit_edge.79:                                   ; preds = %._crit_edge96.78
  br label %._crit_edge96.79

360:                                              ; preds = %._crit_edge96.78
  %361 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.79

._crit_edge96.79:                                 ; preds = %360, %._crit_edge.79
  %362 = phi i32 [ %361, %360 ], [ 0, %._crit_edge.79 ]
  %363 = bitcast i32 %362 to float
  %.sroa.65.316.vec.insert = insertelement <64 x float> %.sroa.65.312.vec.insert, float %363, i32 15
  br i1 %36, label %364, label %._crit_edge.80

._crit_edge.80:                                   ; preds = %._crit_edge96.79
  br label %._crit_edge96.80

364:                                              ; preds = %._crit_edge96.79
  %365 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.80

._crit_edge96.80:                                 ; preds = %364, %._crit_edge.80
  %366 = phi i32 [ %365, %364 ], [ 0, %._crit_edge.80 ]
  %367 = bitcast i32 %366 to float
  %.sroa.65.320.vec.insert = insertelement <64 x float> %.sroa.65.316.vec.insert, float %367, i32 16
  br i1 %36, label %368, label %._crit_edge.81

._crit_edge.81:                                   ; preds = %._crit_edge96.80
  br label %._crit_edge96.81

368:                                              ; preds = %._crit_edge96.80
  %369 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.81

._crit_edge96.81:                                 ; preds = %368, %._crit_edge.81
  %370 = phi i32 [ %369, %368 ], [ 0, %._crit_edge.81 ]
  %371 = bitcast i32 %370 to float
  %.sroa.65.324.vec.insert = insertelement <64 x float> %.sroa.65.320.vec.insert, float %371, i32 17
  br i1 %36, label %372, label %._crit_edge.82

._crit_edge.82:                                   ; preds = %._crit_edge96.81
  br label %._crit_edge96.82

372:                                              ; preds = %._crit_edge96.81
  %373 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.82

._crit_edge96.82:                                 ; preds = %372, %._crit_edge.82
  %374 = phi i32 [ %373, %372 ], [ 0, %._crit_edge.82 ]
  %375 = bitcast i32 %374 to float
  %.sroa.65.328.vec.insert = insertelement <64 x float> %.sroa.65.324.vec.insert, float %375, i32 18
  br i1 %36, label %376, label %._crit_edge.83

._crit_edge.83:                                   ; preds = %._crit_edge96.82
  br label %._crit_edge96.83

376:                                              ; preds = %._crit_edge96.82
  %377 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.83

._crit_edge96.83:                                 ; preds = %376, %._crit_edge.83
  %378 = phi i32 [ %377, %376 ], [ 0, %._crit_edge.83 ]
  %379 = bitcast i32 %378 to float
  %.sroa.65.332.vec.insert = insertelement <64 x float> %.sroa.65.328.vec.insert, float %379, i32 19
  br i1 %36, label %380, label %._crit_edge.84

._crit_edge.84:                                   ; preds = %._crit_edge96.83
  br label %._crit_edge96.84

380:                                              ; preds = %._crit_edge96.83
  %381 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.84

._crit_edge96.84:                                 ; preds = %380, %._crit_edge.84
  %382 = phi i32 [ %381, %380 ], [ 0, %._crit_edge.84 ]
  %383 = bitcast i32 %382 to float
  %.sroa.65.336.vec.insert = insertelement <64 x float> %.sroa.65.332.vec.insert, float %383, i32 20
  br i1 %36, label %384, label %._crit_edge.85

._crit_edge.85:                                   ; preds = %._crit_edge96.84
  br label %._crit_edge96.85

384:                                              ; preds = %._crit_edge96.84
  %385 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.85

._crit_edge96.85:                                 ; preds = %384, %._crit_edge.85
  %386 = phi i32 [ %385, %384 ], [ 0, %._crit_edge.85 ]
  %387 = bitcast i32 %386 to float
  %.sroa.65.340.vec.insert = insertelement <64 x float> %.sroa.65.336.vec.insert, float %387, i32 21
  br i1 %36, label %388, label %._crit_edge.86

._crit_edge.86:                                   ; preds = %._crit_edge96.85
  br label %._crit_edge96.86

388:                                              ; preds = %._crit_edge96.85
  %389 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.86

._crit_edge96.86:                                 ; preds = %388, %._crit_edge.86
  %390 = phi i32 [ %389, %388 ], [ 0, %._crit_edge.86 ]
  %391 = bitcast i32 %390 to float
  %.sroa.65.344.vec.insert = insertelement <64 x float> %.sroa.65.340.vec.insert, float %391, i32 22
  br i1 %36, label %392, label %._crit_edge.87

._crit_edge.87:                                   ; preds = %._crit_edge96.86
  br label %._crit_edge96.87

392:                                              ; preds = %._crit_edge96.86
  %393 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.87

._crit_edge96.87:                                 ; preds = %392, %._crit_edge.87
  %394 = phi i32 [ %393, %392 ], [ 0, %._crit_edge.87 ]
  %395 = bitcast i32 %394 to float
  %.sroa.65.348.vec.insert = insertelement <64 x float> %.sroa.65.344.vec.insert, float %395, i32 23
  br i1 %36, label %396, label %._crit_edge.88

._crit_edge.88:                                   ; preds = %._crit_edge96.87
  br label %._crit_edge96.88

396:                                              ; preds = %._crit_edge96.87
  %397 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.88

._crit_edge96.88:                                 ; preds = %396, %._crit_edge.88
  %398 = phi i32 [ %397, %396 ], [ 0, %._crit_edge.88 ]
  %399 = bitcast i32 %398 to float
  %.sroa.65.352.vec.insert = insertelement <64 x float> %.sroa.65.348.vec.insert, float %399, i32 24
  br i1 %36, label %400, label %._crit_edge.89

._crit_edge.89:                                   ; preds = %._crit_edge96.88
  br label %._crit_edge96.89

400:                                              ; preds = %._crit_edge96.88
  %401 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.89

._crit_edge96.89:                                 ; preds = %400, %._crit_edge.89
  %402 = phi i32 [ %401, %400 ], [ 0, %._crit_edge.89 ]
  %403 = bitcast i32 %402 to float
  %.sroa.65.356.vec.insert = insertelement <64 x float> %.sroa.65.352.vec.insert, float %403, i32 25
  br i1 %36, label %404, label %._crit_edge.90

._crit_edge.90:                                   ; preds = %._crit_edge96.89
  br label %._crit_edge96.90

404:                                              ; preds = %._crit_edge96.89
  %405 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.90

._crit_edge96.90:                                 ; preds = %404, %._crit_edge.90
  %406 = phi i32 [ %405, %404 ], [ 0, %._crit_edge.90 ]
  %407 = bitcast i32 %406 to float
  %.sroa.65.360.vec.insert = insertelement <64 x float> %.sroa.65.356.vec.insert, float %407, i32 26
  br i1 %36, label %408, label %._crit_edge.91

._crit_edge.91:                                   ; preds = %._crit_edge96.90
  br label %._crit_edge96.91

408:                                              ; preds = %._crit_edge96.90
  %409 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.91

._crit_edge96.91:                                 ; preds = %408, %._crit_edge.91
  %410 = phi i32 [ %409, %408 ], [ 0, %._crit_edge.91 ]
  %411 = bitcast i32 %410 to float
  %.sroa.65.364.vec.insert = insertelement <64 x float> %.sroa.65.360.vec.insert, float %411, i32 27
  br i1 %36, label %412, label %._crit_edge.92

._crit_edge.92:                                   ; preds = %._crit_edge96.91
  br label %._crit_edge96.92

412:                                              ; preds = %._crit_edge96.91
  %413 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.92

._crit_edge96.92:                                 ; preds = %412, %._crit_edge.92
  %414 = phi i32 [ %413, %412 ], [ 0, %._crit_edge.92 ]
  %415 = bitcast i32 %414 to float
  %.sroa.65.368.vec.insert = insertelement <64 x float> %.sroa.65.364.vec.insert, float %415, i32 28
  br i1 %36, label %416, label %._crit_edge.93

._crit_edge.93:                                   ; preds = %._crit_edge96.92
  br label %._crit_edge96.93

416:                                              ; preds = %._crit_edge96.92
  %417 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.93

._crit_edge96.93:                                 ; preds = %416, %._crit_edge.93
  %418 = phi i32 [ %417, %416 ], [ 0, %._crit_edge.93 ]
  %419 = bitcast i32 %418 to float
  %.sroa.65.372.vec.insert = insertelement <64 x float> %.sroa.65.368.vec.insert, float %419, i32 29
  br i1 %36, label %420, label %._crit_edge.94

._crit_edge.94:                                   ; preds = %._crit_edge96.93
  br label %._crit_edge96.94

420:                                              ; preds = %._crit_edge96.93
  %421 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.94

._crit_edge96.94:                                 ; preds = %420, %._crit_edge.94
  %422 = phi i32 [ %421, %420 ], [ 0, %._crit_edge.94 ]
  %423 = bitcast i32 %422 to float
  %.sroa.65.376.vec.insert = insertelement <64 x float> %.sroa.65.372.vec.insert, float %423, i32 30
  br i1 %36, label %424, label %._crit_edge.95

._crit_edge.95:                                   ; preds = %._crit_edge96.94
  br label %._crit_edge96.95

424:                                              ; preds = %._crit_edge96.94
  %425 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.95

._crit_edge96.95:                                 ; preds = %424, %._crit_edge.95
  %426 = phi i32 [ %425, %424 ], [ 0, %._crit_edge.95 ]
  %427 = bitcast i32 %426 to float
  %.sroa.65.380.vec.insert = insertelement <64 x float> %.sroa.65.376.vec.insert, float %427, i32 31
  br i1 %36, label %428, label %._crit_edge.96

._crit_edge.96:                                   ; preds = %._crit_edge96.95
  br label %._crit_edge96.96

428:                                              ; preds = %._crit_edge96.95
  %429 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.96

._crit_edge96.96:                                 ; preds = %428, %._crit_edge.96
  %430 = phi i32 [ %429, %428 ], [ 0, %._crit_edge.96 ]
  %431 = bitcast i32 %430 to float
  %.sroa.65.384.vec.insert = insertelement <64 x float> %.sroa.65.380.vec.insert, float %431, i32 32
  br i1 %36, label %432, label %._crit_edge.97

._crit_edge.97:                                   ; preds = %._crit_edge96.96
  br label %._crit_edge96.97

432:                                              ; preds = %._crit_edge96.96
  %433 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.97

._crit_edge96.97:                                 ; preds = %432, %._crit_edge.97
  %434 = phi i32 [ %433, %432 ], [ 0, %._crit_edge.97 ]
  %435 = bitcast i32 %434 to float
  %.sroa.65.388.vec.insert = insertelement <64 x float> %.sroa.65.384.vec.insert, float %435, i32 33
  br i1 %36, label %436, label %._crit_edge.98

._crit_edge.98:                                   ; preds = %._crit_edge96.97
  br label %._crit_edge96.98

436:                                              ; preds = %._crit_edge96.97
  %437 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.98

._crit_edge96.98:                                 ; preds = %436, %._crit_edge.98
  %438 = phi i32 [ %437, %436 ], [ 0, %._crit_edge.98 ]
  %439 = bitcast i32 %438 to float
  %.sroa.65.392.vec.insert = insertelement <64 x float> %.sroa.65.388.vec.insert, float %439, i32 34
  br i1 %36, label %440, label %._crit_edge.99

._crit_edge.99:                                   ; preds = %._crit_edge96.98
  br label %._crit_edge96.99

440:                                              ; preds = %._crit_edge96.98
  %441 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.99

._crit_edge96.99:                                 ; preds = %440, %._crit_edge.99
  %442 = phi i32 [ %441, %440 ], [ 0, %._crit_edge.99 ]
  %443 = bitcast i32 %442 to float
  %.sroa.65.396.vec.insert = insertelement <64 x float> %.sroa.65.392.vec.insert, float %443, i32 35
  br i1 %36, label %444, label %._crit_edge.100

._crit_edge.100:                                  ; preds = %._crit_edge96.99
  br label %._crit_edge96.100

444:                                              ; preds = %._crit_edge96.99
  %445 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.100

._crit_edge96.100:                                ; preds = %444, %._crit_edge.100
  %446 = phi i32 [ %445, %444 ], [ 0, %._crit_edge.100 ]
  %447 = bitcast i32 %446 to float
  %.sroa.65.400.vec.insert = insertelement <64 x float> %.sroa.65.396.vec.insert, float %447, i32 36
  br i1 %36, label %448, label %._crit_edge.101

._crit_edge.101:                                  ; preds = %._crit_edge96.100
  br label %._crit_edge96.101

448:                                              ; preds = %._crit_edge96.100
  %449 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.101

._crit_edge96.101:                                ; preds = %448, %._crit_edge.101
  %450 = phi i32 [ %449, %448 ], [ 0, %._crit_edge.101 ]
  %451 = bitcast i32 %450 to float
  %.sroa.65.404.vec.insert = insertelement <64 x float> %.sroa.65.400.vec.insert, float %451, i32 37
  br i1 %36, label %452, label %._crit_edge.102

._crit_edge.102:                                  ; preds = %._crit_edge96.101
  br label %._crit_edge96.102

452:                                              ; preds = %._crit_edge96.101
  %453 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.102

._crit_edge96.102:                                ; preds = %452, %._crit_edge.102
  %454 = phi i32 [ %453, %452 ], [ 0, %._crit_edge.102 ]
  %455 = bitcast i32 %454 to float
  %.sroa.65.408.vec.insert = insertelement <64 x float> %.sroa.65.404.vec.insert, float %455, i32 38
  br i1 %36, label %456, label %._crit_edge.103

._crit_edge.103:                                  ; preds = %._crit_edge96.102
  br label %._crit_edge96.103

456:                                              ; preds = %._crit_edge96.102
  %457 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.103

._crit_edge96.103:                                ; preds = %456, %._crit_edge.103
  %458 = phi i32 [ %457, %456 ], [ 0, %._crit_edge.103 ]
  %459 = bitcast i32 %458 to float
  %.sroa.65.412.vec.insert = insertelement <64 x float> %.sroa.65.408.vec.insert, float %459, i32 39
  br i1 %36, label %460, label %._crit_edge.104

._crit_edge.104:                                  ; preds = %._crit_edge96.103
  br label %._crit_edge96.104

460:                                              ; preds = %._crit_edge96.103
  %461 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.104

._crit_edge96.104:                                ; preds = %460, %._crit_edge.104
  %462 = phi i32 [ %461, %460 ], [ 0, %._crit_edge.104 ]
  %463 = bitcast i32 %462 to float
  %.sroa.65.416.vec.insert = insertelement <64 x float> %.sroa.65.412.vec.insert, float %463, i32 40
  br i1 %36, label %464, label %._crit_edge.105

._crit_edge.105:                                  ; preds = %._crit_edge96.104
  br label %._crit_edge96.105

464:                                              ; preds = %._crit_edge96.104
  %465 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.105

._crit_edge96.105:                                ; preds = %464, %._crit_edge.105
  %466 = phi i32 [ %465, %464 ], [ 0, %._crit_edge.105 ]
  %467 = bitcast i32 %466 to float
  %.sroa.65.420.vec.insert = insertelement <64 x float> %.sroa.65.416.vec.insert, float %467, i32 41
  br i1 %36, label %468, label %._crit_edge.106

._crit_edge.106:                                  ; preds = %._crit_edge96.105
  br label %._crit_edge96.106

468:                                              ; preds = %._crit_edge96.105
  %469 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.106

._crit_edge96.106:                                ; preds = %468, %._crit_edge.106
  %470 = phi i32 [ %469, %468 ], [ 0, %._crit_edge.106 ]
  %471 = bitcast i32 %470 to float
  %.sroa.65.424.vec.insert = insertelement <64 x float> %.sroa.65.420.vec.insert, float %471, i32 42
  br i1 %36, label %472, label %._crit_edge.107

._crit_edge.107:                                  ; preds = %._crit_edge96.106
  br label %._crit_edge96.107

472:                                              ; preds = %._crit_edge96.106
  %473 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.107

._crit_edge96.107:                                ; preds = %472, %._crit_edge.107
  %474 = phi i32 [ %473, %472 ], [ 0, %._crit_edge.107 ]
  %475 = bitcast i32 %474 to float
  %.sroa.65.428.vec.insert = insertelement <64 x float> %.sroa.65.424.vec.insert, float %475, i32 43
  br i1 %36, label %476, label %._crit_edge.108

._crit_edge.108:                                  ; preds = %._crit_edge96.107
  br label %._crit_edge96.108

476:                                              ; preds = %._crit_edge96.107
  %477 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.108

._crit_edge96.108:                                ; preds = %476, %._crit_edge.108
  %478 = phi i32 [ %477, %476 ], [ 0, %._crit_edge.108 ]
  %479 = bitcast i32 %478 to float
  %.sroa.65.432.vec.insert = insertelement <64 x float> %.sroa.65.428.vec.insert, float %479, i32 44
  br i1 %36, label %480, label %._crit_edge.109

._crit_edge.109:                                  ; preds = %._crit_edge96.108
  br label %._crit_edge96.109

480:                                              ; preds = %._crit_edge96.108
  %481 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.109

._crit_edge96.109:                                ; preds = %480, %._crit_edge.109
  %482 = phi i32 [ %481, %480 ], [ 0, %._crit_edge.109 ]
  %483 = bitcast i32 %482 to float
  %.sroa.65.436.vec.insert = insertelement <64 x float> %.sroa.65.432.vec.insert, float %483, i32 45
  br i1 %36, label %484, label %._crit_edge.110

._crit_edge.110:                                  ; preds = %._crit_edge96.109
  br label %._crit_edge96.110

484:                                              ; preds = %._crit_edge96.109
  %485 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.110

._crit_edge96.110:                                ; preds = %484, %._crit_edge.110
  %486 = phi i32 [ %485, %484 ], [ 0, %._crit_edge.110 ]
  %487 = bitcast i32 %486 to float
  %.sroa.65.440.vec.insert = insertelement <64 x float> %.sroa.65.436.vec.insert, float %487, i32 46
  br i1 %36, label %488, label %._crit_edge.111

._crit_edge.111:                                  ; preds = %._crit_edge96.110
  br label %._crit_edge96.111

488:                                              ; preds = %._crit_edge96.110
  %489 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.111

._crit_edge96.111:                                ; preds = %488, %._crit_edge.111
  %490 = phi i32 [ %489, %488 ], [ 0, %._crit_edge.111 ]
  %491 = bitcast i32 %490 to float
  %.sroa.65.444.vec.insert = insertelement <64 x float> %.sroa.65.440.vec.insert, float %491, i32 47
  br i1 %36, label %492, label %._crit_edge.112

._crit_edge.112:                                  ; preds = %._crit_edge96.111
  br label %._crit_edge96.112

492:                                              ; preds = %._crit_edge96.111
  %493 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.112

._crit_edge96.112:                                ; preds = %492, %._crit_edge.112
  %494 = phi i32 [ %493, %492 ], [ 0, %._crit_edge.112 ]
  %495 = bitcast i32 %494 to float
  %.sroa.65.448.vec.insert = insertelement <64 x float> %.sroa.65.444.vec.insert, float %495, i32 48
  br i1 %36, label %496, label %._crit_edge.113

._crit_edge.113:                                  ; preds = %._crit_edge96.112
  br label %._crit_edge96.113

496:                                              ; preds = %._crit_edge96.112
  %497 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.113

._crit_edge96.113:                                ; preds = %496, %._crit_edge.113
  %498 = phi i32 [ %497, %496 ], [ 0, %._crit_edge.113 ]
  %499 = bitcast i32 %498 to float
  %.sroa.65.452.vec.insert = insertelement <64 x float> %.sroa.65.448.vec.insert, float %499, i32 49
  br i1 %36, label %500, label %._crit_edge.114

._crit_edge.114:                                  ; preds = %._crit_edge96.113
  br label %._crit_edge96.114

500:                                              ; preds = %._crit_edge96.113
  %501 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.114

._crit_edge96.114:                                ; preds = %500, %._crit_edge.114
  %502 = phi i32 [ %501, %500 ], [ 0, %._crit_edge.114 ]
  %503 = bitcast i32 %502 to float
  %.sroa.65.456.vec.insert = insertelement <64 x float> %.sroa.65.452.vec.insert, float %503, i32 50
  br i1 %36, label %504, label %._crit_edge.115

._crit_edge.115:                                  ; preds = %._crit_edge96.114
  br label %._crit_edge96.115

504:                                              ; preds = %._crit_edge96.114
  %505 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.115

._crit_edge96.115:                                ; preds = %504, %._crit_edge.115
  %506 = phi i32 [ %505, %504 ], [ 0, %._crit_edge.115 ]
  %507 = bitcast i32 %506 to float
  %.sroa.65.460.vec.insert = insertelement <64 x float> %.sroa.65.456.vec.insert, float %507, i32 51
  br i1 %36, label %508, label %._crit_edge.116

._crit_edge.116:                                  ; preds = %._crit_edge96.115
  br label %._crit_edge96.116

508:                                              ; preds = %._crit_edge96.115
  %509 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.116

._crit_edge96.116:                                ; preds = %508, %._crit_edge.116
  %510 = phi i32 [ %509, %508 ], [ 0, %._crit_edge.116 ]
  %511 = bitcast i32 %510 to float
  %.sroa.65.464.vec.insert = insertelement <64 x float> %.sroa.65.460.vec.insert, float %511, i32 52
  br i1 %36, label %512, label %._crit_edge.117

._crit_edge.117:                                  ; preds = %._crit_edge96.116
  br label %._crit_edge96.117

512:                                              ; preds = %._crit_edge96.116
  %513 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.117

._crit_edge96.117:                                ; preds = %512, %._crit_edge.117
  %514 = phi i32 [ %513, %512 ], [ 0, %._crit_edge.117 ]
  %515 = bitcast i32 %514 to float
  %.sroa.65.468.vec.insert = insertelement <64 x float> %.sroa.65.464.vec.insert, float %515, i32 53
  br i1 %36, label %516, label %._crit_edge.118

._crit_edge.118:                                  ; preds = %._crit_edge96.117
  br label %._crit_edge96.118

516:                                              ; preds = %._crit_edge96.117
  %517 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.118

._crit_edge96.118:                                ; preds = %516, %._crit_edge.118
  %518 = phi i32 [ %517, %516 ], [ 0, %._crit_edge.118 ]
  %519 = bitcast i32 %518 to float
  %.sroa.65.472.vec.insert = insertelement <64 x float> %.sroa.65.468.vec.insert, float %519, i32 54
  br i1 %36, label %520, label %._crit_edge.119

._crit_edge.119:                                  ; preds = %._crit_edge96.118
  br label %._crit_edge96.119

520:                                              ; preds = %._crit_edge96.118
  %521 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.119

._crit_edge96.119:                                ; preds = %520, %._crit_edge.119
  %522 = phi i32 [ %521, %520 ], [ 0, %._crit_edge.119 ]
  %523 = bitcast i32 %522 to float
  %.sroa.65.476.vec.insert = insertelement <64 x float> %.sroa.65.472.vec.insert, float %523, i32 55
  br i1 %36, label %524, label %._crit_edge.120

._crit_edge.120:                                  ; preds = %._crit_edge96.119
  br label %._crit_edge96.120

524:                                              ; preds = %._crit_edge96.119
  %525 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.120

._crit_edge96.120:                                ; preds = %524, %._crit_edge.120
  %526 = phi i32 [ %525, %524 ], [ 0, %._crit_edge.120 ]
  %527 = bitcast i32 %526 to float
  %.sroa.65.480.vec.insert = insertelement <64 x float> %.sroa.65.476.vec.insert, float %527, i32 56
  br i1 %36, label %528, label %._crit_edge.121

._crit_edge.121:                                  ; preds = %._crit_edge96.120
  br label %._crit_edge96.121

528:                                              ; preds = %._crit_edge96.120
  %529 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.121

._crit_edge96.121:                                ; preds = %528, %._crit_edge.121
  %530 = phi i32 [ %529, %528 ], [ 0, %._crit_edge.121 ]
  %531 = bitcast i32 %530 to float
  %.sroa.65.484.vec.insert = insertelement <64 x float> %.sroa.65.480.vec.insert, float %531, i32 57
  br i1 %36, label %532, label %._crit_edge.122

._crit_edge.122:                                  ; preds = %._crit_edge96.121
  br label %._crit_edge96.122

532:                                              ; preds = %._crit_edge96.121
  %533 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.122

._crit_edge96.122:                                ; preds = %532, %._crit_edge.122
  %534 = phi i32 [ %533, %532 ], [ 0, %._crit_edge.122 ]
  %535 = bitcast i32 %534 to float
  %.sroa.65.488.vec.insert = insertelement <64 x float> %.sroa.65.484.vec.insert, float %535, i32 58
  br i1 %36, label %536, label %._crit_edge.123

._crit_edge.123:                                  ; preds = %._crit_edge96.122
  br label %._crit_edge96.123

536:                                              ; preds = %._crit_edge96.122
  %537 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.123

._crit_edge96.123:                                ; preds = %536, %._crit_edge.123
  %538 = phi i32 [ %537, %536 ], [ 0, %._crit_edge.123 ]
  %539 = bitcast i32 %538 to float
  %.sroa.65.492.vec.insert = insertelement <64 x float> %.sroa.65.488.vec.insert, float %539, i32 59
  br i1 %36, label %540, label %._crit_edge.124

._crit_edge.124:                                  ; preds = %._crit_edge96.123
  br label %._crit_edge96.124

540:                                              ; preds = %._crit_edge96.123
  %541 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.124

._crit_edge96.124:                                ; preds = %540, %._crit_edge.124
  %542 = phi i32 [ %541, %540 ], [ 0, %._crit_edge.124 ]
  %543 = bitcast i32 %542 to float
  %.sroa.65.496.vec.insert = insertelement <64 x float> %.sroa.65.492.vec.insert, float %543, i32 60
  br i1 %36, label %544, label %._crit_edge.125

._crit_edge.125:                                  ; preds = %._crit_edge96.124
  br label %._crit_edge96.125

544:                                              ; preds = %._crit_edge96.124
  %545 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.125

._crit_edge96.125:                                ; preds = %544, %._crit_edge.125
  %546 = phi i32 [ %545, %544 ], [ 0, %._crit_edge.125 ]
  %547 = bitcast i32 %546 to float
  %.sroa.65.500.vec.insert = insertelement <64 x float> %.sroa.65.496.vec.insert, float %547, i32 61
  br i1 %36, label %548, label %._crit_edge.126

._crit_edge.126:                                  ; preds = %._crit_edge96.125
  br label %._crit_edge96.126

548:                                              ; preds = %._crit_edge96.125
  %549 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.126

._crit_edge96.126:                                ; preds = %548, %._crit_edge.126
  %550 = phi i32 [ %549, %548 ], [ 0, %._crit_edge.126 ]
  %551 = bitcast i32 %550 to float
  %.sroa.65.504.vec.insert = insertelement <64 x float> %.sroa.65.500.vec.insert, float %551, i32 62
  br i1 %36, label %552, label %._crit_edge.127

._crit_edge.127:                                  ; preds = %._crit_edge96.126
  br label %._crit_edge96.127

552:                                              ; preds = %._crit_edge96.126
  %553 = load i32, i32 addrspace(1)* %42, align 4, !tbaa !521
  br label %._crit_edge96.127

._crit_edge96.127:                                ; preds = %552, %._crit_edge.127
  %554 = phi i32 [ %553, %552 ], [ 0, %._crit_edge.127 ]
  %555 = bitcast i32 %554 to float
  %.sroa.65.508.vec.insert = insertelement <64 x float> %.sroa.65.504.vec.insert, float %555, i32 63
  br label %_Z18get_sub_group_sizev.exit.i

_Z18get_sub_group_sizev.exit.i:                   ; preds = %._crit_edge96.127
  %556 = shl nuw nsw i32 %simdLaneId, 1
  %557 = and i32 %556, 131008
  %558 = or i32 %557, %35
  %559 = zext i32 %558 to i64
  %560 = getelementptr inbounds float, float addrspace(1)* %34, i64 %559
  %561 = bitcast float addrspace(1)* %560 to i32 addrspace(1)*
  br label %562

562:                                              ; preds = %_Z18get_sub_group_sizev.exit.i
  br i1 %36, label %563, label %._crit_edge97

._crit_edge97:                                    ; preds = %562
  br label %._crit_edge98

563:                                              ; preds = %562
  %.sroa.0101.0.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 0
  %564 = bitcast float %.sroa.0101.0.vec.extract to i32
  store i32 %564, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98

._crit_edge98:                                    ; preds = %._crit_edge97, %563
  br i1 %36, label %565, label %._crit_edge97.1

._crit_edge97.1:                                  ; preds = %._crit_edge98
  br label %._crit_edge98.1

565:                                              ; preds = %._crit_edge98
  %.sroa.0101.4.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 1
  %566 = bitcast float %.sroa.0101.4.vec.extract to i32
  store i32 %566, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.1

._crit_edge98.1:                                  ; preds = %565, %._crit_edge97.1
  br i1 %36, label %567, label %._crit_edge97.2

._crit_edge97.2:                                  ; preds = %._crit_edge98.1
  br label %._crit_edge98.2

567:                                              ; preds = %._crit_edge98.1
  %.sroa.0101.8.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 2
  %568 = bitcast float %.sroa.0101.8.vec.extract to i32
  store i32 %568, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.2

._crit_edge98.2:                                  ; preds = %567, %._crit_edge97.2
  br i1 %36, label %569, label %._crit_edge97.3

._crit_edge97.3:                                  ; preds = %._crit_edge98.2
  br label %._crit_edge98.3

569:                                              ; preds = %._crit_edge98.2
  %.sroa.0101.12.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 3
  %570 = bitcast float %.sroa.0101.12.vec.extract to i32
  store i32 %570, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.3

._crit_edge98.3:                                  ; preds = %569, %._crit_edge97.3
  br i1 %36, label %571, label %._crit_edge97.4

._crit_edge97.4:                                  ; preds = %._crit_edge98.3
  br label %._crit_edge98.4

571:                                              ; preds = %._crit_edge98.3
  %.sroa.0101.16.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 4
  %572 = bitcast float %.sroa.0101.16.vec.extract to i32
  store i32 %572, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.4

._crit_edge98.4:                                  ; preds = %571, %._crit_edge97.4
  br i1 %36, label %573, label %._crit_edge97.5

._crit_edge97.5:                                  ; preds = %._crit_edge98.4
  br label %._crit_edge98.5

573:                                              ; preds = %._crit_edge98.4
  %.sroa.0101.20.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 5
  %574 = bitcast float %.sroa.0101.20.vec.extract to i32
  store i32 %574, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.5

._crit_edge98.5:                                  ; preds = %573, %._crit_edge97.5
  br i1 %36, label %575, label %._crit_edge97.6

._crit_edge97.6:                                  ; preds = %._crit_edge98.5
  br label %._crit_edge98.6

575:                                              ; preds = %._crit_edge98.5
  %.sroa.0101.24.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 6
  %576 = bitcast float %.sroa.0101.24.vec.extract to i32
  store i32 %576, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.6

._crit_edge98.6:                                  ; preds = %575, %._crit_edge97.6
  br i1 %36, label %577, label %._crit_edge97.7

._crit_edge97.7:                                  ; preds = %._crit_edge98.6
  br label %._crit_edge98.7

577:                                              ; preds = %._crit_edge98.6
  %.sroa.0101.28.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 7
  %578 = bitcast float %.sroa.0101.28.vec.extract to i32
  store i32 %578, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.7

._crit_edge98.7:                                  ; preds = %577, %._crit_edge97.7
  br i1 %36, label %579, label %._crit_edge97.8

._crit_edge97.8:                                  ; preds = %._crit_edge98.7
  br label %._crit_edge98.8

579:                                              ; preds = %._crit_edge98.7
  %.sroa.0101.32.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 8
  %580 = bitcast float %.sroa.0101.32.vec.extract to i32
  store i32 %580, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.8

._crit_edge98.8:                                  ; preds = %579, %._crit_edge97.8
  br i1 %36, label %581, label %._crit_edge97.9

._crit_edge97.9:                                  ; preds = %._crit_edge98.8
  br label %._crit_edge98.9

581:                                              ; preds = %._crit_edge98.8
  %.sroa.0101.36.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 9
  %582 = bitcast float %.sroa.0101.36.vec.extract to i32
  store i32 %582, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.9

._crit_edge98.9:                                  ; preds = %581, %._crit_edge97.9
  br i1 %36, label %583, label %._crit_edge97.10

._crit_edge97.10:                                 ; preds = %._crit_edge98.9
  br label %._crit_edge98.10

583:                                              ; preds = %._crit_edge98.9
  %.sroa.0101.40.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 10
  %584 = bitcast float %.sroa.0101.40.vec.extract to i32
  store i32 %584, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.10

._crit_edge98.10:                                 ; preds = %583, %._crit_edge97.10
  br i1 %36, label %585, label %._crit_edge97.11

._crit_edge97.11:                                 ; preds = %._crit_edge98.10
  br label %._crit_edge98.11

585:                                              ; preds = %._crit_edge98.10
  %.sroa.0101.44.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 11
  %586 = bitcast float %.sroa.0101.44.vec.extract to i32
  store i32 %586, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.11

._crit_edge98.11:                                 ; preds = %585, %._crit_edge97.11
  br i1 %36, label %587, label %._crit_edge97.12

._crit_edge97.12:                                 ; preds = %._crit_edge98.11
  br label %._crit_edge98.12

587:                                              ; preds = %._crit_edge98.11
  %.sroa.0101.48.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 12
  %588 = bitcast float %.sroa.0101.48.vec.extract to i32
  store i32 %588, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.12

._crit_edge98.12:                                 ; preds = %587, %._crit_edge97.12
  br i1 %36, label %589, label %._crit_edge97.13

._crit_edge97.13:                                 ; preds = %._crit_edge98.12
  br label %._crit_edge98.13

589:                                              ; preds = %._crit_edge98.12
  %.sroa.0101.52.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 13
  %590 = bitcast float %.sroa.0101.52.vec.extract to i32
  store i32 %590, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.13

._crit_edge98.13:                                 ; preds = %589, %._crit_edge97.13
  br i1 %36, label %591, label %._crit_edge97.14

._crit_edge97.14:                                 ; preds = %._crit_edge98.13
  br label %._crit_edge98.14

591:                                              ; preds = %._crit_edge98.13
  %.sroa.0101.56.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 14
  %592 = bitcast float %.sroa.0101.56.vec.extract to i32
  store i32 %592, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.14

._crit_edge98.14:                                 ; preds = %591, %._crit_edge97.14
  br i1 %36, label %593, label %._crit_edge97.15

._crit_edge97.15:                                 ; preds = %._crit_edge98.14
  br label %._crit_edge98.15

593:                                              ; preds = %._crit_edge98.14
  %.sroa.0101.60.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 15
  %594 = bitcast float %.sroa.0101.60.vec.extract to i32
  store i32 %594, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.15

._crit_edge98.15:                                 ; preds = %593, %._crit_edge97.15
  br i1 %36, label %595, label %._crit_edge97.16

._crit_edge97.16:                                 ; preds = %._crit_edge98.15
  br label %._crit_edge98.16

595:                                              ; preds = %._crit_edge98.15
  %.sroa.0101.64.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 16
  %596 = bitcast float %.sroa.0101.64.vec.extract to i32
  store i32 %596, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.16

._crit_edge98.16:                                 ; preds = %595, %._crit_edge97.16
  br i1 %36, label %597, label %._crit_edge97.17

._crit_edge97.17:                                 ; preds = %._crit_edge98.16
  br label %._crit_edge98.17

597:                                              ; preds = %._crit_edge98.16
  %.sroa.0101.68.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 17
  %598 = bitcast float %.sroa.0101.68.vec.extract to i32
  store i32 %598, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.17

._crit_edge98.17:                                 ; preds = %597, %._crit_edge97.17
  br i1 %36, label %599, label %._crit_edge97.18

._crit_edge97.18:                                 ; preds = %._crit_edge98.17
  br label %._crit_edge98.18

599:                                              ; preds = %._crit_edge98.17
  %.sroa.0101.72.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 18
  %600 = bitcast float %.sroa.0101.72.vec.extract to i32
  store i32 %600, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.18

._crit_edge98.18:                                 ; preds = %599, %._crit_edge97.18
  br i1 %36, label %601, label %._crit_edge97.19

._crit_edge97.19:                                 ; preds = %._crit_edge98.18
  br label %._crit_edge98.19

601:                                              ; preds = %._crit_edge98.18
  %.sroa.0101.76.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 19
  %602 = bitcast float %.sroa.0101.76.vec.extract to i32
  store i32 %602, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.19

._crit_edge98.19:                                 ; preds = %601, %._crit_edge97.19
  br i1 %36, label %603, label %._crit_edge97.20

._crit_edge97.20:                                 ; preds = %._crit_edge98.19
  br label %._crit_edge98.20

603:                                              ; preds = %._crit_edge98.19
  %.sroa.0101.80.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 20
  %604 = bitcast float %.sroa.0101.80.vec.extract to i32
  store i32 %604, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.20

._crit_edge98.20:                                 ; preds = %603, %._crit_edge97.20
  br i1 %36, label %605, label %._crit_edge97.21

._crit_edge97.21:                                 ; preds = %._crit_edge98.20
  br label %._crit_edge98.21

605:                                              ; preds = %._crit_edge98.20
  %.sroa.0101.84.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 21
  %606 = bitcast float %.sroa.0101.84.vec.extract to i32
  store i32 %606, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.21

._crit_edge98.21:                                 ; preds = %605, %._crit_edge97.21
  br i1 %36, label %607, label %._crit_edge97.22

._crit_edge97.22:                                 ; preds = %._crit_edge98.21
  br label %._crit_edge98.22

607:                                              ; preds = %._crit_edge98.21
  %.sroa.0101.88.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 22
  %608 = bitcast float %.sroa.0101.88.vec.extract to i32
  store i32 %608, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.22

._crit_edge98.22:                                 ; preds = %607, %._crit_edge97.22
  br i1 %36, label %609, label %._crit_edge97.23

._crit_edge97.23:                                 ; preds = %._crit_edge98.22
  br label %._crit_edge98.23

609:                                              ; preds = %._crit_edge98.22
  %.sroa.0101.92.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 23
  %610 = bitcast float %.sroa.0101.92.vec.extract to i32
  store i32 %610, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.23

._crit_edge98.23:                                 ; preds = %609, %._crit_edge97.23
  br i1 %36, label %611, label %._crit_edge97.24

._crit_edge97.24:                                 ; preds = %._crit_edge98.23
  br label %._crit_edge98.24

611:                                              ; preds = %._crit_edge98.23
  %.sroa.0101.96.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 24
  %612 = bitcast float %.sroa.0101.96.vec.extract to i32
  store i32 %612, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.24

._crit_edge98.24:                                 ; preds = %611, %._crit_edge97.24
  br i1 %36, label %613, label %._crit_edge97.25

._crit_edge97.25:                                 ; preds = %._crit_edge98.24
  br label %._crit_edge98.25

613:                                              ; preds = %._crit_edge98.24
  %.sroa.0101.100.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 25
  %614 = bitcast float %.sroa.0101.100.vec.extract to i32
  store i32 %614, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.25

._crit_edge98.25:                                 ; preds = %613, %._crit_edge97.25
  br i1 %36, label %615, label %._crit_edge97.26

._crit_edge97.26:                                 ; preds = %._crit_edge98.25
  br label %._crit_edge98.26

615:                                              ; preds = %._crit_edge98.25
  %.sroa.0101.104.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 26
  %616 = bitcast float %.sroa.0101.104.vec.extract to i32
  store i32 %616, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.26

._crit_edge98.26:                                 ; preds = %615, %._crit_edge97.26
  br i1 %36, label %617, label %._crit_edge97.27

._crit_edge97.27:                                 ; preds = %._crit_edge98.26
  br label %._crit_edge98.27

617:                                              ; preds = %._crit_edge98.26
  %.sroa.0101.108.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 27
  %618 = bitcast float %.sroa.0101.108.vec.extract to i32
  store i32 %618, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.27

._crit_edge98.27:                                 ; preds = %617, %._crit_edge97.27
  br i1 %36, label %619, label %._crit_edge97.28

._crit_edge97.28:                                 ; preds = %._crit_edge98.27
  br label %._crit_edge98.28

619:                                              ; preds = %._crit_edge98.27
  %.sroa.0101.112.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 28
  %620 = bitcast float %.sroa.0101.112.vec.extract to i32
  store i32 %620, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.28

._crit_edge98.28:                                 ; preds = %619, %._crit_edge97.28
  br i1 %36, label %621, label %._crit_edge97.29

._crit_edge97.29:                                 ; preds = %._crit_edge98.28
  br label %._crit_edge98.29

621:                                              ; preds = %._crit_edge98.28
  %.sroa.0101.116.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 29
  %622 = bitcast float %.sroa.0101.116.vec.extract to i32
  store i32 %622, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.29

._crit_edge98.29:                                 ; preds = %621, %._crit_edge97.29
  br i1 %36, label %623, label %._crit_edge97.30

._crit_edge97.30:                                 ; preds = %._crit_edge98.29
  br label %._crit_edge98.30

623:                                              ; preds = %._crit_edge98.29
  %.sroa.0101.120.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 30
  %624 = bitcast float %.sroa.0101.120.vec.extract to i32
  store i32 %624, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.30

._crit_edge98.30:                                 ; preds = %623, %._crit_edge97.30
  br i1 %36, label %625, label %._crit_edge97.31

._crit_edge97.31:                                 ; preds = %._crit_edge98.30
  br label %._crit_edge98.31

625:                                              ; preds = %._crit_edge98.30
  %.sroa.0101.124.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 31
  %626 = bitcast float %.sroa.0101.124.vec.extract to i32
  store i32 %626, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.31

._crit_edge98.31:                                 ; preds = %625, %._crit_edge97.31
  br i1 %36, label %627, label %._crit_edge97.32

._crit_edge97.32:                                 ; preds = %._crit_edge98.31
  br label %._crit_edge98.32

627:                                              ; preds = %._crit_edge98.31
  %.sroa.0101.128.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 32
  %628 = bitcast float %.sroa.0101.128.vec.extract to i32
  store i32 %628, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.32

._crit_edge98.32:                                 ; preds = %627, %._crit_edge97.32
  br i1 %36, label %629, label %._crit_edge97.33

._crit_edge97.33:                                 ; preds = %._crit_edge98.32
  br label %._crit_edge98.33

629:                                              ; preds = %._crit_edge98.32
  %.sroa.0101.132.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 33
  %630 = bitcast float %.sroa.0101.132.vec.extract to i32
  store i32 %630, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.33

._crit_edge98.33:                                 ; preds = %629, %._crit_edge97.33
  br i1 %36, label %631, label %._crit_edge97.34

._crit_edge97.34:                                 ; preds = %._crit_edge98.33
  br label %._crit_edge98.34

631:                                              ; preds = %._crit_edge98.33
  %.sroa.0101.136.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 34
  %632 = bitcast float %.sroa.0101.136.vec.extract to i32
  store i32 %632, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.34

._crit_edge98.34:                                 ; preds = %631, %._crit_edge97.34
  br i1 %36, label %633, label %._crit_edge97.35

._crit_edge97.35:                                 ; preds = %._crit_edge98.34
  br label %._crit_edge98.35

633:                                              ; preds = %._crit_edge98.34
  %.sroa.0101.140.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 35
  %634 = bitcast float %.sroa.0101.140.vec.extract to i32
  store i32 %634, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.35

._crit_edge98.35:                                 ; preds = %633, %._crit_edge97.35
  br i1 %36, label %635, label %._crit_edge97.36

._crit_edge97.36:                                 ; preds = %._crit_edge98.35
  br label %._crit_edge98.36

635:                                              ; preds = %._crit_edge98.35
  %.sroa.0101.144.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 36
  %636 = bitcast float %.sroa.0101.144.vec.extract to i32
  store i32 %636, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.36

._crit_edge98.36:                                 ; preds = %635, %._crit_edge97.36
  br i1 %36, label %637, label %._crit_edge97.37

._crit_edge97.37:                                 ; preds = %._crit_edge98.36
  br label %._crit_edge98.37

637:                                              ; preds = %._crit_edge98.36
  %.sroa.0101.148.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 37
  %638 = bitcast float %.sroa.0101.148.vec.extract to i32
  store i32 %638, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.37

._crit_edge98.37:                                 ; preds = %637, %._crit_edge97.37
  br i1 %36, label %639, label %._crit_edge97.38

._crit_edge97.38:                                 ; preds = %._crit_edge98.37
  br label %._crit_edge98.38

639:                                              ; preds = %._crit_edge98.37
  %.sroa.0101.152.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 38
  %640 = bitcast float %.sroa.0101.152.vec.extract to i32
  store i32 %640, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.38

._crit_edge98.38:                                 ; preds = %639, %._crit_edge97.38
  br i1 %36, label %641, label %._crit_edge97.39

._crit_edge97.39:                                 ; preds = %._crit_edge98.38
  br label %._crit_edge98.39

641:                                              ; preds = %._crit_edge98.38
  %.sroa.0101.156.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 39
  %642 = bitcast float %.sroa.0101.156.vec.extract to i32
  store i32 %642, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.39

._crit_edge98.39:                                 ; preds = %641, %._crit_edge97.39
  br i1 %36, label %643, label %._crit_edge97.40

._crit_edge97.40:                                 ; preds = %._crit_edge98.39
  br label %._crit_edge98.40

643:                                              ; preds = %._crit_edge98.39
  %.sroa.0101.160.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 40
  %644 = bitcast float %.sroa.0101.160.vec.extract to i32
  store i32 %644, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.40

._crit_edge98.40:                                 ; preds = %643, %._crit_edge97.40
  br i1 %36, label %645, label %._crit_edge97.41

._crit_edge97.41:                                 ; preds = %._crit_edge98.40
  br label %._crit_edge98.41

645:                                              ; preds = %._crit_edge98.40
  %.sroa.0101.164.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 41
  %646 = bitcast float %.sroa.0101.164.vec.extract to i32
  store i32 %646, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.41

._crit_edge98.41:                                 ; preds = %645, %._crit_edge97.41
  br i1 %36, label %647, label %._crit_edge97.42

._crit_edge97.42:                                 ; preds = %._crit_edge98.41
  br label %._crit_edge98.42

647:                                              ; preds = %._crit_edge98.41
  %.sroa.0101.168.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 42
  %648 = bitcast float %.sroa.0101.168.vec.extract to i32
  store i32 %648, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.42

._crit_edge98.42:                                 ; preds = %647, %._crit_edge97.42
  br i1 %36, label %649, label %._crit_edge97.43

._crit_edge97.43:                                 ; preds = %._crit_edge98.42
  br label %._crit_edge98.43

649:                                              ; preds = %._crit_edge98.42
  %.sroa.0101.172.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 43
  %650 = bitcast float %.sroa.0101.172.vec.extract to i32
  store i32 %650, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.43

._crit_edge98.43:                                 ; preds = %649, %._crit_edge97.43
  br i1 %36, label %651, label %._crit_edge97.44

._crit_edge97.44:                                 ; preds = %._crit_edge98.43
  br label %._crit_edge98.44

651:                                              ; preds = %._crit_edge98.43
  %.sroa.0101.176.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 44
  %652 = bitcast float %.sroa.0101.176.vec.extract to i32
  store i32 %652, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.44

._crit_edge98.44:                                 ; preds = %651, %._crit_edge97.44
  br i1 %36, label %653, label %._crit_edge97.45

._crit_edge97.45:                                 ; preds = %._crit_edge98.44
  br label %._crit_edge98.45

653:                                              ; preds = %._crit_edge98.44
  %.sroa.0101.180.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 45
  %654 = bitcast float %.sroa.0101.180.vec.extract to i32
  store i32 %654, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.45

._crit_edge98.45:                                 ; preds = %653, %._crit_edge97.45
  br i1 %36, label %655, label %._crit_edge97.46

._crit_edge97.46:                                 ; preds = %._crit_edge98.45
  br label %._crit_edge98.46

655:                                              ; preds = %._crit_edge98.45
  %.sroa.0101.184.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 46
  %656 = bitcast float %.sroa.0101.184.vec.extract to i32
  store i32 %656, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.46

._crit_edge98.46:                                 ; preds = %655, %._crit_edge97.46
  br i1 %36, label %657, label %._crit_edge97.47

._crit_edge97.47:                                 ; preds = %._crit_edge98.46
  br label %._crit_edge98.47

657:                                              ; preds = %._crit_edge98.46
  %.sroa.0101.188.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 47
  %658 = bitcast float %.sroa.0101.188.vec.extract to i32
  store i32 %658, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.47

._crit_edge98.47:                                 ; preds = %657, %._crit_edge97.47
  br i1 %36, label %659, label %._crit_edge97.48

._crit_edge97.48:                                 ; preds = %._crit_edge98.47
  br label %._crit_edge98.48

659:                                              ; preds = %._crit_edge98.47
  %.sroa.0101.192.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 48
  %660 = bitcast float %.sroa.0101.192.vec.extract to i32
  store i32 %660, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.48

._crit_edge98.48:                                 ; preds = %659, %._crit_edge97.48
  br i1 %36, label %661, label %._crit_edge97.49

._crit_edge97.49:                                 ; preds = %._crit_edge98.48
  br label %._crit_edge98.49

661:                                              ; preds = %._crit_edge98.48
  %.sroa.0101.196.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 49
  %662 = bitcast float %.sroa.0101.196.vec.extract to i32
  store i32 %662, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.49

._crit_edge98.49:                                 ; preds = %661, %._crit_edge97.49
  br i1 %36, label %663, label %._crit_edge97.50

._crit_edge97.50:                                 ; preds = %._crit_edge98.49
  br label %._crit_edge98.50

663:                                              ; preds = %._crit_edge98.49
  %.sroa.0101.200.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 50
  %664 = bitcast float %.sroa.0101.200.vec.extract to i32
  store i32 %664, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.50

._crit_edge98.50:                                 ; preds = %663, %._crit_edge97.50
  br i1 %36, label %665, label %._crit_edge97.51

._crit_edge97.51:                                 ; preds = %._crit_edge98.50
  br label %._crit_edge98.51

665:                                              ; preds = %._crit_edge98.50
  %.sroa.0101.204.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 51
  %666 = bitcast float %.sroa.0101.204.vec.extract to i32
  store i32 %666, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.51

._crit_edge98.51:                                 ; preds = %665, %._crit_edge97.51
  br i1 %36, label %667, label %._crit_edge97.52

._crit_edge97.52:                                 ; preds = %._crit_edge98.51
  br label %._crit_edge98.52

667:                                              ; preds = %._crit_edge98.51
  %.sroa.0101.208.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 52
  %668 = bitcast float %.sroa.0101.208.vec.extract to i32
  store i32 %668, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.52

._crit_edge98.52:                                 ; preds = %667, %._crit_edge97.52
  br i1 %36, label %669, label %._crit_edge97.53

._crit_edge97.53:                                 ; preds = %._crit_edge98.52
  br label %._crit_edge98.53

669:                                              ; preds = %._crit_edge98.52
  %.sroa.0101.212.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 53
  %670 = bitcast float %.sroa.0101.212.vec.extract to i32
  store i32 %670, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.53

._crit_edge98.53:                                 ; preds = %669, %._crit_edge97.53
  br i1 %36, label %671, label %._crit_edge97.54

._crit_edge97.54:                                 ; preds = %._crit_edge98.53
  br label %._crit_edge98.54

671:                                              ; preds = %._crit_edge98.53
  %.sroa.0101.216.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 54
  %672 = bitcast float %.sroa.0101.216.vec.extract to i32
  store i32 %672, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.54

._crit_edge98.54:                                 ; preds = %671, %._crit_edge97.54
  br i1 %36, label %673, label %._crit_edge97.55

._crit_edge97.55:                                 ; preds = %._crit_edge98.54
  br label %._crit_edge98.55

673:                                              ; preds = %._crit_edge98.54
  %.sroa.0101.220.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 55
  %674 = bitcast float %.sroa.0101.220.vec.extract to i32
  store i32 %674, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.55

._crit_edge98.55:                                 ; preds = %673, %._crit_edge97.55
  br i1 %36, label %675, label %._crit_edge97.56

._crit_edge97.56:                                 ; preds = %._crit_edge98.55
  br label %._crit_edge98.56

675:                                              ; preds = %._crit_edge98.55
  %.sroa.0101.224.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 56
  %676 = bitcast float %.sroa.0101.224.vec.extract to i32
  store i32 %676, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.56

._crit_edge98.56:                                 ; preds = %675, %._crit_edge97.56
  br i1 %36, label %677, label %._crit_edge97.57

._crit_edge97.57:                                 ; preds = %._crit_edge98.56
  br label %._crit_edge98.57

677:                                              ; preds = %._crit_edge98.56
  %.sroa.0101.228.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 57
  %678 = bitcast float %.sroa.0101.228.vec.extract to i32
  store i32 %678, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.57

._crit_edge98.57:                                 ; preds = %677, %._crit_edge97.57
  br i1 %36, label %679, label %._crit_edge97.58

._crit_edge97.58:                                 ; preds = %._crit_edge98.57
  br label %._crit_edge98.58

679:                                              ; preds = %._crit_edge98.57
  %.sroa.0101.232.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 58
  %680 = bitcast float %.sroa.0101.232.vec.extract to i32
  store i32 %680, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.58

._crit_edge98.58:                                 ; preds = %679, %._crit_edge97.58
  br i1 %36, label %681, label %._crit_edge97.59

._crit_edge97.59:                                 ; preds = %._crit_edge98.58
  br label %._crit_edge98.59

681:                                              ; preds = %._crit_edge98.58
  %.sroa.0101.236.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 59
  %682 = bitcast float %.sroa.0101.236.vec.extract to i32
  store i32 %682, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.59

._crit_edge98.59:                                 ; preds = %681, %._crit_edge97.59
  br i1 %36, label %683, label %._crit_edge97.60

._crit_edge97.60:                                 ; preds = %._crit_edge98.59
  br label %._crit_edge98.60

683:                                              ; preds = %._crit_edge98.59
  %.sroa.0101.240.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 60
  %684 = bitcast float %.sroa.0101.240.vec.extract to i32
  store i32 %684, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.60

._crit_edge98.60:                                 ; preds = %683, %._crit_edge97.60
  br i1 %36, label %685, label %._crit_edge97.61

._crit_edge97.61:                                 ; preds = %._crit_edge98.60
  br label %._crit_edge98.61

685:                                              ; preds = %._crit_edge98.60
  %.sroa.0101.244.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 61
  %686 = bitcast float %.sroa.0101.244.vec.extract to i32
  store i32 %686, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.61

._crit_edge98.61:                                 ; preds = %685, %._crit_edge97.61
  br i1 %36, label %687, label %._crit_edge97.62

._crit_edge97.62:                                 ; preds = %._crit_edge98.61
  br label %._crit_edge98.62

687:                                              ; preds = %._crit_edge98.61
  %.sroa.0101.248.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 62
  %688 = bitcast float %.sroa.0101.248.vec.extract to i32
  store i32 %688, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.62

._crit_edge98.62:                                 ; preds = %687, %._crit_edge97.62
  br i1 %36, label %689, label %._crit_edge97.63

._crit_edge97.63:                                 ; preds = %._crit_edge98.62
  br label %._crit_edge98.63

689:                                              ; preds = %._crit_edge98.62
  %.sroa.0101.252.vec.extract = extractelement <64 x float> %.sroa.0.252.vec.insert, i32 63
  %690 = bitcast float %.sroa.0101.252.vec.extract to i32
  store i32 %690, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.63

._crit_edge98.63:                                 ; preds = %689, %._crit_edge97.63
  br i1 %36, label %691, label %._crit_edge97.64

._crit_edge97.64:                                 ; preds = %._crit_edge98.63
  br label %._crit_edge98.64

691:                                              ; preds = %._crit_edge98.63
  %.sroa.65102.256.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 0
  %692 = bitcast float %.sroa.65102.256.vec.extract to i32
  store i32 %692, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.64

._crit_edge98.64:                                 ; preds = %691, %._crit_edge97.64
  br i1 %36, label %693, label %._crit_edge97.65

._crit_edge97.65:                                 ; preds = %._crit_edge98.64
  br label %._crit_edge98.65

693:                                              ; preds = %._crit_edge98.64
  %.sroa.65102.260.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 1
  %694 = bitcast float %.sroa.65102.260.vec.extract to i32
  store i32 %694, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.65

._crit_edge98.65:                                 ; preds = %693, %._crit_edge97.65
  br i1 %36, label %695, label %._crit_edge97.66

._crit_edge97.66:                                 ; preds = %._crit_edge98.65
  br label %._crit_edge98.66

695:                                              ; preds = %._crit_edge98.65
  %.sroa.65102.264.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 2
  %696 = bitcast float %.sroa.65102.264.vec.extract to i32
  store i32 %696, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.66

._crit_edge98.66:                                 ; preds = %695, %._crit_edge97.66
  br i1 %36, label %697, label %._crit_edge97.67

._crit_edge97.67:                                 ; preds = %._crit_edge98.66
  br label %._crit_edge98.67

697:                                              ; preds = %._crit_edge98.66
  %.sroa.65102.268.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 3
  %698 = bitcast float %.sroa.65102.268.vec.extract to i32
  store i32 %698, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.67

._crit_edge98.67:                                 ; preds = %697, %._crit_edge97.67
  br i1 %36, label %699, label %._crit_edge97.68

._crit_edge97.68:                                 ; preds = %._crit_edge98.67
  br label %._crit_edge98.68

699:                                              ; preds = %._crit_edge98.67
  %.sroa.65102.272.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 4
  %700 = bitcast float %.sroa.65102.272.vec.extract to i32
  store i32 %700, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.68

._crit_edge98.68:                                 ; preds = %699, %._crit_edge97.68
  br i1 %36, label %701, label %._crit_edge97.69

._crit_edge97.69:                                 ; preds = %._crit_edge98.68
  br label %._crit_edge98.69

701:                                              ; preds = %._crit_edge98.68
  %.sroa.65102.276.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 5
  %702 = bitcast float %.sroa.65102.276.vec.extract to i32
  store i32 %702, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.69

._crit_edge98.69:                                 ; preds = %701, %._crit_edge97.69
  br i1 %36, label %703, label %._crit_edge97.70

._crit_edge97.70:                                 ; preds = %._crit_edge98.69
  br label %._crit_edge98.70

703:                                              ; preds = %._crit_edge98.69
  %.sroa.65102.280.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 6
  %704 = bitcast float %.sroa.65102.280.vec.extract to i32
  store i32 %704, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.70

._crit_edge98.70:                                 ; preds = %703, %._crit_edge97.70
  br i1 %36, label %705, label %._crit_edge97.71

._crit_edge97.71:                                 ; preds = %._crit_edge98.70
  br label %._crit_edge98.71

705:                                              ; preds = %._crit_edge98.70
  %.sroa.65102.284.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 7
  %706 = bitcast float %.sroa.65102.284.vec.extract to i32
  store i32 %706, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.71

._crit_edge98.71:                                 ; preds = %705, %._crit_edge97.71
  br i1 %36, label %707, label %._crit_edge97.72

._crit_edge97.72:                                 ; preds = %._crit_edge98.71
  br label %._crit_edge98.72

707:                                              ; preds = %._crit_edge98.71
  %.sroa.65102.288.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 8
  %708 = bitcast float %.sroa.65102.288.vec.extract to i32
  store i32 %708, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.72

._crit_edge98.72:                                 ; preds = %707, %._crit_edge97.72
  br i1 %36, label %709, label %._crit_edge97.73

._crit_edge97.73:                                 ; preds = %._crit_edge98.72
  br label %._crit_edge98.73

709:                                              ; preds = %._crit_edge98.72
  %.sroa.65102.292.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 9
  %710 = bitcast float %.sroa.65102.292.vec.extract to i32
  store i32 %710, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.73

._crit_edge98.73:                                 ; preds = %709, %._crit_edge97.73
  br i1 %36, label %711, label %._crit_edge97.74

._crit_edge97.74:                                 ; preds = %._crit_edge98.73
  br label %._crit_edge98.74

711:                                              ; preds = %._crit_edge98.73
  %.sroa.65102.296.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 10
  %712 = bitcast float %.sroa.65102.296.vec.extract to i32
  store i32 %712, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.74

._crit_edge98.74:                                 ; preds = %711, %._crit_edge97.74
  br i1 %36, label %713, label %._crit_edge97.75

._crit_edge97.75:                                 ; preds = %._crit_edge98.74
  br label %._crit_edge98.75

713:                                              ; preds = %._crit_edge98.74
  %.sroa.65102.300.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 11
  %714 = bitcast float %.sroa.65102.300.vec.extract to i32
  store i32 %714, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.75

._crit_edge98.75:                                 ; preds = %713, %._crit_edge97.75
  br i1 %36, label %715, label %._crit_edge97.76

._crit_edge97.76:                                 ; preds = %._crit_edge98.75
  br label %._crit_edge98.76

715:                                              ; preds = %._crit_edge98.75
  %.sroa.65102.304.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 12
  %716 = bitcast float %.sroa.65102.304.vec.extract to i32
  store i32 %716, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.76

._crit_edge98.76:                                 ; preds = %715, %._crit_edge97.76
  br i1 %36, label %717, label %._crit_edge97.77

._crit_edge97.77:                                 ; preds = %._crit_edge98.76
  br label %._crit_edge98.77

717:                                              ; preds = %._crit_edge98.76
  %.sroa.65102.308.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 13
  %718 = bitcast float %.sroa.65102.308.vec.extract to i32
  store i32 %718, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.77

._crit_edge98.77:                                 ; preds = %717, %._crit_edge97.77
  br i1 %36, label %719, label %._crit_edge97.78

._crit_edge97.78:                                 ; preds = %._crit_edge98.77
  br label %._crit_edge98.78

719:                                              ; preds = %._crit_edge98.77
  %.sroa.65102.312.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 14
  %720 = bitcast float %.sroa.65102.312.vec.extract to i32
  store i32 %720, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.78

._crit_edge98.78:                                 ; preds = %719, %._crit_edge97.78
  br i1 %36, label %721, label %._crit_edge97.79

._crit_edge97.79:                                 ; preds = %._crit_edge98.78
  br label %._crit_edge98.79

721:                                              ; preds = %._crit_edge98.78
  %.sroa.65102.316.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 15
  %722 = bitcast float %.sroa.65102.316.vec.extract to i32
  store i32 %722, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.79

._crit_edge98.79:                                 ; preds = %721, %._crit_edge97.79
  br i1 %36, label %723, label %._crit_edge97.80

._crit_edge97.80:                                 ; preds = %._crit_edge98.79
  br label %._crit_edge98.80

723:                                              ; preds = %._crit_edge98.79
  %.sroa.65102.320.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 16
  %724 = bitcast float %.sroa.65102.320.vec.extract to i32
  store i32 %724, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.80

._crit_edge98.80:                                 ; preds = %723, %._crit_edge97.80
  br i1 %36, label %725, label %._crit_edge97.81

._crit_edge97.81:                                 ; preds = %._crit_edge98.80
  br label %._crit_edge98.81

725:                                              ; preds = %._crit_edge98.80
  %.sroa.65102.324.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 17
  %726 = bitcast float %.sroa.65102.324.vec.extract to i32
  store i32 %726, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.81

._crit_edge98.81:                                 ; preds = %725, %._crit_edge97.81
  br i1 %36, label %727, label %._crit_edge97.82

._crit_edge97.82:                                 ; preds = %._crit_edge98.81
  br label %._crit_edge98.82

727:                                              ; preds = %._crit_edge98.81
  %.sroa.65102.328.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 18
  %728 = bitcast float %.sroa.65102.328.vec.extract to i32
  store i32 %728, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.82

._crit_edge98.82:                                 ; preds = %727, %._crit_edge97.82
  br i1 %36, label %729, label %._crit_edge97.83

._crit_edge97.83:                                 ; preds = %._crit_edge98.82
  br label %._crit_edge98.83

729:                                              ; preds = %._crit_edge98.82
  %.sroa.65102.332.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 19
  %730 = bitcast float %.sroa.65102.332.vec.extract to i32
  store i32 %730, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.83

._crit_edge98.83:                                 ; preds = %729, %._crit_edge97.83
  br i1 %36, label %731, label %._crit_edge97.84

._crit_edge97.84:                                 ; preds = %._crit_edge98.83
  br label %._crit_edge98.84

731:                                              ; preds = %._crit_edge98.83
  %.sroa.65102.336.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 20
  %732 = bitcast float %.sroa.65102.336.vec.extract to i32
  store i32 %732, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.84

._crit_edge98.84:                                 ; preds = %731, %._crit_edge97.84
  br i1 %36, label %733, label %._crit_edge97.85

._crit_edge97.85:                                 ; preds = %._crit_edge98.84
  br label %._crit_edge98.85

733:                                              ; preds = %._crit_edge98.84
  %.sroa.65102.340.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 21
  %734 = bitcast float %.sroa.65102.340.vec.extract to i32
  store i32 %734, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.85

._crit_edge98.85:                                 ; preds = %733, %._crit_edge97.85
  br i1 %36, label %735, label %._crit_edge97.86

._crit_edge97.86:                                 ; preds = %._crit_edge98.85
  br label %._crit_edge98.86

735:                                              ; preds = %._crit_edge98.85
  %.sroa.65102.344.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 22
  %736 = bitcast float %.sroa.65102.344.vec.extract to i32
  store i32 %736, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.86

._crit_edge98.86:                                 ; preds = %735, %._crit_edge97.86
  br i1 %36, label %737, label %._crit_edge97.87

._crit_edge97.87:                                 ; preds = %._crit_edge98.86
  br label %._crit_edge98.87

737:                                              ; preds = %._crit_edge98.86
  %.sroa.65102.348.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 23
  %738 = bitcast float %.sroa.65102.348.vec.extract to i32
  store i32 %738, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.87

._crit_edge98.87:                                 ; preds = %737, %._crit_edge97.87
  br i1 %36, label %739, label %._crit_edge97.88

._crit_edge97.88:                                 ; preds = %._crit_edge98.87
  br label %._crit_edge98.88

739:                                              ; preds = %._crit_edge98.87
  %.sroa.65102.352.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 24
  %740 = bitcast float %.sroa.65102.352.vec.extract to i32
  store i32 %740, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.88

._crit_edge98.88:                                 ; preds = %739, %._crit_edge97.88
  br i1 %36, label %741, label %._crit_edge97.89

._crit_edge97.89:                                 ; preds = %._crit_edge98.88
  br label %._crit_edge98.89

741:                                              ; preds = %._crit_edge98.88
  %.sroa.65102.356.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 25
  %742 = bitcast float %.sroa.65102.356.vec.extract to i32
  store i32 %742, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.89

._crit_edge98.89:                                 ; preds = %741, %._crit_edge97.89
  br i1 %36, label %743, label %._crit_edge97.90

._crit_edge97.90:                                 ; preds = %._crit_edge98.89
  br label %._crit_edge98.90

743:                                              ; preds = %._crit_edge98.89
  %.sroa.65102.360.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 26
  %744 = bitcast float %.sroa.65102.360.vec.extract to i32
  store i32 %744, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.90

._crit_edge98.90:                                 ; preds = %743, %._crit_edge97.90
  br i1 %36, label %745, label %._crit_edge97.91

._crit_edge97.91:                                 ; preds = %._crit_edge98.90
  br label %._crit_edge98.91

745:                                              ; preds = %._crit_edge98.90
  %.sroa.65102.364.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 27
  %746 = bitcast float %.sroa.65102.364.vec.extract to i32
  store i32 %746, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.91

._crit_edge98.91:                                 ; preds = %745, %._crit_edge97.91
  br i1 %36, label %747, label %._crit_edge97.92

._crit_edge97.92:                                 ; preds = %._crit_edge98.91
  br label %._crit_edge98.92

747:                                              ; preds = %._crit_edge98.91
  %.sroa.65102.368.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 28
  %748 = bitcast float %.sroa.65102.368.vec.extract to i32
  store i32 %748, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.92

._crit_edge98.92:                                 ; preds = %747, %._crit_edge97.92
  br i1 %36, label %749, label %._crit_edge97.93

._crit_edge97.93:                                 ; preds = %._crit_edge98.92
  br label %._crit_edge98.93

749:                                              ; preds = %._crit_edge98.92
  %.sroa.65102.372.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 29
  %750 = bitcast float %.sroa.65102.372.vec.extract to i32
  store i32 %750, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.93

._crit_edge98.93:                                 ; preds = %749, %._crit_edge97.93
  br i1 %36, label %751, label %._crit_edge97.94

._crit_edge97.94:                                 ; preds = %._crit_edge98.93
  br label %._crit_edge98.94

751:                                              ; preds = %._crit_edge98.93
  %.sroa.65102.376.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 30
  %752 = bitcast float %.sroa.65102.376.vec.extract to i32
  store i32 %752, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.94

._crit_edge98.94:                                 ; preds = %751, %._crit_edge97.94
  br i1 %36, label %753, label %._crit_edge97.95

._crit_edge97.95:                                 ; preds = %._crit_edge98.94
  br label %._crit_edge98.95

753:                                              ; preds = %._crit_edge98.94
  %.sroa.65102.380.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 31
  %754 = bitcast float %.sroa.65102.380.vec.extract to i32
  store i32 %754, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.95

._crit_edge98.95:                                 ; preds = %753, %._crit_edge97.95
  br i1 %36, label %755, label %._crit_edge97.96

._crit_edge97.96:                                 ; preds = %._crit_edge98.95
  br label %._crit_edge98.96

755:                                              ; preds = %._crit_edge98.95
  %.sroa.65102.384.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 32
  %756 = bitcast float %.sroa.65102.384.vec.extract to i32
  store i32 %756, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.96

._crit_edge98.96:                                 ; preds = %755, %._crit_edge97.96
  br i1 %36, label %757, label %._crit_edge97.97

._crit_edge97.97:                                 ; preds = %._crit_edge98.96
  br label %._crit_edge98.97

757:                                              ; preds = %._crit_edge98.96
  %.sroa.65102.388.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 33
  %758 = bitcast float %.sroa.65102.388.vec.extract to i32
  store i32 %758, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.97

._crit_edge98.97:                                 ; preds = %757, %._crit_edge97.97
  br i1 %36, label %759, label %._crit_edge97.98

._crit_edge97.98:                                 ; preds = %._crit_edge98.97
  br label %._crit_edge98.98

759:                                              ; preds = %._crit_edge98.97
  %.sroa.65102.392.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 34
  %760 = bitcast float %.sroa.65102.392.vec.extract to i32
  store i32 %760, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.98

._crit_edge98.98:                                 ; preds = %759, %._crit_edge97.98
  br i1 %36, label %761, label %._crit_edge97.99

._crit_edge97.99:                                 ; preds = %._crit_edge98.98
  br label %._crit_edge98.99

761:                                              ; preds = %._crit_edge98.98
  %.sroa.65102.396.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 35
  %762 = bitcast float %.sroa.65102.396.vec.extract to i32
  store i32 %762, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.99

._crit_edge98.99:                                 ; preds = %761, %._crit_edge97.99
  br i1 %36, label %763, label %._crit_edge97.100

._crit_edge97.100:                                ; preds = %._crit_edge98.99
  br label %._crit_edge98.100

763:                                              ; preds = %._crit_edge98.99
  %.sroa.65102.400.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 36
  %764 = bitcast float %.sroa.65102.400.vec.extract to i32
  store i32 %764, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.100

._crit_edge98.100:                                ; preds = %763, %._crit_edge97.100
  br i1 %36, label %765, label %._crit_edge97.101

._crit_edge97.101:                                ; preds = %._crit_edge98.100
  br label %._crit_edge98.101

765:                                              ; preds = %._crit_edge98.100
  %.sroa.65102.404.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 37
  %766 = bitcast float %.sroa.65102.404.vec.extract to i32
  store i32 %766, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.101

._crit_edge98.101:                                ; preds = %765, %._crit_edge97.101
  br i1 %36, label %767, label %._crit_edge97.102

._crit_edge97.102:                                ; preds = %._crit_edge98.101
  br label %._crit_edge98.102

767:                                              ; preds = %._crit_edge98.101
  %.sroa.65102.408.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 38
  %768 = bitcast float %.sroa.65102.408.vec.extract to i32
  store i32 %768, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.102

._crit_edge98.102:                                ; preds = %767, %._crit_edge97.102
  br i1 %36, label %769, label %._crit_edge97.103

._crit_edge97.103:                                ; preds = %._crit_edge98.102
  br label %._crit_edge98.103

769:                                              ; preds = %._crit_edge98.102
  %.sroa.65102.412.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 39
  %770 = bitcast float %.sroa.65102.412.vec.extract to i32
  store i32 %770, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.103

._crit_edge98.103:                                ; preds = %769, %._crit_edge97.103
  br i1 %36, label %771, label %._crit_edge97.104

._crit_edge97.104:                                ; preds = %._crit_edge98.103
  br label %._crit_edge98.104

771:                                              ; preds = %._crit_edge98.103
  %.sroa.65102.416.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 40
  %772 = bitcast float %.sroa.65102.416.vec.extract to i32
  store i32 %772, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.104

._crit_edge98.104:                                ; preds = %771, %._crit_edge97.104
  br i1 %36, label %773, label %._crit_edge97.105

._crit_edge97.105:                                ; preds = %._crit_edge98.104
  br label %._crit_edge98.105

773:                                              ; preds = %._crit_edge98.104
  %.sroa.65102.420.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 41
  %774 = bitcast float %.sroa.65102.420.vec.extract to i32
  store i32 %774, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.105

._crit_edge98.105:                                ; preds = %773, %._crit_edge97.105
  br i1 %36, label %775, label %._crit_edge97.106

._crit_edge97.106:                                ; preds = %._crit_edge98.105
  br label %._crit_edge98.106

775:                                              ; preds = %._crit_edge98.105
  %.sroa.65102.424.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 42
  %776 = bitcast float %.sroa.65102.424.vec.extract to i32
  store i32 %776, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.106

._crit_edge98.106:                                ; preds = %775, %._crit_edge97.106
  br i1 %36, label %777, label %._crit_edge97.107

._crit_edge97.107:                                ; preds = %._crit_edge98.106
  br label %._crit_edge98.107

777:                                              ; preds = %._crit_edge98.106
  %.sroa.65102.428.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 43
  %778 = bitcast float %.sroa.65102.428.vec.extract to i32
  store i32 %778, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.107

._crit_edge98.107:                                ; preds = %777, %._crit_edge97.107
  br i1 %36, label %779, label %._crit_edge97.108

._crit_edge97.108:                                ; preds = %._crit_edge98.107
  br label %._crit_edge98.108

779:                                              ; preds = %._crit_edge98.107
  %.sroa.65102.432.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 44
  %780 = bitcast float %.sroa.65102.432.vec.extract to i32
  store i32 %780, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.108

._crit_edge98.108:                                ; preds = %779, %._crit_edge97.108
  br i1 %36, label %781, label %._crit_edge97.109

._crit_edge97.109:                                ; preds = %._crit_edge98.108
  br label %._crit_edge98.109

781:                                              ; preds = %._crit_edge98.108
  %.sroa.65102.436.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 45
  %782 = bitcast float %.sroa.65102.436.vec.extract to i32
  store i32 %782, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.109

._crit_edge98.109:                                ; preds = %781, %._crit_edge97.109
  br i1 %36, label %783, label %._crit_edge97.110

._crit_edge97.110:                                ; preds = %._crit_edge98.109
  br label %._crit_edge98.110

783:                                              ; preds = %._crit_edge98.109
  %.sroa.65102.440.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 46
  %784 = bitcast float %.sroa.65102.440.vec.extract to i32
  store i32 %784, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.110

._crit_edge98.110:                                ; preds = %783, %._crit_edge97.110
  br i1 %36, label %785, label %._crit_edge97.111

._crit_edge97.111:                                ; preds = %._crit_edge98.110
  br label %._crit_edge98.111

785:                                              ; preds = %._crit_edge98.110
  %.sroa.65102.444.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 47
  %786 = bitcast float %.sroa.65102.444.vec.extract to i32
  store i32 %786, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.111

._crit_edge98.111:                                ; preds = %785, %._crit_edge97.111
  br i1 %36, label %787, label %._crit_edge97.112

._crit_edge97.112:                                ; preds = %._crit_edge98.111
  br label %._crit_edge98.112

787:                                              ; preds = %._crit_edge98.111
  %.sroa.65102.448.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 48
  %788 = bitcast float %.sroa.65102.448.vec.extract to i32
  store i32 %788, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.112

._crit_edge98.112:                                ; preds = %787, %._crit_edge97.112
  br i1 %36, label %789, label %._crit_edge97.113

._crit_edge97.113:                                ; preds = %._crit_edge98.112
  br label %._crit_edge98.113

789:                                              ; preds = %._crit_edge98.112
  %.sroa.65102.452.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 49
  %790 = bitcast float %.sroa.65102.452.vec.extract to i32
  store i32 %790, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.113

._crit_edge98.113:                                ; preds = %789, %._crit_edge97.113
  br i1 %36, label %791, label %._crit_edge97.114

._crit_edge97.114:                                ; preds = %._crit_edge98.113
  br label %._crit_edge98.114

791:                                              ; preds = %._crit_edge98.113
  %.sroa.65102.456.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 50
  %792 = bitcast float %.sroa.65102.456.vec.extract to i32
  store i32 %792, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.114

._crit_edge98.114:                                ; preds = %791, %._crit_edge97.114
  br i1 %36, label %793, label %._crit_edge97.115

._crit_edge97.115:                                ; preds = %._crit_edge98.114
  br label %._crit_edge98.115

793:                                              ; preds = %._crit_edge98.114
  %.sroa.65102.460.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 51
  %794 = bitcast float %.sroa.65102.460.vec.extract to i32
  store i32 %794, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.115

._crit_edge98.115:                                ; preds = %793, %._crit_edge97.115
  br i1 %36, label %795, label %._crit_edge97.116

._crit_edge97.116:                                ; preds = %._crit_edge98.115
  br label %._crit_edge98.116

795:                                              ; preds = %._crit_edge98.115
  %.sroa.65102.464.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 52
  %796 = bitcast float %.sroa.65102.464.vec.extract to i32
  store i32 %796, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.116

._crit_edge98.116:                                ; preds = %795, %._crit_edge97.116
  br i1 %36, label %797, label %._crit_edge97.117

._crit_edge97.117:                                ; preds = %._crit_edge98.116
  br label %._crit_edge98.117

797:                                              ; preds = %._crit_edge98.116
  %.sroa.65102.468.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 53
  %798 = bitcast float %.sroa.65102.468.vec.extract to i32
  store i32 %798, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.117

._crit_edge98.117:                                ; preds = %797, %._crit_edge97.117
  br i1 %36, label %799, label %._crit_edge97.118

._crit_edge97.118:                                ; preds = %._crit_edge98.117
  br label %._crit_edge98.118

799:                                              ; preds = %._crit_edge98.117
  %.sroa.65102.472.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 54
  %800 = bitcast float %.sroa.65102.472.vec.extract to i32
  store i32 %800, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.118

._crit_edge98.118:                                ; preds = %799, %._crit_edge97.118
  br i1 %36, label %801, label %._crit_edge97.119

._crit_edge97.119:                                ; preds = %._crit_edge98.118
  br label %._crit_edge98.119

801:                                              ; preds = %._crit_edge98.118
  %.sroa.65102.476.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 55
  %802 = bitcast float %.sroa.65102.476.vec.extract to i32
  store i32 %802, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.119

._crit_edge98.119:                                ; preds = %801, %._crit_edge97.119
  br i1 %36, label %803, label %._crit_edge97.120

._crit_edge97.120:                                ; preds = %._crit_edge98.119
  br label %._crit_edge98.120

803:                                              ; preds = %._crit_edge98.119
  %.sroa.65102.480.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 56
  %804 = bitcast float %.sroa.65102.480.vec.extract to i32
  store i32 %804, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.120

._crit_edge98.120:                                ; preds = %803, %._crit_edge97.120
  br i1 %36, label %805, label %._crit_edge97.121

._crit_edge97.121:                                ; preds = %._crit_edge98.120
  br label %._crit_edge98.121

805:                                              ; preds = %._crit_edge98.120
  %.sroa.65102.484.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 57
  %806 = bitcast float %.sroa.65102.484.vec.extract to i32
  store i32 %806, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.121

._crit_edge98.121:                                ; preds = %805, %._crit_edge97.121
  br i1 %36, label %807, label %._crit_edge97.122

._crit_edge97.122:                                ; preds = %._crit_edge98.121
  br label %._crit_edge98.122

807:                                              ; preds = %._crit_edge98.121
  %.sroa.65102.488.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 58
  %808 = bitcast float %.sroa.65102.488.vec.extract to i32
  store i32 %808, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.122

._crit_edge98.122:                                ; preds = %807, %._crit_edge97.122
  br i1 %36, label %809, label %._crit_edge97.123

._crit_edge97.123:                                ; preds = %._crit_edge98.122
  br label %._crit_edge98.123

809:                                              ; preds = %._crit_edge98.122
  %.sroa.65102.492.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 59
  %810 = bitcast float %.sroa.65102.492.vec.extract to i32
  store i32 %810, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.123

._crit_edge98.123:                                ; preds = %809, %._crit_edge97.123
  br i1 %36, label %811, label %._crit_edge97.124

._crit_edge97.124:                                ; preds = %._crit_edge98.123
  br label %._crit_edge98.124

811:                                              ; preds = %._crit_edge98.123
  %.sroa.65102.496.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 60
  %812 = bitcast float %.sroa.65102.496.vec.extract to i32
  store i32 %812, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.124

._crit_edge98.124:                                ; preds = %811, %._crit_edge97.124
  br i1 %36, label %813, label %._crit_edge97.125

._crit_edge97.125:                                ; preds = %._crit_edge98.124
  br label %._crit_edge98.125

813:                                              ; preds = %._crit_edge98.124
  %.sroa.65102.500.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 61
  %814 = bitcast float %.sroa.65102.500.vec.extract to i32
  store i32 %814, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.125

._crit_edge98.125:                                ; preds = %813, %._crit_edge97.125
  br i1 %36, label %815, label %._crit_edge97.126

._crit_edge97.126:                                ; preds = %._crit_edge98.125
  br label %._crit_edge98.126

815:                                              ; preds = %._crit_edge98.125
  %.sroa.65102.504.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 62
  %816 = bitcast float %.sroa.65102.504.vec.extract to i32
  store i32 %816, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.126

._crit_edge98.126:                                ; preds = %815, %._crit_edge97.126
  br i1 %36, label %817, label %._crit_edge97.127

._crit_edge97.127:                                ; preds = %._crit_edge98.126
  br label %._crit_edge98.127

817:                                              ; preds = %._crit_edge98.126
  %.sroa.65102.508.vec.extract = extractelement <64 x float> %.sroa.65.508.vec.insert, i32 63
  %818 = bitcast float %.sroa.65102.508.vec.extract to i32
  store i32 %818, i32 addrspace(1)* %561, align 4, !tbaa !521
  br label %._crit_edge98.127

._crit_edge98.127:                                ; preds = %817, %._crit_edge97.127
  ret void
}

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_get_simd_size() local_unnamed_addr #2

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_get_simd_id() local_unnamed_addr #2

; Function Attrs: convergent
declare spir_func i32 @__builtin_IB_simd_block_read_1_global(i32 addrspace(1)* noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_size(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_enqueued_local_size(i32 noundef) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_global_offset(i32 noundef) local_unnamed_addr #3

; Function Attrs: nounwind readnone
declare i16 @llvm.genx.GenISA.simdLaneId() #4

; Function Attrs: nounwind readonly
declare i32 @llvm.genx.GenISA.simdBlockRead.i32.p1i32(i32 addrspace(1)*) #5

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #2 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind readnone }
attributes #5 = { nounwind readonly }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.functions = !{!3}
!IGCMetadata = !{!42}
!opencl.ocl.version = !{!516, !516, !516, !516, !516}
!opencl.spir.version = !{!516, !516, !516, !516, !516}
!llvm.ident = !{!517, !517, !517, !517, !517}
!llvm.module.flags = !{!518}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm32ELm32ELm16EE, !4}
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
!42 = !{!"ModuleMD", !43, !44, !141, !311, !342, !343, !347, !350, !351, !352, !387, !413, !426, !427, !428, !443, !444, !445, !446, !447, !448, !449, !450, !451, !452, !456, !457, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !477, !481, !482, !483, !484, !485, !486, !487, !488, !489, !490, !491, !492, !493, !494, !495, !496, !497, !232, !498, !499, !500, !502, !505, !506, !507, !509, !510, !511}
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
!142 = !{!"FuncMDMap[0]", void (float addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, i64, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::ext::oneapi::bfloat16" addrspace(1)*, %"class.sycl::_V1::range"*, %"class.sycl::_V1::range"*, <8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16, i8*, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32, i32)* @_ZTS7imatrixIfLm32ELm32ELm16EE}
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
!178 = !{!"uavsNumType", i32 4}
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
!473 = !{!"MinNOSPushConstantSize", i32 0}
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
!498 = !{!"PrivateMemoryPerFG"}
!499 = !{!"m_OptsToDisable"}
!500 = !{!"capabilities", !501}
!501 = !{!"globalVariableDecorationsINTEL", i1 false}
!502 = !{!"m_ShaderResourceViewMcsMask", !503, !504}
!503 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!504 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!505 = !{!"computedDepthMode", i32 0}
!506 = !{!"isHDCFastClearShader", i1 false}
!507 = !{!"argRegisterReservations", !508}
!508 = !{!"argRegisterReservationsVec[0]", i32 0}
!509 = !{!"SIMD16_SpillThreshold", i8 0}
!510 = !{!"SIMD32_SpillThreshold", i8 0}
!511 = !{!"m_CacheControlOption", !512, !513, !514, !515}
!512 = !{!"LscLoadCacheControlOverride", i8 0}
!513 = !{!"LscStoreCacheControlOverride", i8 0}
!514 = !{!"TgmLoadCacheControlOverride", i8 0}
!515 = !{!"TgmStoreCacheControlOverride", i8 0}
!516 = !{i32 2, i32 0}
!517 = !{!"clang version 14.0.5"}
!518 = !{i32 1, !"wchar_size", i32 4}
!519 = !{!520}
!520 = !{i32 4469}
!521 = !{!522, !522, i64 0}
!522 = !{!"int", !523, i64 0}
!523 = !{!"omnipotent char", !524, i64 0}
!524 = !{!"Simple C/C++ TBAA"}
