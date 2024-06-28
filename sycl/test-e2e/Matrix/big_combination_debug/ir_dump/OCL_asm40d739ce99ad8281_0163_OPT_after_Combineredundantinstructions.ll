; ------------------------------------------------
; OCL_asm40d739ce99ad8281_0163_OPT_after_Combineredundantinstructions.ll
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
  %10 = mul i32 %enqueuedLocalSize.scalar85, %r0.scalar83
  %localIdY15 = zext i16 %localIdY to i32
  %11 = add i32 %10, %localIdY15
  %12 = add i32 %11, %payloadHeader.scalar87
  %13 = icmp sgt i32 %12, -1
  call void @llvm.assume(i1 %13)
  %14 = mul i32 %enqueuedLocalSize.scalar, %r0.scalar78
  %localIdX21 = zext i16 %localIdX to i32
  %15 = add i32 %14, %localIdX21
  %16 = add i32 %15, %payloadHeader.scalar
  %17 = icmp sgt i32 %16, -1
  call void @llvm.assume(i1 %17)
  %simdLaneId16 = call i16 @llvm.genx.GenISA.simdLaneId()
  %18 = icmp ult i16 %simdLaneId16, 1024
  br i1 %18, label %._crit_edge98, label %._crit_edge98.127

._crit_edge98:                                    ; preds = %_Z18get_sub_group_sizev.exit.i4
  br label %._crit_edge98.127

._crit_edge98.127:                                ; preds = %_Z18get_sub_group_sizev.exit.i4, %._crit_edge98
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
