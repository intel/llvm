; ------------------------------------------------
; OCL_asm0eec18246d6ad435_beforeUnification.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @_ZTS7imatrixIfLm8ELm8ELm8EE() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_type_qual !4 !kernel_arg_base_type !4 !kernel_arg_name !4 {
  call spir_func void @__itt_offload_wi_start_wrapper() #1
  %1 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #5
  %2 = icmp ult i64 %1, 2147483648
  call void @llvm.assume(i1 %2)
  %3 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #5
  %4 = icmp ult i64 %3, 2147483648
  call void @llvm.assume(i1 %4)
  %5 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #5
  %6 = icmp ult i64 %5, 2147483648
  call void @llvm.assume(i1 %6)
  %7 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #5
  %8 = icmp ult i64 %7, 2147483648
  call void @llvm.assume(i1 %8)
  call spir_func void @__itt_offload_wi_finish_wrapper() #1
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !313
  br i1 true, label %25, label %2

2:                                                ; preds = %0
  %3 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %3)
  %4 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %5 = addrspacecast i64* %4 to i64 addrspace(4)*
  %6 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %7 = insertelement <3 x i64> undef, i64 %6, i32 0
  %8 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %9 = insertelement <3 x i64> %7, i64 %8, i32 1
  %10 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %11 = insertelement <3 x i64> %9, i64 %10, i32 2
  %12 = extractelement <3 x i64> %11, i32 0
  store i64 %12, i64* %4, align 8
  %13 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %14 = extractelement <3 x i64> %11, i32 1
  store i64 %14, i64* %13, align 8
  %15 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %16 = extractelement <3 x i64> %11, i32 2
  store i64 %16, i64* %15, align 8
  %17 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5
  %18 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 0) #5
  %19 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 1) #5
  %20 = mul i64 %18, %19
  %21 = call spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32 2) #5
  %22 = mul i64 %20, %21
  %23 = trunc i64 %22 to i32
  call spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %5, i64 %17, i32 %23) #3
  %24 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %24)
  br label %25

25:                                               ; preds = %2, %0
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %0, i64 %1, i32 %2) #3 {
  %4 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !313
  %5 = alloca i64, align 8, !spirv.Decorations !313
  %6 = alloca i32, align 4, !spirv.Decorations !315
  %7 = addrspacecast i64 addrspace(4)** %4 to i64 addrspace(4)* addrspace(4)*
  %8 = addrspacecast i64* %5 to i64 addrspace(4)*
  %9 = addrspacecast i32* %6 to i32 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %7, align 8
  store i64 %1, i64 addrspace(4)* %8, align 8
  store i32 %2, i32 addrspace(4)* %9, align 4
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #4

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #1 {
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !313
  br i1 true, label %19, label %2

2:                                                ; preds = %0
  %3 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %3)
  %4 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %5 = addrspacecast i64* %4 to i64 addrspace(4)*
  %6 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #5
  %7 = insertelement <3 x i64> undef, i64 %6, i32 0
  %8 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 1) #5
  %9 = insertelement <3 x i64> %7, i64 %8, i32 1
  %10 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 2) #5
  %11 = insertelement <3 x i64> %9, i64 %10, i32 2
  %12 = extractelement <3 x i64> %11, i32 0
  store i64 %12, i64* %4, align 8
  %13 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %14 = extractelement <3 x i64> %11, i32 1
  store i64 %14, i64* %13, align 8
  %15 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %16 = extractelement <3 x i64> %11, i32 2
  store i64 %16, i64* %15, align 8
  %17 = call spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5
  call spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %5, i64 %17) #3
  %18 = bitcast [3 x i64]* %1 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %18)
  br label %19

19:                                               ; preds = %2, %0
  ret void
}

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %0, i64 %1) #3 {
  %3 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !313
  %4 = alloca i64, align 8, !spirv.Decorations !313
  %5 = addrspacecast i64 addrspace(4)** %3 to i64 addrspace(4)* addrspace(4)*
  %6 = addrspacecast i64* %4 to i64 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %5, align 8
  store i64 %1, i64 addrspace(4)* %6, align 8
  ret void
}

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z29__spirv_BuiltInGlobalLinearIdv() #5

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z28__spirv_BuiltInWorkgroupSizei(i32) #5

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { noinline nounwind optnone }
attributes #4 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #5 = { nounwind readnone willreturn }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!3}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!5}
!opencl.compiler.options = !{!6}
!igc.functions = !{}
!IGCMetadata = !{!7}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{i32 1, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
!6 = !{!"-cl-intel-enable-zebin"}
!7 = !{!"ModuleMD", !8, !9, !106, !107, !138, !139, !143, !146, !147, !148, !183, !209, !222, !223, !224, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !252, !253, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !273, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287, !288, !289, !290, !291, !292, !293, !294, !295, !296, !297, !299, !302, !303, !304, !306, !307, !308}
!8 = !{!"isPrecise", i1 false}
!9 = !{!"compOpt", !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105}
!10 = !{!"DenormsAreZero", i1 false}
!11 = !{!"BFTFDenormsAreZero", i1 false}
!12 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!13 = !{!"OptDisable", i1 false}
!14 = !{!"MadEnable", i1 false}
!15 = !{!"NoSignedZeros", i1 false}
!16 = !{!"NoNaNs", i1 false}
!17 = !{!"FloatRoundingMode", i32 0}
!18 = !{!"FloatCvtIntRoundingMode", i32 3}
!19 = !{!"LoadCacheDefault", i32 -1}
!20 = !{!"StoreCacheDefault", i32 -1}
!21 = !{!"VISAPreSchedRPThreshold", i32 0}
!22 = !{!"SetLoopUnrollThreshold", i32 0}
!23 = !{!"UnsafeMathOptimizations", i1 false}
!24 = !{!"disableCustomUnsafeOpts", i1 false}
!25 = !{!"disableReducePow", i1 false}
!26 = !{!"disableSqrtOpt", i1 false}
!27 = !{!"FiniteMathOnly", i1 false}
!28 = !{!"FastRelaxedMath", i1 false}
!29 = !{!"DashGSpecified", i1 false}
!30 = !{!"FastCompilation", i1 false}
!31 = !{!"UseScratchSpacePrivateMemory", i1 true}
!32 = !{!"RelaxedBuiltins", i1 false}
!33 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!34 = !{!"GreaterThan2GBBufferRequired", i1 true}
!35 = !{!"GreaterThan4GBBufferRequired", i1 true}
!36 = !{!"DisableA64WA", i1 false}
!37 = !{!"ForceEnableA64WA", i1 false}
!38 = !{!"PushConstantsEnable", i1 true}
!39 = !{!"HasPositivePointerOffset", i1 false}
!40 = !{!"HasBufferOffsetArg", i1 false}
!41 = !{!"BufferOffsetArgOptional", i1 true}
!42 = !{!"replaceGlobalOffsetsByZero", i1 false}
!43 = !{!"forcePixelShaderSIMDMode", i32 0}
!44 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!45 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!46 = !{!"UniformWGS", i1 false}
!47 = !{!"disableVertexComponentPacking", i1 false}
!48 = !{!"disablePartialVertexComponentPacking", i1 false}
!49 = !{!"PreferBindlessImages", i1 false}
!50 = !{!"UseBindlessMode", i1 false}
!51 = !{!"UseLegacyBindlessMode", i1 true}
!52 = !{!"disableMathRefactoring", i1 false}
!53 = !{!"atomicBranch", i1 false}
!54 = !{!"spillCompression", i1 false}
!55 = !{!"DisableEarlyOut", i1 false}
!56 = !{!"ForceInt32DivRemEmu", i1 false}
!57 = !{!"ForceInt32DivRemEmuSP", i1 false}
!58 = !{!"WaveIntrinsicUsed", i1 false}
!59 = !{!"DisableMultiPolyPS", i1 false}
!60 = !{!"NeedTexture3DLODWA", i1 false}
!61 = !{!"DisableFastestSingleCSSIMD", i1 false}
!62 = !{!"DisableFastestLinearScan", i1 false}
!63 = !{!"UseStatelessforPrivateMemory", i1 false}
!64 = !{!"EnableTakeGlobalAddress", i1 false}
!65 = !{!"IsLibraryCompilation", i1 false}
!66 = !{!"LibraryCompileSIMDSize", i32 0}
!67 = !{!"FastVISACompile", i1 false}
!68 = !{!"MatchSinCosPi", i1 false}
!69 = !{!"ExcludeIRFromZEBinary", i1 false}
!70 = !{!"EmitZeBinVISASections", i1 false}
!71 = !{!"FP64GenEmulationEnabled", i1 false}
!72 = !{!"FP64GenConvEmulationEnabled", i1 false}
!73 = !{!"allowDisableRematforCS", i1 false}
!74 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!75 = !{!"DisableCPSOmaskWA", i1 false}
!76 = !{!"DisableFastestGopt", i1 false}
!77 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!78 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!79 = !{!"DisableConstantCoalescing", i1 false}
!80 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!81 = !{!"WaEnableALTModeVisaWA", i1 false}
!82 = !{!"WaEnableAtomicWaveFusion", i1 false}
!83 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!84 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!85 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!86 = !{!"ForceCBThroughSampler3D", i1 false}
!87 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!88 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!89 = !{!"WaZeroSLMBeforeUse", i1 false}
!90 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!91 = !{!"NewSpillCostFunction", i1 false}
!92 = !{!"EnableVRT", i1 false}
!93 = !{!"ForceLargeGRFNum4RQ", i1 false}
!94 = !{!"Enable2xGRFRetry", i1 false}
!95 = !{!"Detect2xGRFCandidate", i1 false}
!96 = !{!"EnableURBWritesMerging", i1 true}
!97 = !{!"DisableEUFusion", i1 false}
!98 = !{!"DisableFDivToFMulInvOpt", i1 false}
!99 = !{!"initializePhiSampleSourceWA", i1 false}
!100 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!101 = !{!"DisableLoosenSimd32Occu", i1 false}
!102 = !{!"FastestS1Options", i32 0}
!103 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!104 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!105 = !{!"DisableLscSamplerRouting", i1 false}
!106 = !{!"FuncMD"}
!107 = !{!"pushInfo", !108, !109, !110, !114, !115, !116, !117, !118, !119, !120, !121, !134, !135, !136, !137}
!108 = !{!"pushableAddresses"}
!109 = !{!"bindlessPushInfo"}
!110 = !{!"dynamicBufferInfo", !111, !112, !113}
!111 = !{!"firstIndex", i32 0}
!112 = !{!"numOffsets", i32 0}
!113 = !{!"forceDisabled", i1 false}
!114 = !{!"MaxNumberOfPushedBuffers", i32 0}
!115 = !{!"inlineConstantBufferSlot", i32 -1}
!116 = !{!"inlineConstantBufferOffset", i32 -1}
!117 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!118 = !{!"constants"}
!119 = !{!"inputs"}
!120 = !{!"constantReg"}
!121 = !{!"simplePushInfoArr", !122, !131, !132, !133}
!122 = !{!"simplePushInfoArrVec[0]", !123, !124, !125, !126, !127, !128, !129, !130}
!123 = !{!"cbIdx", i32 0}
!124 = !{!"pushableAddressGrfOffset", i32 -1}
!125 = !{!"pushableOffsetGrfOffset", i32 -1}
!126 = !{!"offset", i32 0}
!127 = !{!"size", i32 0}
!128 = !{!"isStateless", i1 false}
!129 = !{!"isBindless", i1 false}
!130 = !{!"simplePushLoads"}
!131 = !{!"simplePushInfoArrVec[1]", !123, !124, !125, !126, !127, !128, !129, !130}
!132 = !{!"simplePushInfoArrVec[2]", !123, !124, !125, !126, !127, !128, !129, !130}
!133 = !{!"simplePushInfoArrVec[3]", !123, !124, !125, !126, !127, !128, !129, !130}
!134 = !{!"simplePushBufferUsed", i32 0}
!135 = !{!"pushAnalysisWIInfos"}
!136 = !{!"inlineRTGlobalPtrOffset", i32 0}
!137 = !{!"rtSyncSurfPtrOffset", i32 0}
!138 = !{!"WaEnableICBPromotion", i1 false}
!139 = !{!"vsInfo", !140, !141, !142}
!140 = !{!"DrawIndirectBufferIndex", i32 -1}
!141 = !{!"vertexReordering", i32 -1}
!142 = !{!"MaxNumOfOutputs", i32 0}
!143 = !{!"hsInfo", !144, !145}
!144 = !{!"numPatchAttributesPatchBaseName", !""}
!145 = !{!"numVertexAttributesPatchBaseName", !""}
!146 = !{!"dsInfo", !142}
!147 = !{!"gsInfo", !142}
!148 = !{!"psInfo", !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182}
!149 = !{!"BlendStateDisabledMask", i8 0}
!150 = !{!"SkipSrc0Alpha", i1 false}
!151 = !{!"DualSourceBlendingDisabled", i1 false}
!152 = !{!"ForceEnableSimd32", i1 false}
!153 = !{!"outputDepth", i1 false}
!154 = !{!"outputStencil", i1 false}
!155 = !{!"outputMask", i1 false}
!156 = !{!"blendToFillEnabled", i1 false}
!157 = !{!"forceEarlyZ", i1 false}
!158 = !{!"hasVersionedLoop", i1 false}
!159 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!160 = !{!"requestCPSizeRelevant", i1 false}
!161 = !{!"requestCPSize", i1 false}
!162 = !{!"texelMaskFastClearMode", !"Disabled"}
!163 = !{!"NumSamples", i8 0}
!164 = !{!"blendOptimizationMode"}
!165 = !{!"colorOutputMask"}
!166 = !{!"ProvokingVertexModeNosIndex", i32 0}
!167 = !{!"ProvokingVertexModeNosPatch", !""}
!168 = !{!"ProvokingVertexModeLast", !"Negative"}
!169 = !{!"VertexAttributesBypass", i1 false}
!170 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!171 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!172 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!173 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!174 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!175 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!176 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!177 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!178 = !{!"generatePatchesForRTWriteSends", i1 false}
!179 = !{!"forceVMask", i1 false}
!180 = !{!"WaDisableVRS", i1 false}
!181 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!182 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!183 = !{!"csInfo", !184, !185, !186, !187, !188, !21, !22, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !54, !202, !203, !204, !205, !206, !207, !208}
!184 = !{!"maxWorkGroupSize", i32 0}
!185 = !{!"waveSize", i32 0}
!186 = !{!"ComputeShaderSecondCompile"}
!187 = !{!"forcedSIMDSize", i8 0}
!188 = !{!"forceTotalGRFNum", i32 0}
!189 = !{!"forceSpillCompression", i1 false}
!190 = !{!"allowLowerSimd", i1 false}
!191 = !{!"disableSimd32Slicing", i1 false}
!192 = !{!"disableSplitOnSpill", i1 false}
!193 = !{!"enableNewSpillCostFunction", i1 false}
!194 = !{!"forceVISAPreSched", i1 false}
!195 = !{!"forceUniformBuffer", i1 false}
!196 = !{!"forceUniformSurfaceSampler", i1 false}
!197 = !{!"disableLocalIdOrderOptimizations", i1 false}
!198 = !{!"disableDispatchAlongY", i1 false}
!199 = !{!"neededThreadIdLayout", i1* null}
!200 = !{!"forceTileYWalk", i1 false}
!201 = !{!"atomicBranch", i32 0}
!202 = !{!"disableEarlyOut", i1 false}
!203 = !{!"walkOrderEnabled", i1 false}
!204 = !{!"walkOrderOverride", i32 0}
!205 = !{!"ResForHfPacking"}
!206 = !{!"hasWaveMatrix", i1 false}
!207 = !{!"constantFoldSimdSize", i1 false}
!208 = !{!"isNodeShader", i1 false}
!209 = !{!"msInfo", !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !168, !166, !221}
!210 = !{!"PrimitiveTopology", i32 3}
!211 = !{!"MaxNumOfPrimitives", i32 0}
!212 = !{!"MaxNumOfVertices", i32 0}
!213 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!214 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!215 = !{!"WorkGroupSize", i32 0}
!216 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!217 = !{!"IndexFormat", i32 6}
!218 = !{!"SubgroupSize", i32 0}
!219 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!220 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!221 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!222 = !{!"taskInfo", !142, !215, !216, !218}
!223 = !{!"NBarrierCnt", i32 0}
!224 = !{!"rtInfo", !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238}
!225 = !{!"RayQueryAllocSizeInBytes", i32 0}
!226 = !{!"NumContinuations", i32 0}
!227 = !{!"RTAsyncStackAddrspace", i32 -1}
!228 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!229 = !{!"SWHotZoneAddrspace", i32 -1}
!230 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!231 = !{!"SWStackAddrspace", i32 -1}
!232 = !{!"SWStackSurfaceStateOffset", i1* null}
!233 = !{!"RTSyncStackAddrspace", i32 -1}
!234 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!235 = !{!"doSyncDispatchRays", i1 false}
!236 = !{!"MemStyle", !"Xe"}
!237 = !{!"GlobalDataStyle", !"Xe"}
!238 = !{!"NeedsBTD", i1 true}
!239 = !{!"EnableTextureIndirection", i1 false}
!240 = !{!"EnableSamplerIndirection", i1 false}
!241 = !{!"samplerStateStride", i32 0}
!242 = !{!"samplerStateOffset", i32 0}
!243 = !{!"textureStateStride", i32 0}
!244 = !{!"textureStateOffset", i32 0}
!245 = !{!"CurUniqueIndirectIdx", i32 0}
!246 = !{!"inlineDynTextures"}
!247 = !{!"inlineResInfoData"}
!248 = !{!"immConstant", !249, !250, !251}
!249 = !{!"data"}
!250 = !{!"sizes"}
!251 = !{!"zeroIdxs"}
!252 = !{!"stringConstants"}
!253 = !{!"inlineBuffers", !254, !258, !259}
!254 = !{!"inlineBuffersVec[0]", !255, !256, !257}
!255 = !{!"alignment", i32 0}
!256 = !{!"allocSize", i64 0}
!257 = !{!"Buffer"}
!258 = !{!"inlineBuffersVec[1]", !255, !256, !257}
!259 = !{!"inlineBuffersVec[2]", !255, !256, !257}
!260 = !{!"GlobalPointerProgramBinaryInfos"}
!261 = !{!"ConstantPointerProgramBinaryInfos"}
!262 = !{!"GlobalBufferAddressRelocInfo"}
!263 = !{!"ConstantBufferAddressRelocInfo"}
!264 = !{!"forceLscCacheList"}
!265 = !{!"SrvMap"}
!266 = !{!"RootConstantBufferOffsetInBytes"}
!267 = !{!"RasterizerOrderedByteAddressBuffer"}
!268 = !{!"RasterizerOrderedViews"}
!269 = !{!"MinNOSPushConstantSize", i32 0}
!270 = !{!"inlineProgramScopeOffsets"}
!271 = !{!"shaderData", !272}
!272 = !{!"numReplicas", i32 0}
!273 = !{!"URBInfo", !274, !275, !276}
!274 = !{!"has64BVertexHeaderInput", i1 false}
!275 = !{!"has64BVertexHeaderOutput", i1 false}
!276 = !{!"hasVertexHeader", i1 true}
!277 = !{!"m_ForcePullModel", i1 false}
!278 = !{!"UseBindlessImage", i1 false}
!279 = !{!"enableRangeReduce", i1 false}
!280 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!281 = !{!"enableFRemToSRemOpt", i1 false}
!282 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!283 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!284 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!285 = !{!"allowMatchMadOptimizationforVS", i1 false}
!286 = !{!"disableMatchMadOptimizationForCS", i1 false}
!287 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!288 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!289 = !{!"statefulResourcesNotAliased", i1 false}
!290 = !{!"disableMixMode", i1 false}
!291 = !{!"genericAccessesResolved", i1 false}
!292 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!293 = !{!"disableSeparateScratchWA", i1 false}
!294 = !{!"privateMemoryPerWI", i32 0}
!295 = !{!"PrivateMemoryPerFG"}
!296 = !{!"m_OptsToDisable"}
!297 = !{!"capabilities", !298}
!298 = !{!"globalVariableDecorationsINTEL", i1 false}
!299 = !{!"m_ShaderResourceViewMcsMask", !300, !301}
!300 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!301 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!302 = !{!"computedDepthMode", i32 0}
!303 = !{!"isHDCFastClearShader", i1 false}
!304 = !{!"argRegisterReservations", !305}
!305 = !{!"argRegisterReservationsVec[0]", i32 0}
!306 = !{!"SIMD16_SpillThreshold", i8 0}
!307 = !{!"SIMD32_SpillThreshold", i8 0}
!308 = !{!"m_CacheControlOption", !309, !310, !311, !312}
!309 = !{!"LscLoadCacheControlOverride", i8 0}
!310 = !{!"LscStoreCacheControlOverride", i8 0}
!311 = !{!"TgmLoadCacheControlOverride", i8 0}
!312 = !{!"TgmStoreCacheControlOverride", i8 0}
!313 = !{!314}
!314 = !{i32 44, i32 8}
!315 = !{!316}
!316 = !{i32 44, i32 4}
