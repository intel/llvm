; ------------------------------------------------
; OCL_asm1b8e887fff860eac_beforeUnification.ll
; LLVM major version: 14
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @_ZTS7imatrixIfLm16ELm16ELm16EE() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_type_qual !4 !kernel_arg_base_type !4 !kernel_arg_name !4 {
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
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !312
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
  %4 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !312
  %5 = alloca i64, align 8, !spirv.Decorations !312
  %6 = alloca i32, align 4, !spirv.Decorations !314
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
  %1 = alloca [3 x i64], align 8, !spirv.Decorations !312
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
  %3 = alloca i64 addrspace(4)*, align 8, !spirv.Decorations !312
  %4 = alloca i64, align 8, !spirv.Decorations !312
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
!opencl.compiler.options = !{!4}
!igc.functions = !{}
!IGCMetadata = !{!6}

!0 = !{i32 2, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{i32 1, i32 2}
!3 = !{i32 1, i32 0}
!4 = !{}
!5 = !{i16 6, i16 14}
!6 = !{!"ModuleMD", !7, !8, !105, !106, !137, !138, !142, !145, !146, !147, !182, !208, !221, !222, !223, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !251, !252, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !272, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286, !287, !288, !289, !290, !291, !292, !293, !294, !295, !296, !298, !301, !302, !303, !305, !306, !307}
!7 = !{!"isPrecise", i1 false}
!8 = !{!"compOpt", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104}
!9 = !{!"DenormsAreZero", i1 false}
!10 = !{!"BFTFDenormsAreZero", i1 false}
!11 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!12 = !{!"OptDisable", i1 false}
!13 = !{!"MadEnable", i1 false}
!14 = !{!"NoSignedZeros", i1 false}
!15 = !{!"NoNaNs", i1 false}
!16 = !{!"FloatRoundingMode", i32 0}
!17 = !{!"FloatCvtIntRoundingMode", i32 3}
!18 = !{!"LoadCacheDefault", i32 -1}
!19 = !{!"StoreCacheDefault", i32 -1}
!20 = !{!"VISAPreSchedRPThreshold", i32 0}
!21 = !{!"SetLoopUnrollThreshold", i32 0}
!22 = !{!"UnsafeMathOptimizations", i1 false}
!23 = !{!"disableCustomUnsafeOpts", i1 false}
!24 = !{!"disableReducePow", i1 false}
!25 = !{!"disableSqrtOpt", i1 false}
!26 = !{!"FiniteMathOnly", i1 false}
!27 = !{!"FastRelaxedMath", i1 false}
!28 = !{!"DashGSpecified", i1 false}
!29 = !{!"FastCompilation", i1 false}
!30 = !{!"UseScratchSpacePrivateMemory", i1 true}
!31 = !{!"RelaxedBuiltins", i1 false}
!32 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!33 = !{!"GreaterThan2GBBufferRequired", i1 true}
!34 = !{!"GreaterThan4GBBufferRequired", i1 true}
!35 = !{!"DisableA64WA", i1 false}
!36 = !{!"ForceEnableA64WA", i1 false}
!37 = !{!"PushConstantsEnable", i1 true}
!38 = !{!"HasPositivePointerOffset", i1 false}
!39 = !{!"HasBufferOffsetArg", i1 false}
!40 = !{!"BufferOffsetArgOptional", i1 true}
!41 = !{!"replaceGlobalOffsetsByZero", i1 false}
!42 = !{!"forcePixelShaderSIMDMode", i32 0}
!43 = !{!"ForceGeomFFShaderSIMDMode", i32 0}
!44 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!45 = !{!"UniformWGS", i1 false}
!46 = !{!"disableVertexComponentPacking", i1 false}
!47 = !{!"disablePartialVertexComponentPacking", i1 false}
!48 = !{!"PreferBindlessImages", i1 false}
!49 = !{!"UseBindlessMode", i1 false}
!50 = !{!"UseLegacyBindlessMode", i1 true}
!51 = !{!"disableMathRefactoring", i1 false}
!52 = !{!"atomicBranch", i1 false}
!53 = !{!"spillCompression", i1 false}
!54 = !{!"DisableEarlyOut", i1 false}
!55 = !{!"ForceInt32DivRemEmu", i1 false}
!56 = !{!"ForceInt32DivRemEmuSP", i1 false}
!57 = !{!"WaveIntrinsicUsed", i1 false}
!58 = !{!"DisableMultiPolyPS", i1 false}
!59 = !{!"NeedTexture3DLODWA", i1 false}
!60 = !{!"DisableFastestSingleCSSIMD", i1 false}
!61 = !{!"DisableFastestLinearScan", i1 false}
!62 = !{!"UseStatelessforPrivateMemory", i1 false}
!63 = !{!"EnableTakeGlobalAddress", i1 false}
!64 = !{!"IsLibraryCompilation", i1 false}
!65 = !{!"LibraryCompileSIMDSize", i32 0}
!66 = !{!"FastVISACompile", i1 false}
!67 = !{!"MatchSinCosPi", i1 false}
!68 = !{!"ExcludeIRFromZEBinary", i1 false}
!69 = !{!"EmitZeBinVISASections", i1 false}
!70 = !{!"FP64GenEmulationEnabled", i1 false}
!71 = !{!"FP64GenConvEmulationEnabled", i1 false}
!72 = !{!"allowDisableRematforCS", i1 false}
!73 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!74 = !{!"DisableCPSOmaskWA", i1 false}
!75 = !{!"DisableFastestGopt", i1 false}
!76 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!77 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!78 = !{!"DisableConstantCoalescing", i1 false}
!79 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!80 = !{!"WaEnableALTModeVisaWA", i1 false}
!81 = !{!"WaEnableAtomicWaveFusion", i1 false}
!82 = !{!"WaEnableAtomicWaveFusionNonNullResource", i1 false}
!83 = !{!"WaEnableAtomicWaveFusionStateless", i1 false}
!84 = !{!"WaEnableAtomicWaveFusionTyped", i1 false}
!85 = !{!"ForceCBThroughSampler3D", i1 false}
!86 = !{!"WaStoreRawVectorToTypedWrite", i1 false}
!87 = !{!"WaLoadRawVectorToTypedRead", i1 false}
!88 = !{!"WaZeroSLMBeforeUse", i1 false}
!89 = !{!"WaFlagGroupTypedUAVGloballyCoherent", i1 false}
!90 = !{!"NewSpillCostFunction", i1 false}
!91 = !{!"EnableVRT", i1 false}
!92 = !{!"ForceLargeGRFNum4RQ", i1 false}
!93 = !{!"Enable2xGRFRetry", i1 false}
!94 = !{!"Detect2xGRFCandidate", i1 false}
!95 = !{!"EnableURBWritesMerging", i1 true}
!96 = !{!"DisableEUFusion", i1 false}
!97 = !{!"DisableFDivToFMulInvOpt", i1 false}
!98 = !{!"initializePhiSampleSourceWA", i1 false}
!99 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!100 = !{!"DisableLoosenSimd32Occu", i1 false}
!101 = !{!"FastestS1Options", i32 0}
!102 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!103 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!104 = !{!"DisableLscSamplerRouting", i1 false}
!105 = !{!"FuncMD"}
!106 = !{!"pushInfo", !107, !108, !109, !113, !114, !115, !116, !117, !118, !119, !120, !133, !134, !135, !136}
!107 = !{!"pushableAddresses"}
!108 = !{!"bindlessPushInfo"}
!109 = !{!"dynamicBufferInfo", !110, !111, !112}
!110 = !{!"firstIndex", i32 0}
!111 = !{!"numOffsets", i32 0}
!112 = !{!"forceDisabled", i1 false}
!113 = !{!"MaxNumberOfPushedBuffers", i32 0}
!114 = !{!"inlineConstantBufferSlot", i32 -1}
!115 = !{!"inlineConstantBufferOffset", i32 -1}
!116 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!117 = !{!"constants"}
!118 = !{!"inputs"}
!119 = !{!"constantReg"}
!120 = !{!"simplePushInfoArr", !121, !130, !131, !132}
!121 = !{!"simplePushInfoArrVec[0]", !122, !123, !124, !125, !126, !127, !128, !129}
!122 = !{!"cbIdx", i32 0}
!123 = !{!"pushableAddressGrfOffset", i32 -1}
!124 = !{!"pushableOffsetGrfOffset", i32 -1}
!125 = !{!"offset", i32 0}
!126 = !{!"size", i32 0}
!127 = !{!"isStateless", i1 false}
!128 = !{!"isBindless", i1 false}
!129 = !{!"simplePushLoads"}
!130 = !{!"simplePushInfoArrVec[1]", !122, !123, !124, !125, !126, !127, !128, !129}
!131 = !{!"simplePushInfoArrVec[2]", !122, !123, !124, !125, !126, !127, !128, !129}
!132 = !{!"simplePushInfoArrVec[3]", !122, !123, !124, !125, !126, !127, !128, !129}
!133 = !{!"simplePushBufferUsed", i32 0}
!134 = !{!"pushAnalysisWIInfos"}
!135 = !{!"inlineRTGlobalPtrOffset", i32 0}
!136 = !{!"rtSyncSurfPtrOffset", i32 0}
!137 = !{!"WaEnableICBPromotion", i1 false}
!138 = !{!"vsInfo", !139, !140, !141}
!139 = !{!"DrawIndirectBufferIndex", i32 -1}
!140 = !{!"vertexReordering", i32 -1}
!141 = !{!"MaxNumOfOutputs", i32 0}
!142 = !{!"hsInfo", !143, !144}
!143 = !{!"numPatchAttributesPatchBaseName", !""}
!144 = !{!"numVertexAttributesPatchBaseName", !""}
!145 = !{!"dsInfo", !141}
!146 = !{!"gsInfo", !141}
!147 = !{!"psInfo", !148, !149, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181}
!148 = !{!"BlendStateDisabledMask", i8 0}
!149 = !{!"SkipSrc0Alpha", i1 false}
!150 = !{!"DualSourceBlendingDisabled", i1 false}
!151 = !{!"ForceEnableSimd32", i1 false}
!152 = !{!"outputDepth", i1 false}
!153 = !{!"outputStencil", i1 false}
!154 = !{!"outputMask", i1 false}
!155 = !{!"blendToFillEnabled", i1 false}
!156 = !{!"forceEarlyZ", i1 false}
!157 = !{!"hasVersionedLoop", i1 false}
!158 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!159 = !{!"requestCPSizeRelevant", i1 false}
!160 = !{!"requestCPSize", i1 false}
!161 = !{!"texelMaskFastClearMode", !"Disabled"}
!162 = !{!"NumSamples", i8 0}
!163 = !{!"blendOptimizationMode"}
!164 = !{!"colorOutputMask"}
!165 = !{!"ProvokingVertexModeNosIndex", i32 0}
!166 = !{!"ProvokingVertexModeNosPatch", !""}
!167 = !{!"ProvokingVertexModeLast", !"Negative"}
!168 = !{!"VertexAttributesBypass", i1 false}
!169 = !{!"LegacyBaryAssignmentDisableLinear", i1 false}
!170 = !{!"LegacyBaryAssignmentDisableLinearNoPerspective", i1 false}
!171 = !{!"LegacyBaryAssignmentDisableLinearCentroid", i1 false}
!172 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveCentroid", i1 false}
!173 = !{!"LegacyBaryAssignmentDisableLinearSample", i1 false}
!174 = !{!"LegacyBaryAssignmentDisableLinearNoPerspectiveSample", i1 false}
!175 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", !"Negative"}
!176 = !{!"meshShaderWAPerPrimitiveUserDataEnablePatchName", !""}
!177 = !{!"generatePatchesForRTWriteSends", i1 false}
!178 = !{!"forceVMask", i1 false}
!179 = !{!"WaDisableVRS", i1 false}
!180 = !{!"RelaxMemoryVisibilityFromPSOrdering", i1 false}
!181 = !{!"WaEnableVMaskUnderNonUnifromCF", i1 false}
!182 = !{!"csInfo", !183, !184, !185, !186, !187, !20, !21, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !53, !201, !202, !203, !204, !205, !206, !207}
!183 = !{!"maxWorkGroupSize", i32 0}
!184 = !{!"waveSize", i32 0}
!185 = !{!"ComputeShaderSecondCompile"}
!186 = !{!"forcedSIMDSize", i8 0}
!187 = !{!"forceTotalGRFNum", i32 0}
!188 = !{!"forceSpillCompression", i1 false}
!189 = !{!"allowLowerSimd", i1 false}
!190 = !{!"disableSimd32Slicing", i1 false}
!191 = !{!"disableSplitOnSpill", i1 false}
!192 = !{!"enableNewSpillCostFunction", i1 false}
!193 = !{!"forceVISAPreSched", i1 false}
!194 = !{!"forceUniformBuffer", i1 false}
!195 = !{!"forceUniformSurfaceSampler", i1 false}
!196 = !{!"disableLocalIdOrderOptimizations", i1 false}
!197 = !{!"disableDispatchAlongY", i1 false}
!198 = !{!"neededThreadIdLayout", i1* null}
!199 = !{!"forceTileYWalk", i1 false}
!200 = !{!"atomicBranch", i32 0}
!201 = !{!"disableEarlyOut", i1 false}
!202 = !{!"walkOrderEnabled", i1 false}
!203 = !{!"walkOrderOverride", i32 0}
!204 = !{!"ResForHfPacking"}
!205 = !{!"hasWaveMatrix", i1 false}
!206 = !{!"constantFoldSimdSize", i1 false}
!207 = !{!"isNodeShader", i1 false}
!208 = !{!"msInfo", !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !167, !165, !220}
!209 = !{!"PrimitiveTopology", i32 3}
!210 = !{!"MaxNumOfPrimitives", i32 0}
!211 = !{!"MaxNumOfVertices", i32 0}
!212 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!213 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!214 = !{!"WorkGroupSize", i32 0}
!215 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!216 = !{!"IndexFormat", i32 6}
!217 = !{!"SubgroupSize", i32 0}
!218 = !{!"VPandRTAIndexAutostripEnable", i1 false}
!219 = !{!"MeshShaderWAPerPrimitiveUserDataEnable", i1 false}
!220 = !{!"numPrimitiveAttributesPatchBaseName", !""}
!221 = !{!"taskInfo", !141, !214, !215, !217}
!222 = !{!"NBarrierCnt", i32 0}
!223 = !{!"rtInfo", !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237}
!224 = !{!"RayQueryAllocSizeInBytes", i32 0}
!225 = !{!"NumContinuations", i32 0}
!226 = !{!"RTAsyncStackAddrspace", i32 -1}
!227 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!228 = !{!"SWHotZoneAddrspace", i32 -1}
!229 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!230 = !{!"SWStackAddrspace", i32 -1}
!231 = !{!"SWStackSurfaceStateOffset", i1* null}
!232 = !{!"RTSyncStackAddrspace", i32 -1}
!233 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!234 = !{!"doSyncDispatchRays", i1 false}
!235 = !{!"MemStyle", !"Xe"}
!236 = !{!"GlobalDataStyle", !"Xe"}
!237 = !{!"NeedsBTD", i1 true}
!238 = !{!"EnableTextureIndirection", i1 false}
!239 = !{!"EnableSamplerIndirection", i1 false}
!240 = !{!"samplerStateStride", i32 0}
!241 = !{!"samplerStateOffset", i32 0}
!242 = !{!"textureStateStride", i32 0}
!243 = !{!"textureStateOffset", i32 0}
!244 = !{!"CurUniqueIndirectIdx", i32 0}
!245 = !{!"inlineDynTextures"}
!246 = !{!"inlineResInfoData"}
!247 = !{!"immConstant", !248, !249, !250}
!248 = !{!"data"}
!249 = !{!"sizes"}
!250 = !{!"zeroIdxs"}
!251 = !{!"stringConstants"}
!252 = !{!"inlineBuffers", !253, !257, !258}
!253 = !{!"inlineBuffersVec[0]", !254, !255, !256}
!254 = !{!"alignment", i32 0}
!255 = !{!"allocSize", i64 0}
!256 = !{!"Buffer"}
!257 = !{!"inlineBuffersVec[1]", !254, !255, !256}
!258 = !{!"inlineBuffersVec[2]", !254, !255, !256}
!259 = !{!"GlobalPointerProgramBinaryInfos"}
!260 = !{!"ConstantPointerProgramBinaryInfos"}
!261 = !{!"GlobalBufferAddressRelocInfo"}
!262 = !{!"ConstantBufferAddressRelocInfo"}
!263 = !{!"forceLscCacheList"}
!264 = !{!"SrvMap"}
!265 = !{!"RootConstantBufferOffsetInBytes"}
!266 = !{!"RasterizerOrderedByteAddressBuffer"}
!267 = !{!"RasterizerOrderedViews"}
!268 = !{!"MinNOSPushConstantSize", i32 0}
!269 = !{!"inlineProgramScopeOffsets"}
!270 = !{!"shaderData", !271}
!271 = !{!"numReplicas", i32 0}
!272 = !{!"URBInfo", !273, !274, !275}
!273 = !{!"has64BVertexHeaderInput", i1 false}
!274 = !{!"has64BVertexHeaderOutput", i1 false}
!275 = !{!"hasVertexHeader", i1 true}
!276 = !{!"m_ForcePullModel", i1 false}
!277 = !{!"UseBindlessImage", i1 false}
!278 = !{!"enableRangeReduce", i1 false}
!279 = !{!"disableNewTrigFuncRangeReduction", i1 false}
!280 = !{!"enableFRemToSRemOpt", i1 false}
!281 = !{!"enableSampleptrToLdmsptrSample0", i1 false}
!282 = !{!"enableSampleLptrToLdmsptrSample0", i1 false}
!283 = !{!"WaForceSIMD32MicropolyRasterize", i1 false}
!284 = !{!"allowMatchMadOptimizationforVS", i1 false}
!285 = !{!"disableMatchMadOptimizationForCS", i1 false}
!286 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!287 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!288 = !{!"statefulResourcesNotAliased", i1 false}
!289 = !{!"disableMixMode", i1 false}
!290 = !{!"genericAccessesResolved", i1 false}
!291 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!292 = !{!"disableSeparateScratchWA", i1 false}
!293 = !{!"privateMemoryPerWI", i32 0}
!294 = !{!"PrivateMemoryPerFG"}
!295 = !{!"m_OptsToDisable"}
!296 = !{!"capabilities", !297}
!297 = !{!"globalVariableDecorationsINTEL", i1 false}
!298 = !{!"m_ShaderResourceViewMcsMask", !299, !300}
!299 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!300 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!301 = !{!"computedDepthMode", i32 0}
!302 = !{!"isHDCFastClearShader", i1 false}
!303 = !{!"argRegisterReservations", !304}
!304 = !{!"argRegisterReservationsVec[0]", i32 0}
!305 = !{!"SIMD16_SpillThreshold", i8 0}
!306 = !{!"SIMD32_SpillThreshold", i8 0}
!307 = !{!"m_CacheControlOption", !308, !309, !310, !311}
!308 = !{!"LscLoadCacheControlOverride", i8 0}
!309 = !{!"LscStoreCacheControlOverride", i8 0}
!310 = !{!"TgmLoadCacheControlOverride", i8 0}
!311 = !{!"TgmStoreCacheControlOverride", i8 0}
!312 = !{!313}
!313 = !{i32 44, i32 8}
!314 = !{!315}
!315 = !{i32 44, i32 4}
