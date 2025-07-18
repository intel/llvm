//===- llvm/InitializePasses.h - Initialize All Passes ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the pass initialization routines
// for the entire LLVM project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INITIALIZEPASSES_H
#define LLVM_INITIALIZEPASSES_H

namespace llvm {

class PassRegistry;

/// Initialize all passes linked into the Core library.
void initializeCore(PassRegistry &);

/// Initialize all passes linked into the TransformUtils library.
void initializeTransformUtils(PassRegistry &);

/// Initialize all passes linked into the ScalarOpts library.
void initializeScalarOpts(PassRegistry &);

/// Initialize all passes linked into the Vectorize library.
void initializeVectorization(PassRegistry &);

/// Initialize all passes linked into the InstCombine library.
void initializeInstCombine(PassRegistry &);

/// Initialize all passes linked into the IPO library.
void initializeIPO(PassRegistry &);

/// Initialize all passes linked into the Analysis library.
void initializeAnalysis(PassRegistry &);

/// Initialize all passes linked into the CodeGen library.
void initializeCodeGen(PassRegistry &);

/// Initialize all passes linked into the GlobalISel library.
void initializeGlobalISel(PassRegistry &);

/// Initialize all passes linked into the CodeGen library.
void initializeTarget(PassRegistry &);

void initializeAAResultsWrapperPassPass(PassRegistry &);
void initializeAlwaysInlinerLegacyPassPass(PassRegistry &);
void initializeAssignmentTrackingAnalysisPass(PassRegistry &);
void initializeAssumptionCacheTrackerPass(PassRegistry &);
void initializeAtomicExpandLegacyPass(PassRegistry &);
void initializeBasicBlockPathCloningPass(PassRegistry &);
void initializeBasicBlockSectionsProfileReaderWrapperPassPass(PassRegistry &);
void initializeBasicBlockSectionsPass(PassRegistry &);
void initializeBarrierNoopPass(PassRegistry &);
void initializeBasicAAWrapperPassPass(PassRegistry &);
void initializeBlockFrequencyInfoWrapperPassPass(PassRegistry &);
void initializeBranchFolderLegacyPass(PassRegistry &);
void initializeBranchProbabilityInfoWrapperPassPass(PassRegistry &);
void initializeBranchRelaxationLegacyPass(PassRegistry &);
void initializeBreakCriticalEdgesPass(PassRegistry &);
void initializeBreakFalseDepsPass(PassRegistry &);
void initializeCanonicalizeFreezeInLoopsPass(PassRegistry &);
void initializeCFGSimplifyPassPass(PassRegistry &);
void initializeCFGuardPass(PassRegistry &);
void initializeCFGuardLongjmpPass(PassRegistry &);
void initializeCFIFixupPass(PassRegistry &);
void initializeCFIInstrInserterPass(PassRegistry &);
void initializeCallBrPreparePass(PassRegistry &);
void initializeCallGraphDOTPrinterPass(PassRegistry &);
void initializeCallGraphViewerPass(PassRegistry &);
void initializeCallGraphWrapperPassPass(PassRegistry &);
void initializeCheckDebugMachineModulePass(PassRegistry &);
void initializeCodeGenPrepareLegacyPassPass(PassRegistry &);
void initializeComplexDeinterleavingLegacyPassPass(PassRegistry &);
void initializeConstantHoistingLegacyPassPass(PassRegistry &);
void initializeCycleInfoWrapperPassPass(PassRegistry &);
void initializeDAEPass(PassRegistry &);
void initializeDAHPass(PassRegistry &);
void initializeDAESYCLPass(PassRegistry &);
void initializeDCELegacyPassPass(PassRegistry &);
void initializeDXILMetadataAnalysisWrapperPassPass(PassRegistry &);
void initializeDXILMetadataAnalysisWrapperPrinterPass(PassRegistry &);
void initializeDXILResourceBindingWrapperPassPass(PassRegistry &);
void initializeDXILResourceImplicitBindingLegacyPass(PassRegistry &);
void initializeDXILResourceTypeWrapperPassPass(PassRegistry &);
void initializeDXILResourceWrapperPassPass(PassRegistry &);
void initializeDeadMachineInstructionElimPass(PassRegistry &);
void initializeDebugifyMachineModulePass(PassRegistry &);
void initializeDependenceAnalysisWrapperPassPass(PassRegistry &);
void initializeDetectDeadLanesLegacyPass(PassRegistry &);
void initializeDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializeDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializeDomPrinterWrapperPassPass(PassRegistry &);
void initializeDomViewerWrapperPassPass(PassRegistry &);
void initializeDominanceFrontierWrapperPassPass(PassRegistry &);
void initializeDominatorTreeWrapperPassPass(PassRegistry &);
void initializeDwarfEHPrepareLegacyPassPass(PassRegistry &);
void initializeEarlyCSELegacyPassPass(PassRegistry &);
void initializeEarlyCSEMemSSALegacyPassPass(PassRegistry &);
void initializeEarlyIfConverterLegacyPass(PassRegistry &);
void initializeEarlyIfPredicatorPass(PassRegistry &);
void initializeEarlyMachineLICMPass(PassRegistry &);
void initializeEarlyTailDuplicateLegacyPass(PassRegistry &);
void initializeEdgeBundlesWrapperLegacyPass(PassRegistry &);
void initializeEHContGuardTargetsPass(PassRegistry &);
void initializeExpandFpLegacyPassPass(PassRegistry &);
void initializeExpandLargeDivRemLegacyPassPass(PassRegistry &);
void initializeExpandMemCmpLegacyPassPass(PassRegistry &);
void initializeExpandPostRALegacyPass(PassRegistry &);
void initializeExpandReductionsPass(PassRegistry &);
void initializeExpandVariadicsPass(PassRegistry &);
void initializeExternalAAWrapperPassPass(PassRegistry &);
void initializeFPBuiltinFnSelectionLegacyPassPass(PassRegistry &);
void initializeFEntryInserterLegacyPass(PassRegistry &);
void initializeFinalizeISelPass(PassRegistry &);
void initializeFinalizeMachineBundlesPass(PassRegistry &);
void initializeFixIrreduciblePass(PassRegistry &);
void initializeFixupStatepointCallerSavedLegacyPass(PassRegistry &);
void initializeFlattenCFGLegacyPassPass(PassRegistry &);
void initializeFuncletLayoutPass(PassRegistry &);
void initializeGCEmptyBasicBlocksPass(PassRegistry &);
void initializeGCMachineCodeAnalysisPass(PassRegistry &);
void initializeGCModuleInfoPass(PassRegistry &);
void initializeGVNLegacyPassPass(PassRegistry &);
void initializeGlobalMergeFuncPassWrapperPass(PassRegistry &);
void initializeGlobalMergePass(PassRegistry &);
void initializeGlobalsAAWrapperPassPass(PassRegistry &);
void initializeHardwareLoopsLegacyPass(PassRegistry &);
void initializeMIRProfileLoaderPassPass(PassRegistry &);
void initializeIRSimilarityIdentifierWrapperPassPass(PassRegistry &);
void initializeIRTranslatorPass(PassRegistry &);
void initializeIVUsersWrapperPassPass(PassRegistry &);
void initializeIfConverterPass(PassRegistry &);
void initializeImmutableModuleSummaryIndexWrapperPassPass(PassRegistry &);
void initializeImplicitNullChecksPass(PassRegistry &);
void initializeIndirectBrExpandLegacyPassPass(PassRegistry &);
void initializeInferAddressSpacesPass(PassRegistry &);
void initializeInstSimplifyLegacyPassPass(PassRegistry &);
void initializeInstructionCombiningPassPass(PassRegistry &);
void initializeInstructionSelectPass(PassRegistry &);
void initializeInterleavedAccessPass(PassRegistry &);
void initializeInterleavedLoadCombinePass(PassRegistry &);
void initializeJMCInstrumenterPass(PassRegistry &);
void initializeKCFIPass(PassRegistry &);
void initializeLCSSAVerificationPassPass(PassRegistry &);
void initializeLCSSAWrapperPassPass(PassRegistry &);
void initializeLazyBFIPassPass(PassRegistry &);
void initializeLazyBlockFrequencyInfoPassPass(PassRegistry &);
void initializeLazyBranchProbabilityInfoPassPass(PassRegistry &);
void initializeLazyMachineBlockFrequencyInfoPassPass(PassRegistry &);
void initializeLazyValueInfoWrapperPassPass(PassRegistry &);
void initializeLegacyLICMPassPass(PassRegistry &);
void initializeLegalizerPass(PassRegistry &);
void initializeGISelCSEAnalysisWrapperPassPass(PassRegistry &);
void initializeGISelValueTrackingAnalysisLegacyPass(PassRegistry &);
void initializeLiveDebugValuesLegacyPass(PassRegistry &);
void initializeLiveDebugVariablesWrapperLegacyPass(PassRegistry &);
void initializeLiveIntervalsWrapperPassPass(PassRegistry &);
void initializeLiveRangeShrinkPass(PassRegistry &);
void initializeLiveRegMatrixWrapperLegacyPass(PassRegistry &);
void initializeLiveStacksWrapperLegacyPass(PassRegistry &);
void initializeLiveVariablesWrapperPassPass(PassRegistry &);
void initializeLoadStoreOptPass(PassRegistry &);
void initializeLoadStoreVectorizerLegacyPassPass(PassRegistry &);
void initializeLocalStackSlotPassPass(PassRegistry &);
void initializeLocalizerPass(PassRegistry &);
void initializeLoopDataPrefetchLegacyPassPass(PassRegistry &);
void initializeLoopExtractorLegacyPassPass(PassRegistry &);
void initializeLoopInfoWrapperPassPass(PassRegistry &);
void initializeLoopPassPass(PassRegistry &);
void initializeLoopSimplifyPass(PassRegistry &);
void initializeLoopStrengthReducePass(PassRegistry &);
void initializeLoopTermFoldPass(PassRegistry &);
void initializeLoopUnrollPass(PassRegistry &);
void initializeLowerAtomicLegacyPassPass(PassRegistry &);
void initializeLowerEmuTLSPass(PassRegistry &);
void initializeLowerGlobalDtorsLegacyPassPass(PassRegistry &);
void initializeLowerIntrinsicsPass(PassRegistry &);
void initializeLowerInvokeLegacyPassPass(PassRegistry &);
void initializeLowerSwitchLegacyPassPass(PassRegistry &);
void initializeKCFIPass(PassRegistry &);
void initializeMIRAddFSDiscriminatorsPass(PassRegistry &);
void initializeMIRCanonicalizerPass(PassRegistry &);
void initializeMIRNamerPass(PassRegistry &);
void initializeMIRPrintingPassPass(PassRegistry &);
void initializeMachineBlockFrequencyInfoWrapperPassPass(PassRegistry &);
void initializeMachineBlockPlacementLegacyPass(PassRegistry &);
void initializeMachineBlockPlacementStatsLegacyPass(PassRegistry &);
void initializeMachineBranchProbabilityInfoWrapperPassPass(PassRegistry &);
void initializeMachineCFGPrinterPass(PassRegistry &);
void initializeMachineCSELegacyPass(PassRegistry &);
void initializeMachineCombinerPass(PassRegistry &);
void initializeMachineCopyPropagationLegacyPass(PassRegistry &);
void initializeMachineCycleInfoPrinterLegacyPass(PassRegistry &);
void initializeMachineCycleInfoWrapperPassPass(PassRegistry &);
void initializeMachineDominanceFrontierPass(PassRegistry &);
void initializeMachineDominatorTreeWrapperPassPass(PassRegistry &);
void initializeMachineFunctionPrinterPassPass(PassRegistry &);
void initializeMachineFunctionSplitterPass(PassRegistry &);
void initializeMachineLateInstrsCleanupLegacyPass(PassRegistry &);
void initializeMachineLICMPass(PassRegistry &);
void initializeMachineLoopInfoWrapperPassPass(PassRegistry &);
void initializeMachineModuleInfoWrapperPassPass(PassRegistry &);
void initializeMachineOptimizationRemarkEmitterPassPass(PassRegistry &);
void initializeMachineOutlinerPass(PassRegistry &);
void initializeStaticDataProfileInfoWrapperPassPass(PassRegistry &);
void initializeStaticDataAnnotatorPass(PassRegistry &);
void initializeMachinePipelinerPass(PassRegistry &);
void initializeMachinePostDominatorTreeWrapperPassPass(PassRegistry &);
void initializeMachineRegionInfoPassPass(PassRegistry &);
void initializeMachineSanitizerBinaryMetadataLegacyPass(PassRegistry &);
void initializeMachineSchedulerLegacyPass(PassRegistry &);
void initializeMachineSinkingLegacyPass(PassRegistry &);
void initializeMachineTraceMetricsWrapperPassPass(PassRegistry &);
void initializeMachineUniformityInfoPrinterPassPass(PassRegistry &);
void initializeMachineUniformityAnalysisPassPass(PassRegistry &);
void initializeMachineVerifierLegacyPassPass(PassRegistry &);
void initializeMemoryDependenceWrapperPassPass(PassRegistry &);
void initializeMemorySSAWrapperPassPass(PassRegistry &);
void initializeMergeICmpsLegacyPassPass(PassRegistry &);
void initializeModuleSummaryIndexWrapperPassPass(PassRegistry &);
void initializeModuloScheduleTestPass(PassRegistry &);
void initializeNaryReassociateLegacyPassPass(PassRegistry &);
void initializeObjCARCContractLegacyPassPass(PassRegistry &);
void initializeOptimizationRemarkEmitterWrapperPassPass(PassRegistry &);
void initializeOptimizePHIsLegacyPass(PassRegistry &);
void initializePEILegacyPass(PassRegistry &);
void initializePHIEliminationPass(PassRegistry &);
void initializePartiallyInlineLibCallsLegacyPassPass(PassRegistry &);
void initializePatchableFunctionLegacyPass(PassRegistry &);
void initializePeepholeOptimizerLegacyPass(PassRegistry &);
void initializePhiValuesWrapperPassPass(PassRegistry &);
void initializePhysicalRegisterUsageInfoWrapperLegacyPass(PassRegistry &);
void initializePlaceBackedgeSafepointsLegacyPassPass(PassRegistry &);
void initializePostDomOnlyPrinterWrapperPassPass(PassRegistry &);
void initializePostDomOnlyViewerWrapperPassPass(PassRegistry &);
void initializePostDomPrinterWrapperPassPass(PassRegistry &);
void initializePostDomViewerWrapperPassPass(PassRegistry &);
void initializePostDominatorTreeWrapperPassPass(PassRegistry &);
void initializePostInlineEntryExitInstrumenterPass(PassRegistry &);
void initializePostMachineSchedulerLegacyPass(PassRegistry &);
void initializePostRAHazardRecognizerLegacyPass(PassRegistry &);
void initializePostRAMachineSinkingPass(PassRegistry &);
void initializePostRASchedulerLegacyPass(PassRegistry &);
void initializePreISelIntrinsicLoweringLegacyPassPass(PassRegistry &);
void initializePrintFunctionPassWrapperPass(PassRegistry &);
void initializePrintModulePassWrapperPass(PassRegistry &);
void initializeProcessImplicitDefsPass(PassRegistry &);
void initializeProfileSummaryInfoWrapperPassPass(PassRegistry &);
void initializePromoteLegacyPassPass(PassRegistry &);
void initializeRABasicPass(PassRegistry &);
void initializePseudoProbeInserterPass(PassRegistry &);
void initializeRAGreedyLegacyPass(PassRegistry &);
void initializeReachingDefAnalysisPass(PassRegistry &);
void initializeReassociateLegacyPassPass(PassRegistry &);
void initializeRegAllocEvictionAdvisorAnalysisLegacyPass(PassRegistry &);
void initializeRegAllocFastPass(PassRegistry &);
void initializeRegAllocPriorityAdvisorAnalysisLegacyPass(PassRegistry &);
void initializeRegAllocScoringPass(PassRegistry &);
void initializeRegBankSelectPass(PassRegistry &);
void initializeRegToMemWrapperPassPass(PassRegistry &);
void initializeRegUsageInfoCollectorLegacyPass(PassRegistry &);
void initializeRegUsageInfoPropagationLegacyPass(PassRegistry &);
void initializeRegionInfoPassPass(PassRegistry &);
void initializeRegionOnlyPrinterPass(PassRegistry &);
void initializeRegionOnlyViewerPass(PassRegistry &);
void initializeRegionPrinterPass(PassRegistry &);
void initializeRegionViewerPass(PassRegistry &);
void initializeRegisterCoalescerLegacyPass(PassRegistry &);
void initializeRemoveLoadsIntoFakeUsesLegacyPass(PassRegistry &);
void initializeRemoveRedundantDebugValuesLegacyPass(PassRegistry &);
void initializeRenameIndependentSubregsLegacyPass(PassRegistry &);
void initializeReplaceWithVeclibLegacyPass(PassRegistry &);
void initializeResetMachineFunctionPass(PassRegistry &);
void initializeSCEVAAWrapperPassPass(PassRegistry &);
void initializeSROALegacyPassPass(PassRegistry &);
void initializeSafeStackLegacyPassPass(PassRegistry &);
void initializeSafepointIRVerifierPass(PassRegistry &);
void initializeSelectOptimizePass(PassRegistry &);
void initializeScalarEvolutionWrapperPassPass(PassRegistry &);
void initializeScalarizeMaskedMemIntrinLegacyPassPass(PassRegistry &);
void initializeScalarizerLegacyPassPass(PassRegistry &);
void initializeScavengerTestPass(PassRegistry &);
void initializeScopedNoAliasAAWrapperPassPass(PassRegistry &);
void initializeSeparateConstOffsetFromGEPLegacyPassPass(PassRegistry &);
void initializeShadowStackGCLoweringPass(PassRegistry &);
void initializeShrinkWrapLegacyPass(PassRegistry &);
void initializeSingleLoopExtractorPass(PassRegistry &);
void initializeSinkingLegacyPassPass(PassRegistry &);
void initializeSjLjEHPreparePass(PassRegistry &);
void initializeSlotIndexesWrapperPassPass(PassRegistry &);
void initializeSpeculativeExecutionLegacyPassPass(PassRegistry &);
void initializeSpillPlacementWrapperLegacyPass(PassRegistry &);
void initializeStackColoringLegacyPass(PassRegistry &);
void initializeStackFrameLayoutAnalysisLegacyPass(PassRegistry &);
void initializeStaticDataSplitterPass(PassRegistry &);
void initializeStackMapLivenessPass(PassRegistry &);
void initializeStackProtectorPass(PassRegistry &);
void initializeStackSafetyGlobalInfoWrapperPassPass(PassRegistry &);
void initializeStackSafetyInfoWrapperPassPass(PassRegistry &);
void initializeStackSlotColoringLegacyPass(PassRegistry &);
void initializeStraightLineStrengthReduceLegacyPassPass(PassRegistry &);
void initializeStripDebugMachineModulePass(PassRegistry &);
void initializeStructurizeCFGLegacyPassPass(PassRegistry &);
void initializeSYCLLowerWGScopeLegacyPassPass(PassRegistry &);
void initializeSYCLLowerESIMDLegacyPassPass(PassRegistry &);
void initializeSYCLLowerInvokeSimdLegacyPassPass(PassRegistry &);
void initializeSYCLMutatePrintfAddrspaceLegacyPassPass(PassRegistry &);
void initializeSPIRITTAnnotationsLegacyPassPass(PassRegistry &);
void initializeESIMDLowerLoadStorePass(PassRegistry &);
void initializeESIMDVerifierPass(PassRegistry &);
void initializeSYCLLowerWGLocalMemoryLegacyPass(PassRegistry &);
void initializeTailCallElimPass(PassRegistry &);
void initializeTailDuplicateLegacyPass(PassRegistry &);
void initializeTargetLibraryInfoWrapperPassPass(PassRegistry &);
void initializeTargetPassConfigPass(PassRegistry &);
void initializeTargetTransformInfoWrapperPassPass(PassRegistry &);
void initializeTwoAddressInstructionLegacyPassPass(PassRegistry &);
void initializeTypeBasedAAWrapperPassPass(PassRegistry &);
void initializeTypePromotionLegacyPass(PassRegistry &);
void initializeInitUndefPass(PassRegistry &);
void initializeUniformityInfoWrapperPassPass(PassRegistry &);
void initializeUnifyLoopExitsLegacyPassPass(PassRegistry &);
void initializeUnpackMachineBundlesPass(PassRegistry &);
void initializeUnreachableBlockElimLegacyPassPass(PassRegistry &);
void initializeUnreachableMachineBlockElimLegacyPass(PassRegistry &);
void initializeVerifierLegacyPassPass(PassRegistry &);
void initializeVirtRegMapWrapperLegacyPass(PassRegistry &);
void initializeVirtRegRewriterLegacyPass(PassRegistry &);
void initializeWasmEHPreparePass(PassRegistry &);
void initializeWinEHPreparePass(PassRegistry &);
void initializeWriteBitcodePassPass(PassRegistry &);
void initializeXRayInstrumentationLegacyPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_INITIALIZEPASSES_H
