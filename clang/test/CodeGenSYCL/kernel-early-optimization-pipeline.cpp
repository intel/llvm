// Check LLVM optimization pipeline is run by default for SPIR-V compiled for
// SYCL device target, and can be disabled with -fno-sycl-early-optimizations.
// New pass manager doesn't print all passes tree, only module level.
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s
// CHECK: SYCLVirtualFunctionsAnalysisPass
// CHECK: ESIMDVerifierPass
// CHECK: SYCLConditionalCallOnDevicePass
// CHECK: SYCLPropagateAspectsUsagePass
// CHECK: SYCLPropagateJointMatrixUsagePass
// CHECK: SYCLLowerWGLocalMemoryPass
// CHECK: CompileTimePropertiesPass
// CHECK: InferFunctionAttrsPass
// CHECK: AlwaysInlinerPass
// CHECK: ModuleInlinerWrapperPass
// CHECK: SYCLOptimizeBarriersPass
// CHECK: ConstantMergePass
// CHECK: SYCLMutatePrintfAddrspacePass
// CHECK: SYCLPropagateAspectsUsagePass
// CHECK: SYCLAddOptLevelAttributePass
// CHECK: RecordSYCLAspectNamesPass
// CHECK: CleanupSYCLMetadataPass
//
// RUN: %clang_cc1 -O2 -fsycl-is-device -triple spir64-unknown-unknown %s -mdebug-pass Structure -emit-llvm -fno-sycl-early-optimizations -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-NOEARLYOPT
// CHECK-NOEARLYOPT-NOT: ConstantMergePass1
// CHECK-NOEARLYOPT-NOT: SYCLOptimizeBarriersPass
// CHECK-NOEARLYOPT: SYCLMutatePrintfAddrspacePass

// RUN: %clang_cc1 -O0 -fsycl-is-device -triple spir64-unknown-unknown %s -mdebug-pass Structure -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-O0
// CHECK-O0-NOT: SYCLOptimizeBarriersPass

// Passes registered via registerOptimizerLastEPCallback don't run on empty
// code
__attribute__((sycl_device)) void bar() {
}
