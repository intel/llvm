//===- driver.cc - cgeist Driver ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for cgeist when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include <clang/Basic/DiagnosticIDs.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"

#include "Options.h"
#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"

#include <fstream>

#define DEBUG_TYPE "cgeist"

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

extern int cc1_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(llvm::ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(llvm::ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr);

static llvm::ExitOnError ExitOnErr;

std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    llvm::SmallString<128> ExecutablePath(Argv0);
    // Do a PATH lookup if Argv0 isn't a valid path.
    if (!llvm::sys::fs::exists(ExecutablePath))
      if (llvm::ErrorOr<std::string> P =
              llvm::sys::findProgramByName(ExecutablePath))
        ExecutablePath = *P;
    return std::string(ExecutablePath.str());
  }

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void *)(intptr_t)GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

static int executeCC1Tool(llvm::SmallVectorImpl<const char *> &ArgV) {
  // If we call the cc1 tool from the clangDriver library (through
  // Driver::CC1Main), we need to clean up the options usage count. The options
  // are currently global, and they might have been used previously by the
  // driver.
  llvm::cl::ResetAllOptionOccurrences();

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  llvm::cl::ExpandResponseFiles(Saver, &llvm::cl::TokenizeGNUCommandLine, ArgV);
  llvm::StringRef Tool = ArgV[1];
  void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
  if (Tool == "-cc1")
    return cc1_main(makeArrayRef(ArgV).slice(1), ArgV[0], GetExecutablePathVP);
  if (Tool == "-cc1as")
    return cc1as_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                      GetExecutablePathVP);
  if (Tool == "-cc1gen-reproducer")
    return cc1gen_reproducer_main(makeArrayRef(ArgV).slice(2), ArgV[0],
                                  GetExecutablePathVP);
  // Reject unknown tools.
  llvm::errs() << "error: unknown integrated tool '" << Tool << "'. "
               << "Valid tools include '-cc1' and '-cc1as'.\n";
  return 1;
}

static int emitBinary(const char *Argv0, const char *Filename,
                      const llvm::SmallVectorImpl<const char *> &LinkArgs,
                      bool LinkOMP) {
  using namespace clang;

  ArgumentList Argv;
  Argv.push_back(Argv0);
  Argv.push_back(Filename);
  if (LinkOMP)
    Argv.push_back("-fopenmp");
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.push_back(ResourceDir);
  }
  if (Verbose)
    Argv.push_back("-v");
  if (CUDAGPUArch != "")
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  if (CUDAPath != "")
    Argv.emplace_back("--cuda-path=", CUDAPath);
  if (Output != "") {
    Argv.push_back("-o");
    Argv.push_back(Output);
  }
  for (const auto *Arg : LinkArgs)
    Argv.push_back(Arg);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = CreateAndPopulateDiagOpts(
      Argv.getArguments()); // new DiagnosticOptions();
  TextDiagnosticPrinter *DiagClient =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  const std::string TargetTriple = (TargetTripleOpt == "")
                                       ? llvm::sys::getDefaultTargetTriple()
                                       : TargetTripleOpt;

  driver::Driver TheDriver(Argv0, TargetTriple, Diags, "cgeist LLVM compiler");

  const std::unique_ptr<driver::Compilation> C(
      TheDriver.BuildCompilation(Argv.getArguments()));
  if (!C)
    return 1;

  if (ResourceDir != "")
    TheDriver.ResourceDir = ResourceDir;
  if (SysRoot != "")
    TheDriver.SysRoot = SysRoot;

  LLVM_DEBUG({
    llvm::dbgs() << "Compilation flow:\n";
    TheDriver.PrintActions(*C);
  });

  SmallVector<std::pair<int, const driver::Command *>, 4> FailingCommands;
  int Res = TheDriver.ExecuteCompilation(*C, FailingCommands);

  for (const auto &P : FailingCommands) {
    int CommandRes = P.first;
    const driver::Command *FailingCommand = P.second;
    if (!Res)
      Res = CommandRes;

    // If result status is < 0, then the driver command signalled an error.
    // If result status is 70, then the driver command reported a fatal error.
    // On Windows, abort will return an exit code of 3.  In these cases,
    // generate additional diagnostic information if possible.
    bool IsCrash = CommandRes < 0 || CommandRes == 70;
#ifdef _WIN32
    IsCrash |= CommandRes == 3;
#endif
    if (IsCrash) {
      TheDriver.generateCompilationDiagnostics(*C, *FailingCommand);
      break;
    }
  }
  Diags.getClient()->finish();

  return Res;
}

#include "Lib/clang-mlir.cc"

// Load MLIR Dialects.
static void loadDialects(MLIRContext &Ctx, const bool SYCLIsDevice) {
  Ctx.disableMultithreading();
  Ctx.getOrLoadDialect<mlir::AffineDialect>();
  Ctx.getOrLoadDialect<func::FuncDialect>();
  Ctx.getOrLoadDialect<mlir::DLTIDialect>();
  Ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  Ctx.getOrLoadDialect<mlir::async::AsyncDialect>();
  Ctx.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  Ctx.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  Ctx.getOrLoadDialect<mlir::gpu::GPUDialect>();
  Ctx.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  Ctx.getOrLoadDialect<mlir::math::MathDialect>();
  Ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  Ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  Ctx.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();
  Ctx.getOrLoadDialect<mlir::vector::VectorDialect>();

  if (SYCLIsDevice) {
    Ctx.getOrLoadDialect<mlir::sycl::SYCLDialect>();
  }

  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(Ctx);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(Ctx);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(Ctx);
  IndexType::attachInterface<PtrElementModel<IndexType>>(Ctx);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      Ctx);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(Ctx);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      Ctx);
}

// Register MLIR Dialects.
static void registerDialects(MLIRContext &Ctx, const bool SYCLIsDevice) {
  mlir::DialectRegistry Registry;
  mlir::registerOpenMPDialectTranslation(Registry);
  mlir::registerLLVMDialectTranslation(Registry);
  Ctx.appendDialectRegistry(Registry);
  loadDialects(Ctx, SYCLIsDevice);
}

// Enable various options for the passmanager
static void enableOptionsPM(mlir::PassManager &PM) {
  DefaultTimingManager TM;
  applyDefaultTimingManagerCLOptions(TM);
  TimingScope Timing = TM.getRootScope();

  PM.enableVerifier(EarlyVerifier);
  applyPassManagerCLOptions(PM);
  PM.enableTiming(Timing);
}

// MLIR canonicalization & cleanup.
static int canonicalize(mlir::MLIRContext &Ctx,
                        mlir::OwningOpRef<mlir::ModuleOp> &Module) {
  mlir::PassManager PM(&Ctx);
  enableOptionsPM(PM);

  mlir::OpPassManager &OptPM = PM.nestAny();
  GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.maxIterations = CanonicalizeIterations;

  OptPM.addPass(mlir::createCSEPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createMem2RegPass());
  OptPM.addPass(mlir::createCSEPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createMem2RegPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createRemoveTrivialUsePass());
  OptPM.addPass(polygeist::createMem2RegPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createLoopRestructurePass());
  OptPM.addPass(polygeist::replaceAffineCFGPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  if (ScalarReplacement)
    PM.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  if (ParallelLICM)
    OptPM.addPass(polygeist::createParallelLICMPass());
  else
    OptPM.addPass(mlir::createLoopInvariantCodeMotionPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createCanonicalizeForPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  if (RaiseToAffine) {
    OptPM.addPass(polygeist::createCanonicalizeForPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      OptPM.addPass(polygeist::createParallelLICMPass());
    else
      OptPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    OptPM.addPass(polygeist::createRaiseSCFToAffinePass());
    OptPM.addPass(polygeist::replaceAffineCFGPass());
    if (ScalarReplacement)
      PM.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(PM.run(Module.get()))) {
    llvm::errs() << "*** Canonicalization failed. Module: ***\n";
    Module->dump();
    return 4;
  }
  if (mlir::failed(mlir::verify(Module.get()))) {
    llvm::errs()
        << "*** Verification after canonicalization failed. Module: ***\n";
    Module->dump();
    return 5;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after canonicalize ***\n";
    Module->dump();
  });

  return 0;
}

// Optimize the MLIR.
static int optimize(mlir::MLIRContext &Ctx,
                    const llvm::OptimizationLevel &OptimizationLevel,
                    mlir::OwningOpRef<mlir::ModuleOp> &Module) {
  mlir::PassManager PM(&Ctx);
  enableOptionsPM(PM);

  mlir::OpPassManager &OptPM = PM.nestAny();
  GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.maxIterations = CanonicalizeIterations;

  if (DetectReduction)
    OptPM.addPass(polygeist::detectReductionPass());

  if (OptimizationLevel != llvm::OptimizationLevel::O0) {
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM.addPass(mlir::createCSEPass());
    // Note: affine dialects must be lowered to allow callees containing affine
    // operations to be inlined.
    if (RaiseToAffine)
      OptPM.addPass(mlir::createLowerAffinePass());
    PM.addPass(sycl::createInlinePass(sycl::InlineMode::Simple,
                                      /* RemoveDeadCallees */ true));

    mlir::OpPassManager &OptPM2 = PM.nestAny();
    OptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM2.addPass(mlir::createCSEPass());
    OptPM2.addPass(polygeist::createMem2RegPass());
    OptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM2.addPass(mlir::createCSEPass());
    OptPM2.addPass(polygeist::createCanonicalizeForPass());
    if (RaiseToAffine)
      OptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    OptPM2.addPass(polygeist::replaceAffineCFGPass());
    OptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM2.addPass(mlir::createCSEPass());
    if (ParallelLICM)
      OptPM2.addPass(polygeist::createParallelLICMPass());
    else
      OptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    OptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  }

  if (mlir::failed(PM.run(Module.get()))) {
    llvm::errs() << "*** Optimize failed. Module: ***\n";
    Module->dump();
    return 6;
  }
  if (mlir::failed(mlir::verify(Module.get()))) {
    llvm::errs() << "** Verification after optimization failed. Module: ***\n";
    Module->dump();
    return 7;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after optimize ***\n";
    Module->dump();
  });

  return 0;
}

// CUDA specific optimization (add parallel loops around CUDA).
static int optimizeCUDA(mlir::MLIRContext &Ctx,
                        mlir::OwningOpRef<mlir::ModuleOp> &Module) {
  if (!CudaLower)
    return 0;

  constexpr int UnrollSize = 32;
  GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.maxIterations = CanonicalizeIterations;

  mlir::PassManager PM(&Ctx);
  enableOptionsPM(PM);

  mlir::OpPassManager &OptPM = PM.nestAny();
  OptPM.addPass(mlir::createLowerAffinePass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  PM.addPass(polygeist::createParallelLowerPass());
  PM.addPass(mlir::createSymbolDCEPass());
  mlir::OpPassManager &NOptPM = PM.nestAny();
  NOptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  NOptPM.addPass(polygeist::createMem2RegPass());
  NOptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));

  mlir::OpPassManager &NOptPM2 = PM.nestAny();
  NOptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  NOptPM2.addPass(polygeist::createMem2RegPass());
  NOptPM2.addPass(polygeist::createCanonicalizeForPass());
  NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  NOptPM2.addPass(mlir::createCSEPass());
  if (ParallelLICM)
    NOptPM2.addPass(polygeist::createParallelLICMPass());
  else
    NOptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
  NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  if (RaiseToAffine) {
    NOptPM2.addPass(polygeist::createCanonicalizeForPass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      NOptPM2.addPass(polygeist::createParallelLICMPass());
    else
      NOptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    NOptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    NOptPM2.addPass(polygeist::replaceAffineCFGPass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (LoopUnroll)
      NOptPM2.addPass(mlir::createLoopUnrollPass(UnrollSize, false, true));
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    NOptPM2.addPass(mlir::createCSEPass());
    NOptPM2.addPass(polygeist::createMem2RegPass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      NOptPM2.addPass(polygeist::createParallelLICMPass());
    else
      NOptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    NOptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    NOptPM2.addPass(polygeist::replaceAffineCFGPass());
    NOptPM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      PM.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(PM.run(Module.get()))) {
    llvm::errs() << "*** Optimize CUDA failed. Module: ***\n";
    Module->dump();
    return 8;
  }
  if (mlir::failed(mlir::verify(Module.get()))) {
    llvm::errs()
        << "*** Verification after CUDA optimization failed. Module: ***\n";
    Module->dump();
    return 9;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after optimize CUDA ***\n";
    Module->dump();
  });

  return 0;
}

static void finalizeCUDA(mlir::PassManager &PM) {
  if (!CudaLower)
    return;

  mlir::OpPassManager &OptPM = PM.nestAny();

  constexpr int UnrollSize = 32;
  GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.maxIterations = CanonicalizeIterations;

  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(mlir::createCSEPass());
  OptPM.addPass(polygeist::createMem2RegPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(mlir::createCSEPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(polygeist::createCanonicalizeForPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));

  if (RaiseToAffine) {
    OptPM.addPass(polygeist::createCanonicalizeForPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      OptPM.addPass(polygeist::createParallelLICMPass());
    else
      OptPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    OptPM.addPass(polygeist::createRaiseSCFToAffinePass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM.addPass(polygeist::replaceAffineCFGPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      PM.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (ToCPU == "continuation") {
    OptPM.addPass(polygeist::createBarrierRemovalContinuation());
    // PM.nest<mlir::FuncOp>().addPass(mlir::createCanonicalizerPass());
  } else if (ToCPU.size() != 0) {
    OptPM.addPass(polygeist::createCPUifyPass(ToCPU));
  }
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(mlir::createCSEPass());
  OptPM.addPass(polygeist::createMem2RegPass());
  OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPM.addPass(mlir::createCSEPass());
  if (RaiseToAffine) {
    OptPM.addPass(polygeist::createCanonicalizeForPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      OptPM.addPass(polygeist::createParallelLICMPass());
    else
      OptPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    OptPM.addPass(polygeist::createRaiseSCFToAffinePass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM.addPass(polygeist::replaceAffineCFGPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (LoopUnroll)
      OptPM.addPass(mlir::createLoopUnrollPass(UnrollSize, false, true));
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM.addPass(mlir::createCSEPass());
    OptPM.addPass(polygeist::createMem2RegPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ParallelLICM)
      OptPM.addPass(polygeist::createParallelLICMPass());
    else
      OptPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    OptPM.addPass(polygeist::createRaiseSCFToAffinePass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    OptPM.addPass(polygeist::replaceAffineCFGPass());
    OptPM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      PM.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
}

static int finalize(mlir::MLIRContext &Ctx,
                    mlir::OwningOpRef<mlir::ModuleOp> &Module,
                    llvm::DataLayout &DL, bool &LinkOMP) {
  mlir::PassManager PM(&Ctx);
  enableOptionsPM(PM);

  GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.maxIterations = CanonicalizeIterations;

  finalizeCUDA(PM);

  PM.addPass(mlir::createSymbolDCEPass());

  if (EmitLLVM || !EmitAssembly || EmitOpenMPIR) {
    PM.addPass(mlir::createLowerAffinePass());
    if (InnerSerialize)
      PM.addPass(polygeist::createInnerSerializationPass());

    // PM.nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
    if (mlir::failed(PM.run(Module.get()))) {
      llvm::errs() << "*** Finalize failed (phase 1). Module: ***\n";
      Module->dump();
      return 10;
    }
    if (mlir::failed(mlir::verify(Module.get()))) {
      llvm::errs() << "*** Verification after finalization failed (phase 1). "
                      "Module: ***\n";
      Module->dump();
      return 11;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize (phase 1) ***\n";
      Module->dump();
    });

    mlir::PassManager PM2(&Ctx);
    if (SCFOpenMP) {
      PM2.addPass(createConvertSCFToOpenMPPass());
    }
    PM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (OpenMPOpt) {
      PM2.addPass(polygeist::createOpenMPOptPass());
      PM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    }
    PM2.addPass(polygeist::createMem2RegPass());
    PM2.addPass(mlir::createCSEPass());
    PM2.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
    if (mlir::failed(PM2.run(Module.get()))) {
      llvm::errs() << "*** Finalize failed (phase 2). Module: ***\n";
      Module->dump();
      return 12;
    }
    if (mlir::failed(mlir::verify(Module.get()))) {
      llvm::errs() << "*** Verification after finalization failed (phase 2). "
                      "Module: ***\n";
      Module->dump();
      return 13;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize (phase 2) ***\n";
      Module->dump();
    });

    if (!EmitOpenMPIR) {
      Module->walk([&](mlir::omp::ParallelOp) { LinkOMP = true; });
      mlir::PassManager PM3(&Ctx);
      PM3.addPass(mlir::sycl::createSYCLMethodToSYCLCallPass());
      LowerToLLVMOptions Options(&Ctx);
      Options.dataLayout = DL;
      // invalid for gemm.c init array
      Options.useBarePtrCallConv = true;
      PM3.addPass(polygeist::createConvertPolygeistToLLVMPass(Options));
      // PM3.addPass(mlir::createLowerFuncToLLVMPass(options));
      PM3.addPass(polygeist::createLegalizeForSPIRVPass());

      // Needed because SYCLMethodOps lowering might introduce redundant
      // operations.
      PM3.addPass(mlir::createCSEPass());
      PM3.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
      if (mlir::failed(PM3.run(Module.get()))) {
        llvm::errs() << "*** Finalize failed (phase 3). Module: ***\n";
        Module->dump();
        return 14;
      }
      if (mlir::failed(mlir::verify(Module.get()))) {
        llvm::errs() << "Verification after finalization failed (phase 3). "
                        "Module: ***\n";
        Module->dump();
        return 15;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "*** Module after finalize (phase 3) ***\n";
        Module->dump();
      });
    }
  } else {
    if (mlir::failed(PM.run(Module.get()))) {
      llvm::errs() << "*** Finalize failed. Module: ***\n";
      Module->dump();
      return 16;
    }
    if (mlir::failed(mlir::verify(Module.get()))) {
      llvm::errs() << "*** Verification after finalization failed. "
                      "Module: ***\n";
      Module->dump();
      return 17;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize ***\n";
      Module->dump();
    });
  }

  return 0;
}

// Create and execute the MLIR transformations pipeline.
static int createAndExecutePassPipeline(
    mlir::MLIRContext &Ctx, mlir::OwningOpRef<mlir::ModuleOp> &Module,
    llvm::DataLayout &DL, llvm::Triple &Triple,
    const llvm::OptimizationLevel &OptimizationLevel, bool &LinkOMP) {
  // MLIR canonicalization & cleanup.
  int RC = canonicalize(Ctx, Module);
  if (RC != 0)
    return RC;

  // MLIR optimizations.
  RC = optimize(Ctx, OptimizationLevel, Module);
  if (RC != 0)
    return RC;

  // CUDA specific MLIR optimizations.
  RC = optimizeCUDA(Ctx, Module);
  if (RC != 0)
    return RC;

  RC = finalize(Ctx, Module, DL, LinkOMP);
  if (RC != 0)
    return RC;

  return 0;
}

/// Run an optimization pipeline on the LLVM module.
static void
runOptimizationPipeline(llvm::Module &Module,
                        const llvm::OptimizationLevel &OptimizationLevel) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassInstrumentationCallbacks PIC;
  llvm::PrintPassOptions PrintPassOpts;
  PrintPassOpts.Verbose = true;
  PrintPassOpts.SkipAnalyses = true;
  llvm::StandardInstrumentations SI(Module.getContext(), false,
                                    true /*VerifyEachPass*/, PrintPassOpts);
  SI.registerCallbacks(PIC, &FAM);

  llvm::TargetMachine *TM = nullptr;
  llvm::PipelineTuningOptions PTO;

  llvm::PassBuilder PB(TM, PTO, std::nullopt, &PIC);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM =
      (OptimizationLevel == llvm::OptimizationLevel::O0)
          ? PB.buildO0DefaultPipeline(OptimizationLevel)
          : PB.buildPerModuleDefaultPipeline(OptimizationLevel);

  // Before executing passes, print the final values of the LLVM options.
  llvm::cl::PrintOptionValues();

  // Print a textual representation of the LLVM pipeline.
  LLVM_DEBUG({
    llvm::dbgs() << "*** Run LLVM Optimization pipeline: ***\n";

    std::string Pipeline;
    llvm::raw_string_ostream OS(Pipeline);

    MPM.printPipeline(OS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    llvm::dbgs() << Pipeline << "\n";
  });

  // Now that we have all of the passes ready, run them.
  {
    llvm::PrettyStackTraceString CrashInfo("Optimizer");
    llvm::TimeTraceScope TimeScope("Optimizer");
    MPM.run(Module, MAM);
  }
}

// Lower the MLIR in the given module, compile the generated LLVM IR.
static int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &Module,
                         StringRef ModuleId, mlir::MLIRContext &Ctx,
                         llvm::DataLayout &DL, llvm::Triple &Triple,
                         const llvm::OptimizationLevel &OptimizationLevel,
                         const llvm::SmallVectorImpl<const char *> &LinkArgs,
                         const char *Argv0) {
  bool LinkOMP = FOpenMP;
  int RC = createAndExecutePassPipeline(Ctx, Module, DL, Triple,
                                        OptimizationLevel, LinkOMP);
  if (RC != 0) {
    llvm::errs() << "Failed to execute pass pipeline correctly, RC = " << RC
                 << ".\n";
    return RC;
  }

  bool EmitBC = EmitLLVM && !EmitAssembly;
  bool EmitMLIR = EmitAssembly && !EmitLLVM;
  if (EmitMLIR) {
    if (Output == "-") {
      // Write the MLIR to stdout.
      LLVM_DEBUG(llvm::dbgs() << "*** MLIR Produced ***\n");
      Module->print(llvm::outs());
    } else {
      // Write the MLIR to a file.
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      Module->print(out);
      LLVM_DEBUG(llvm::dbgs()
                 << "*** Dumped MLIR in file '" << Output << "' ***\n");
    }
  } else {
    // Generate LLVM IR.
    llvm::LLVMContext LLVMCtx;
    auto LLVMModule =
        mlir::translateModuleToLLVMIR(Module.get(), LLVMCtx, ModuleId);
    if (!LLVMModule) {
      Module->dump();
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }

    LLVMModule->setDataLayout(DL);
    LLVMModule->setTargetTriple(Triple.getTriple());
    LLVM_DEBUG(llvm::dbgs()
               << "*** Translated MLIR to LLVM IR successfully ***\n");

    if (EmitBC || EmitLLVM) {
      // Not needed when emitting binary for now; will be handled by the driver.
      runOptimizationPipeline(*LLVMModule, OptimizationLevel);
    }

    if (EmitBC) {
      assert(Output != "-" && "Expecting output file");
      // Write the LLVM BC to a file.
      std::error_code EC;
      llvm::raw_fd_ostream Out(Output, EC);
      WriteBitcodeToFile(*LLVMModule, Out);
      LLVM_DEBUG(llvm::dbgs()
                 << "*** Dumped LLVM BC in file '" << Output << "' ***\n");
    } else if (EmitLLVM) {
      if (Output == "-") {
        // Write the LLVM IR to stdout.
        LLVM_DEBUG(llvm::dbgs() << "*** LLVM IR Produced ***\n");
        llvm::outs() << *LLVMModule << "\n";
      } else {
        // Write the LLVM IR to a file.
        std::error_code EC;
        llvm::raw_fd_ostream Out(Output, EC);
        Out << *LLVMModule << "\n";
        LLVM_DEBUG(llvm::dbgs()
                   << "*** Dumped LLVM IR in file '" << Output << "' ***\n");
      }
    } else {
      // Compile the LLVM IR.
      auto TmpFile =
          llvm::sys::fs::TempFile::create("/tmp/intermediate%%%%%%%.ll");
      if (!TmpFile) {
        llvm::errs() << "Failed to create " << TmpFile->TmpName << "\n";
        return -1;
      }
      llvm::raw_fd_ostream Out(TmpFile->FD, /*shouldClose*/ false);
      Out << *LLVMModule << "\n";
      Out.flush();

      int Res = emitBinary(Argv0, TmpFile->TmpName.c_str(), LinkArgs, LinkOMP);
      if (Res != 0)
        llvm::errs() << "Compilation failed\n";

      if (TmpFile->discard()) {
        llvm::errs() << "Failed to erase " << TmpFile->TmpName << "\n";
        return -1;
      }
      return Res;
    }
  }

  return 0;
}

static llvm::OptimizationLevel
getOptimizationLevel(unsigned OptimizationLevel) {
  switch (OptimizationLevel) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  case 3:
    return llvm::OptimizationLevel::O3;
  default:
    // All speed levels above 2 are equivalent to '-O3'
    CGEIST_WARNING(llvm::WithColor::warning()
                   << "optimization level '-O" << OptimizationLevel
                   << "' is not supported; using '-O3' instead\n");
  }
  return llvm::OptimizationLevel::O3;
}

static llvm::Expected<llvm::OptimizationLevel>
getOptimizationLevel(char OptimizationLevel) {
  switch (OptimizationLevel) {
  case 's':
    return llvm::OptimizationLevel::Os;
  case 'z':
    return llvm::OptimizationLevel::Oz;
  case 'g':
    // '-Og' is equivalent to '-O1'
    return llvm::OptimizationLevel::O1;
  default:
    return llvm::createStringError(
        std::errc::invalid_argument,
        "error: invalid integral value '%c' in '-O%c'", OptimizationLevel,
        OptimizationLevel);
  }
}

static llvm::Expected<llvm::OptimizationLevel>
parseOptimizationLevel(llvm::StringRef Arg) {
  if (Arg.empty()) {
    // -O and --optimize options are equivalent to -O1
    return llvm::OptimizationLevel::O1;
  }

  if (Arg.startswith("fast")) {
    // We handle -Ofast like -O3.
    return llvm::OptimizationLevel::O3;
  }

  // Drop '=' from --optimize= args.
  constexpr llvm::StringLiteral EqualsSignSuffix("=");
  Arg.consume_front(EqualsSignSuffix);

  constexpr unsigned Radix(10);
  unsigned SpeedLevel;
  if (!Arg.getAsInteger(Radix, SpeedLevel)) {
    return getOptimizationLevel(SpeedLevel);
  }

  assert(Arg.size() == 1 &&
         "Expecting 'g', 's' or 'z' encoding optimization level.");

  return getOptimizationLevel(Arg.front());
}

// Split the input arguments into 2 sets (LinkageOpts, MLIROpts).
static CgeistOptions
splitCommandLineOptions(int argc, char **argv,
                        SmallVector<const char *> &LinkageArgs,
                        SmallVector<const char *> &MLIRArgs) {
  bool SYCLIsDevice = false;
  bool LinkOnly = false;
  llvm::OptimizationLevel OptimizationLevel =
      CgeistOptions::getDefaultOptimizationLevel();

  for (int I = 0; I < argc; I++) {
    StringRef Ref(argv[I]);
    if (Ref == "-Wl,--start-group")
      LinkOnly = true;
    if (!LinkOnly) {
      if (Ref == "-fPIC" || Ref == "-c" || Ref.startswith("-fsanitize"))
        LinkageArgs.push_back(argv[I]);
      else if (Ref == "-L" || Ref == "-l") {
        LinkageArgs.push_back(argv[I]);
        I++;
        LinkageArgs.push_back(argv[I]);
      } else if (Ref.startswith("-L") || Ref.startswith("-l") ||
                 Ref.startswith("-Wl"))
        LinkageArgs.push_back(argv[I]);
      else if (Ref == "-D" || Ref == "-I") {
        MLIRArgs.push_back(argv[I]);
        I++;
        MLIRArgs.push_back(argv[I]);
      } else if (Ref.startswith("-D")) {
        MLIRArgs.push_back("-D");
        MLIRArgs.push_back(&argv[I][2]);
      } else if (Ref.startswith("-I")) {
        MLIRArgs.push_back("-I");
        MLIRArgs.push_back(&argv[I][2]);
      } else if (Ref == "-fsycl-is-device") {
        SYCLIsDevice = true;
        MLIRArgs.push_back(argv[I]);
      } else if (Ref.consume_front("-O") || Ref.consume_front("--optimize")) {
        // If several flags are passed, we keep the last one.
        OptimizationLevel = ExitOnErr(parseOptimizationLevel(Ref));
        LinkageArgs.push_back(argv[I]);
      } else if (Ref == "-g")
        LinkageArgs.push_back(argv[I]);
      else
        MLIRArgs.push_back(argv[I]);
    } else
      LinkageArgs.push_back(argv[I]);

    if (Ref == "-Wl,--end-group")
      LinkOnly = false;
  }

  return {SYCLIsDevice, OptimizationLevel};
}

// Fill the module with the MLIR in the inputFile.
static void loadMLIR(const std::string &InputFile,
                     mlir::OwningOpRef<ModuleOp> &Module,
                     mlir::MLIRContext &Ctx) {
  assert(InputFile.substr(InputFile.find_last_of(".") + 1) == "mlir" &&
         "Input file has incorrect extension");

  std::string ErrorMsg;
  std::unique_ptr<llvm::MemoryBuffer> Input =
      mlir::openInputFile(InputFile, &ErrorMsg);
  if (!Input) {
    llvm::errs() << ErrorMsg << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr SourceMgr;
  SourceMgrDiagnosticHandler SourceMgrHandler(SourceMgr, &Ctx);
  SourceMgr.AddNewSourceBuffer(std::move(Input), llvm::SMLoc());
  Module = mlir::parseSourceFile<ModuleOp>(SourceMgr, &Ctx);
  if (!Module) {
    llvm::errs() << "Error can't load file " << InputFile << "\n";
    exit(1);
  }
}

// Generate MLIR for the input files.
static void
processInputFiles(const llvm::cl::list<std::string> &InputFiles,
                  const llvm::cl::list<std::string> &InputCommandArgs,
                  mlir::MLIRContext &Ctx, mlir::OwningOpRef<ModuleOp> &Module,
                  llvm::DataLayout &DL, llvm::Triple &Triple, const char *Argv0,
                  bool SYCLIsDevice) {
  assert(!InputFiles.empty() && "inputFiles should not be empty");

  // Ensure all input files can be opened.
  std::vector<std::string> Files;
  for (auto InputFile : InputFiles) {
    std::ifstream Ifs(InputFile);
    if (!Ifs.good()) {
      llvm::errs() << "Not able to open file: " << InputFile << "\n";
      exit(-1);
    }
    Files.push_back(InputFile);
  }

  std::vector<std::string> Commands;
  for (auto Cmd : InputCommandArgs)
    Commands.push_back(Cmd);

  // Count the number of MLIR input files we might have.
  int NumMlirFiles =
      llvm::count_if(InputFiles, [](const std::string &InputFile) {
        std::string Extension =
            InputFile.substr(InputFile.find_last_of(".") + 1);
        return (Extension == "mlir");
      });

  // Early exit if we have a input file with a '.mlir' extension.
  if (NumMlirFiles > 0) {
    if (Files.size() != 1) {
      llvm::errs()
          << "More than one input file has been provided. Only a single "
             "input MLIR file can be processed\n";
      exit(-1);
    }
    loadMLIR(Files[0], Module, Ctx);
    return;
  }

  // Generate MLIR for the C/C++ files.
  std::string Fn = (!SYCLIsDevice) ? Cfunction.getValue() : "";
  parseMLIR(Argv0, Files, Fn, IncludeDirs, Defines, Module, Triple, DL,
            Commands);
}

static bool containsFunctions(mlir::gpu::GPUModuleOp DeviceModule) {
  Region &Rgn = DeviceModule.getRegion();
  return !Rgn.getOps<mlir::gpu::GPUFuncOp>().empty() ||
         !Rgn.getOps<mlir::func::FuncOp>().empty();
}

static void eraseHostCode(mlir::ModuleOp Module) {
  LLVM_DEBUG(llvm::dbgs() << "Erasing host code\n");
  SmallVector<std::reference_wrapper<Operation>> ToRemove;
  std::copy_if(Module.begin(), Module.end(), std::back_inserter(ToRemove),
               [](Operation &Op) { return !isa<mlir::gpu::GPUModuleOp>(Op); });
  for (auto Op : ToRemove)
    Op.get().erase();
}

int main(int argc, char **argv) {
  if (argc >= 1 && std::string(argv[1]) == "-cc1") {
    SmallVector<const char *> Argv;
    for (int I = 0; I < argc; I++)
      Argv.push_back(argv[I]);
    return executeCC1Tool(Argv);
  }

  // Forward '-mllvm' arguments to LLVM.
  {
    SmallVector<const char *> Args;
    for (int I = 0; I < argc; I++)
      Args.push_back(argv[I]);

    llvm::ArrayRef<const char *> Argv = makeArrayRef(Args);
    const llvm::opt::OptTable &OptTbl = clang::driver::getDriverOptTable();
    const unsigned IncludedFlagsBitmask = clang::driver::options::CC1AsOption;
    unsigned MissingArgIndex, MissingArgCount;
    llvm::opt::InputArgList InputArgs = OptTbl.ParseArgs(
        Argv, MissingArgIndex, MissingArgCount, IncludedFlagsBitmask);

    SmallVector<const char *> NewArgv = {Argv[0]};
    for (const llvm::opt::Arg *InputArg :
         InputArgs.filtered(clang::driver::options::OPT_mllvm))
      NewArgv.push_back(InputArg->getValue());
    llvm::cl::ParseCommandLineOptions(NewArgv.size(), &NewArgv[0]);
  }

  // Split up the arguments into MLIR and linkage arguments.
  SmallVector<const char *> LinkageArgs, MLIRArgs;
  const CgeistOptions Options =
      splitCommandLineOptions(argc, argv, LinkageArgs, MLIRArgs);
  assert(!MLIRArgs.empty() && "MLIRArgs should not be empty");

  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();

  // Parse command line options.
  llvm::cl::list<std::string> InputFileNames(
      llvm::cl::Positional, llvm::cl::OneOrMore,
      llvm::cl::desc("<Specify input file>"), llvm::cl::cat(ToolOptions));

  llvm::cl::list<std::string> InputCommandArgs(
      "args", llvm::cl::Positional, llvm::cl::desc("<command arguments>"),
      llvm::cl::ZeroOrMore, llvm::cl::PositionalEatsArgs);

  // Register command line options specific to cgeist.
  int Size = MLIRArgs.size();
  const char **Data = MLIRArgs.data();
  llvm::InitLLVM Y(Size, Data);
  llvm::cl::ParseCommandLineOptions(Size, Data);

  // Register MLIR dialects.
  mlir::MLIRContext Ctx;
  registerDialects(Ctx, Options.syclIsDevice());

  // Generate MLIR for the input files.
  mlir::OpBuilder Builder(&Ctx);
  const Location Loc = Builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> Module(mlir::ModuleOp::create(Loc));
  Builder.setInsertionPointToEnd(Module->getBody());
  auto DeviceModule = Builder.create<mlir::gpu::GPUModuleOp>(
      Loc, MLIRASTConsumer::DeviceModuleName);

  llvm::DataLayout DL("");
  llvm::Triple Triple;
  processInputFiles(InputFileNames, InputCommandArgs, Ctx, Module, DL, Triple,
                    argv[0], Options.syclIsDevice());

  LLVM_DEBUG({
    llvm::dbgs() << "Initial MLIR:\n";
    Module->dump();
  });

  // For now, we will work on the device code if it contains any functions and
  // on the host code otherwise.
  if (containsFunctions(DeviceModule)) {
    eraseHostCode(*Module);
    Module.get()->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                          Builder.getUnitAttr());
  } else
    DeviceModule.erase();

  LLVM_DEBUG({
    llvm::dbgs() << "MLIR before compilation:\n";
    Module->dump();
  });

  // Lower the MLIR to LLVM IR, compile the generated LLVM IR.
  return compileModule(
      Module,
      InputFileNames.size() == 1 ? InputFileNames[0] : "LLVMDialectModule", Ctx,
      DL, Triple, Options.getOptimizationLevel(), LinkageArgs, argv[0]);
}
