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

using namespace llvm;

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

extern int cc1_main(ArrayRef<const char *> Argv, const char *Argv0,
                    void *MainAddr);
extern int cc1as_main(ArrayRef<const char *> Argv, const char *Argv0,
                      void *MainAddr);
extern int cc1gen_reproducer_main(ArrayRef<const char *> Argv,
                                  const char *Argv0, void *MainAddr);

static llvm::ExitOnError ExitOnErr;

std::string GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes) {
    SmallString<128> ExecutablePath(Argv0);
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

static int ExecuteCC1Tool(SmallVectorImpl<const char *> &ArgV) {
  // If we call the cc1 tool from the clangDriver library (through
  // Driver::CC1Main), we need to clean up the options usage count. The options
  // are currently global, and they might have been used previously by the
  // driver.
  llvm::cl::ResetAllOptionOccurrences();

  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  llvm::cl::ExpandResponseFiles(Saver, &llvm::cl::TokenizeGNUCommandLine, ArgV);
  StringRef Tool = ArgV[1];
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
                      const SmallVectorImpl<const char *> &LinkArgs,
                      bool LinkOMP) {

  using namespace clang;
  using namespace clang::driver;
  using namespace std;

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagBuffer =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagBuffer);
  std::string TargetTriple = (TargetTripleOpt == "")
                                 ? llvm::sys::getDefaultTargetTriple()
                                 : TargetTripleOpt;

  const char *Binary = Argv0;
  const unique_ptr<Driver> Driver(
      new class Driver(Binary, TargetTriple, Diags));
  Driver->CC1Main = &ExecuteCC1Tool;
  ArgumentList Argv;
  Argv.push_back(Argv0);
  // Argv.push_back("-x");
  // Argv.push_back("ir");
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

#if 0
  // This forwards '-mllvm' arguments to LLVM if present.
  SmallVector<const char *> NewArgv = {Argv[0]};
  for (const opt::Arg *Arg : Args.filtered(OPT_mllvm))
    NewArgv.push_back(Arg->getValue());
  for (const opt::Arg *Arg : Args.filtered(OPT_offload_opt_eq_minus))
    NewArgv.push_back(Args.MakeArgString(StringRef("-") + Arg->getValue()));
#endif

  const unique_ptr<Compilation> Compilation(
      Driver->BuildCompilation(Argv.getArguments()));

  if (ResourceDir != "")
    Driver->ResourceDir = ResourceDir;
  if (SysRoot != "")
    Driver->SysRoot = SysRoot;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;

  LLVM_DEBUG({
    llvm::dbgs() << "Compilation flow:\n";
    Driver->PrintActions(*Compilation);
  });

  int Res = Driver->ExecuteCompilation(*Compilation, FailingCommands);
  for (const auto &P : FailingCommands) {
    int CommandRes = P.first;
    const Command *FailingCommand = P.second;
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
      Driver->generateCompilationDiagnostics(*Compilation, *FailingCommand);
      break;
    }
  }
  Diags.getClient()->finish();

  return Res;
}

#include "Lib/clang-mlir.cc"

// Load MLIR Dialects.
static void loadDialects(MLIRContext &context, const bool syclIsDevice) {
  context.disableMultithreading();
  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<mlir::DLTIDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::async::AsyncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  context.getOrLoadDialect<mlir::gpu::GPUDialect>();
  context.getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::polygeist::PolygeistDialect>();

  if (syclIsDevice) {
    context.getOrLoadDialect<mlir::sycl::SYCLDialect>();
  }

  LLVM::LLVMPointerType::attachInterface<MemRefInsider>(context);
  LLVM::LLVMStructType::attachInterface<MemRefInsider>(context);
  MemRefType::attachInterface<PtrElementModel<MemRefType>>(context);
  IndexType::attachInterface<PtrElementModel<IndexType>>(context);
  LLVM::LLVMStructType::attachInterface<PtrElementModel<LLVM::LLVMStructType>>(
      context);
  LLVM::LLVMPointerType::attachInterface<
      PtrElementModel<LLVM::LLVMPointerType>>(context);
  LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
      context);
}

// Register MLIR Dialects.
static void registerDialects(MLIRContext &context, const bool syclIsDevice) {
  mlir::DialectRegistry registry;
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  context.appendDialectRegistry(registry);
  loadDialects(context, syclIsDevice);
}

// Enable various options for the passmanager
static void enableOptionsPM(mlir::PassManager &pm) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  pm.enableVerifier(EarlyVerifier);
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);
}

// MLIR canonicalization & cleanup.
static int canonicalize(mlir::MLIRContext &context,
                        mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);
  enableOptionsPM(pm);

  mlir::OpPassManager &optPM = pm.nestAny();
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createMem2RegPass());
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createMem2RegPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createRemoveTrivialUsePass());
  optPM.addPass(polygeist::createMem2RegPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createLoopRestructurePass());
  optPM.addPass(polygeist::replaceAffineCFGPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  if (ScalarReplacement)
    pm.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  if (ParallelLICM)
    optPM.addPass(polygeist::createParallelLICMPass());
  else
    optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createCanonicalizeForPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  if (RaiseToAffine) {
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      optPM.addPass(polygeist::createParallelLICMPass());
    else
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(polygeist::createRaiseSCFToAffinePass());
    optPM.addPass(polygeist::replaceAffineCFGPass());
    if (ScalarReplacement)
      pm.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "*** Canonicalization failed. Module: ***\n";
    module->dump();
    return 4;
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    llvm::errs()
        << "*** Verification after canonicalization failed. Module: ***\n";
    module->dump();
    return 5;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after canonicalize ***\n";
    module->dump();
  });

  return 0;
}

// Optimize the MLIR.
static int optimize(mlir::MLIRContext &context,
                    const llvm::OptimizationLevel &OptimizationLevel,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);
  enableOptionsPM(pm);

  mlir::OpPassManager &optPM = pm.nestAny();
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  if (DetectReduction)
    optPM.addPass(polygeist::detectReductionPass());

  if (OptimizationLevel != OptimizationLevel::O0) {
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(mlir::createCSEPass());
    // Affine must be lowered to enable inlining
    if (RaiseToAffine)
      optPM.addPass(mlir::createLowerAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    pm.addPass(mlir::createInlinerPass());
    mlir::OpPassManager &optPM2 = pm.nestAny();
    optPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM2.addPass(mlir::createCSEPass());
    optPM2.addPass(polygeist::createMem2RegPass());
    optPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM2.addPass(mlir::createCSEPass());
    optPM2.addPass(polygeist::createCanonicalizeForPass());
    if (RaiseToAffine) {
      optPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    }
    optPM2.addPass(polygeist::replaceAffineCFGPass());
    optPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM2.addPass(mlir::createCSEPass());
    if (ParallelLICM)
      optPM2.addPass(polygeist::createParallelLICMPass());
    else
      optPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  }

  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "*** Optimize failed. Module: ***\n";
    module->dump();
    return 6;
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    llvm::errs() << "** Verification after optimization failed. Module: ***\n";
    module->dump();
    return 7;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after optimize ***\n";
    module->dump();
  });

  return 0;
}

// CUDA specific optimization (add parallel loops around CUDA).
static int optimizeCUDA(mlir::MLIRContext &context,
                        mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (!CudaLower)
    return 0;

  constexpr int unrollSize = 32;
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  mlir::PassManager pm(&context);
  enableOptionsPM(pm);

  mlir::OpPassManager &optPM = pm.nestAny();
  optPM.addPass(mlir::createLowerAffinePass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  pm.addPass(polygeist::createParallelLowerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  mlir::OpPassManager &noptPM = pm.nestAny();
  noptPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  noptPM.addPass(polygeist::createMem2RegPass());
  noptPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  pm.addPass(mlir::createInlinerPass());
  mlir::OpPassManager &noptPM2 = pm.nestAny();
  noptPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  noptPM2.addPass(polygeist::createMem2RegPass());
  noptPM2.addPass(polygeist::createCanonicalizeForPass());
  noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  noptPM2.addPass(mlir::createCSEPass());
  if (ParallelLICM)
    noptPM2.addPass(polygeist::createParallelLICMPass());
  else
    noptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
  noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  if (RaiseToAffine) {
    noptPM2.addPass(polygeist::createCanonicalizeForPass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      noptPM2.addPass(polygeist::createParallelLICMPass());
    else
      noptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    noptPM2.addPass(polygeist::replaceAffineCFGPass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (LoopUnroll)
      noptPM2.addPass(mlir::createLoopUnrollPass(unrollSize, false, true));
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    noptPM2.addPass(mlir::createCSEPass());
    noptPM2.addPass(polygeist::createMem2RegPass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      noptPM2.addPass(polygeist::createParallelLICMPass());
    else
      noptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
    noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    noptPM2.addPass(polygeist::replaceAffineCFGPass());
    noptPM2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      pm.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(pm.run(module.get()))) {
    llvm::errs() << "*** Optimize CUDA failed. Module: ***\n";
    module->dump();
    return 8;
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    llvm::errs()
        << "*** Verification after CUDA optimization failed. Module: ***\n";
    module->dump();
    return 9;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "*** Module after optimize CUDA ***\n";
    module->dump();
  });

  return 0;
}

static void finalizeCUDA(mlir::PassManager &pm) {
  if (!CudaLower)
    return;

  mlir::OpPassManager &optPM = pm.nestAny();

  constexpr int unrollSize = 32;
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(polygeist::createMem2RegPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(polygeist::createCanonicalizeForPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));

  if (RaiseToAffine) {
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      optPM.addPass(polygeist::createParallelLICMPass());
    else
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(polygeist::createRaiseSCFToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      pm.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
  if (ToCPU == "continuation") {
    optPM.addPass(polygeist::createBarrierRemovalContinuation());
    // pm.nest<mlir::FuncOp>().addPass(mlir::createCanonicalizerPass());
  } else if (ToCPU.size() != 0) {
    optPM.addPass(polygeist::createCPUifyPass(ToCPU));
  }
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(mlir::createCSEPass());
  optPM.addPass(polygeist::createMem2RegPass());
  optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
  optPM.addPass(mlir::createCSEPass());
  if (RaiseToAffine) {
    optPM.addPass(polygeist::createCanonicalizeForPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      optPM.addPass(polygeist::createParallelLICMPass());
    else
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(polygeist::createRaiseSCFToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (LoopUnroll)
      optPM.addPass(mlir::createLoopUnrollPass(unrollSize, false, true));
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(mlir::createCSEPass());
    optPM.addPass(polygeist::createMem2RegPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ParallelLICM)
      optPM.addPass(polygeist::createParallelLICMPass());
    else
      optPM.addPass(mlir::createLoopInvariantCodeMotionPass());
    optPM.addPass(polygeist::createRaiseSCFToAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(polygeist::replaceAffineCFGPass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (ScalarReplacement)
      pm.addNestedPass<func::FuncOp>(mlir::createAffineScalarReplacementPass());
  }
}

static int finalize(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module,
                    llvm::DataLayout &DL, bool &LinkOMP) {
  mlir::PassManager pm(&context);
  enableOptionsPM(pm);

  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  finalizeCUDA(pm);

  pm.addPass(mlir::createSymbolDCEPass());

  if (EmitLLVM || !EmitAssembly || EmitOpenMPIR) {
    pm.addPass(mlir::createLowerAffinePass());
    if (InnerSerialize)
      pm.addPass(polygeist::createInnerSerializationPass());

    // pm.nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
    if (mlir::failed(pm.run(module.get()))) {
      llvm::errs() << "*** Finalize failed (phase 1). Module: ***\n";
      module->dump();
      return 10;
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      llvm::errs() << "*** Verification after finalization failed (phase 1). "
                      "Module: ***\n";
      module->dump();
      return 11;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize (phase 1) ***\n";
      module->dump();
    });

    mlir::PassManager pm2(&context);
    if (SCFOpenMP) {
      pm2.addPass(createConvertSCFToOpenMPPass());
    }
    pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (OpenMPOpt) {
      pm2.addPass(polygeist::createOpenMPOptPass());
      pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    }
    pm2.addPass(polygeist::createMem2RegPass());
    pm2.addPass(mlir::createCSEPass());
    pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (mlir::failed(pm2.run(module.get()))) {
      llvm::errs() << "*** Finalize failed (phase 2). Module: ***\n";
      module->dump();
      return 12;
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      llvm::errs() << "*** Verification after finalization failed (phase 2). "
                      "Module: ***\n";
      module->dump();
      return 13;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize (phase 2) ***\n";
      module->dump();
    });

    if (!EmitOpenMPIR) {
      module->walk([&](mlir::omp::ParallelOp) { LinkOMP = true; });
      mlir::PassManager pm3(&context);
      pm3.addPass(mlir::sycl::createSYCLMethodToSYCLCallPass());
      pm3.addPass(polygeist::createConvertToLLVMABIPass());
      LowerToLLVMOptions options(&context);
      options.dataLayout = DL;
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(options));
      // pm3.addPass(mlir::createLowerFuncToLLVMPass(options));
      pm3.addPass(polygeist::createLegalizeForSPIRVPass());

      // Needed because SYCLMethodOps lowering might introduce redundant
      // operations.
      pm3.addPass(mlir::createCSEPass());
      pm3.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (mlir::failed(pm3.run(module.get()))) {
        llvm::errs() << "*** Finalize failed (phase 3). Module: ***\n";
        module->dump();
        return 14;
      }
      if (mlir::failed(mlir::verify(module.get()))) {
        llvm::errs() << "Verification after finalization failed (phase 3). "
                        "Module: ***\n";
        module->dump();
        return 15;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "*** Module after finalize (phase 3) ***\n";
        module->dump();
      });
    }
  } else {
    if (mlir::failed(pm.run(module.get()))) {
      llvm::errs() << "*** Finalize failed. Module: ***\n";
      module->dump();
      return 16;
    }
    if (mlir::failed(mlir::verify(module.get()))) {
      llvm::errs() << "*** Verification after finalization failed. "
                      "Module: ***\n";
      module->dump();
      return 17;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "*** Module after finalize ***\n";
      module->dump();
    });
  }

  return 0;
}

// Create and execute the MLIR transformations pipeline.
static int createAndExecutePassPipeline(
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    llvm::DataLayout &DL, llvm::Triple &triple,
    const llvm::OptimizationLevel &OptimizationLevel, bool &LinkOMP) {
  // MLIR canonicalization & cleanup.
  int rc = canonicalize(context, module);
  if (rc != 0)
    return rc;

  // MLIR optimizations.
  rc = optimize(context, OptimizationLevel, module);
  if (rc != 0)
    return rc;

  // CUDA specific MLIR optimizations.
  rc = optimizeCUDA(context, module);
  if (rc != 0)
    return rc;

  rc = finalize(context, module, DL, LinkOMP);
  if (rc != 0)
    return rc;

  return 0;
}

/// Run optimization pipeline in LLVM module.
///
/// Just run default pipelines for now.
static void
runOptimizationPipeline(llvm::Module &module,
                        const llvm::OptimizationLevel &OptimizationLevel) {
  if (OptimizationLevel == llvm::OptimizationLevel::O0)
    return;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassInstrumentationCallbacks PIC;
  llvm::PrintPassOptions PrintPassOpts;
  PrintPassOpts.Verbose = true;
  PrintPassOpts.SkipAnalyses = true;
  llvm::StandardInstrumentations SI(false, true /*VerifyEachPass*/,
                                    PrintPassOpts);
  SI.registerCallbacks(PIC, &FAM);

  TargetMachine *TM = nullptr;
  llvm::Optional<llvm::PGOOptions> P = llvm::None;
  llvm::PipelineTuningOptions PTO;

  llvm::PassBuilder PB(TM, PTO, P, &PIC);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(OptimizationLevel);

  // Before executing passes, print the final values of the LLVM options.
  cl::PrintOptionValues();

  // Print a textual, '-passes=' compatible, representation of pipeline if
  // requested.
  LLVM_DEBUG({
    llvm::dbgs() << "*** Run LLVM Optimization pipeline: ***\n";

    std::string Pipeline;
    raw_string_ostream OS(Pipeline);

    MPM.printPipeline(OS, [&PIC](StringRef ClassName) {
      auto PassName = PIC.getPassNameForClassName(ClassName);
      return PassName.empty() ? ClassName : PassName;
    });
    llvm::dbgs() << Pipeline << "\n";
  });

  // Now that we have all of the passes ready, run them.
  MPM.run(module, MAM);
}

// Lower the MLIR in the given module, compile the generated LLVM IR.
static int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
                         StringRef moduleId, mlir::MLIRContext &context,
                         llvm::DataLayout &DL, llvm::Triple &triple,
                         const llvm::OptimizationLevel &OptimizationLevel,
                         const SmallVectorImpl<const char *> &LinkArgs,
                         const char *Argv0) {
  bool LinkOMP = FOpenMP;
  int rc = createAndExecutePassPipeline(context, module, DL, triple,
                                        OptimizationLevel, LinkOMP);
  if (rc != 0) {
    llvm::errs() << "Failed to execute pass pipeline correctly, rc = " << rc
                 << ".\n";
    return rc;
  }

  bool emitBC = EmitLLVM && !EmitAssembly;
  bool emitMLIR = EmitAssembly && !EmitLLVM;
  if (emitMLIR) {
    if (Output == "-") {
      // Write the MLIR to stdout.
      LLVM_DEBUG(dbgs() << "*** MLIR Produced ***\n");
      module->print(outs());
    } else {
      // Write the MLIR to a file.
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      module->print(out);
      LLVM_DEBUG(dbgs() << "*** Dumped MLIR in file '" << Output << "' ***\n");
    }
  } else {
    // Generate LLVM IR.
    llvm::LLVMContext llvmContext;
    auto llvmModule =
        mlir::translateModuleToLLVMIR(module.get(), llvmContext, moduleId);
    if (!llvmModule) {
      module->dump();
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }

    llvmModule->setDataLayout(DL);
    llvmModule->setTargetTriple(triple.getTriple());
    LLVM_DEBUG(dbgs() << "*** Translated MLIR to LLVM IR successfully ***\n");

    if (emitBC || EmitLLVM) {
      // Not needed when emitting binary for now; will be handled by the driver.
      runOptimizationPipeline(*llvmModule, OptimizationLevel);
    }

    if (emitBC) {
      assert(Output != "-" && "Expecting output file");
      // Write the LLVM BC to a file.
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      WriteBitcodeToFile(*llvmModule, out);
      LLVM_DEBUG(dbgs() << "*** Dumped LLVM BC in file '" << Output
                        << "' ***\n");
    } else if (EmitLLVM) {
      if (Output == "-") {
        // Write the LLVM IR to stdout.
        LLVM_DEBUG(dbgs() << "*** LLVM IR Produced ***\n");
        llvm::outs() << *llvmModule << "\n";
      } else {
        // Write the LLVM IR to a file.
        std::error_code EC;
        llvm::raw_fd_ostream out(Output, EC);
        out << *llvmModule << "\n";
        LLVM_DEBUG(dbgs() << "*** Dumped LLVM IR in file '" << Output
                          << "' ***\n");
      }
    } else {
      // Compile the LLVM IR.
      auto tmpFile =
          llvm::sys::fs::TempFile::create("/tmp/intermediate%%%%%%%.ll");
      if (!tmpFile) {
        llvm::errs() << "Failed to create " << tmpFile->TmpName << "\n";
        return -1;
      }
      llvm::raw_fd_ostream out(tmpFile->FD, /*shouldClose*/ false);
      out << *llvmModule << "\n";
      out.flush();

      int res = emitBinary(Argv0, tmpFile->TmpName.c_str(), LinkArgs, LinkOMP);
      if (res != 0)
        llvm::errs() << "Compilation failed\n";

      if (tmpFile->discard()) {
        llvm::errs() << "Failed to erase " << tmpFile->TmpName << "\n";
        return -1;
      }
      return res;
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
    llvm::WithColor::warning() << "optimization level '-O" << OptimizationLevel
                               << "' is not supported; using '-O3' instead\n";
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
  bool syclIsDevice = false;
  bool linkOnly = false;
  llvm::OptimizationLevel OptimizationLevel =
      CgeistOptions::getDefaultOptimizationLevel();

  for (int i = 0; i < argc; i++) {
    StringRef ref(argv[i]);
    if (ref == "-Wl,--start-group")
      linkOnly = true;
    if (!linkOnly) {
      if (ref == "-fPIC" || ref == "-c" || ref.startswith("-fsanitize"))
        LinkageArgs.push_back(argv[i]);
      else if (ref == "-L" || ref == "-l") {
        LinkageArgs.push_back(argv[i]);
        i++;
        LinkageArgs.push_back(argv[i]);
      } else if (ref.startswith("-L") || ref.startswith("-l") ||
                 ref.startswith("-Wl"))
        LinkageArgs.push_back(argv[i]);
      else if (ref == "-D" || ref == "-I") {
        MLIRArgs.push_back(argv[i]);
        i++;
        MLIRArgs.push_back(argv[i]);
      } else if (ref.startswith("-D")) {
        MLIRArgs.push_back("-D");
        MLIRArgs.push_back(&argv[i][2]);
      } else if (ref.startswith("-I")) {
        MLIRArgs.push_back("-I");
        MLIRArgs.push_back(&argv[i][2]);
      } else if (ref == "-fsycl-is-device") {
        syclIsDevice = true;
        MLIRArgs.push_back(argv[i]);
      } else if (ref.consume_front("-O") || ref.consume_front("--optimize")) {
        // If several flags are passed, we keep the last one.
        OptimizationLevel = ExitOnErr(parseOptimizationLevel(ref));
        LinkageArgs.push_back(argv[i]);
      } else if (ref == "-g")
        LinkageArgs.push_back(argv[i]);
      else
        MLIRArgs.push_back(argv[i]);
    } else
      LinkageArgs.push_back(argv[i]);

    if (ref == "-Wl,--end-group")
      linkOnly = false;
  }

  return {syclIsDevice, OptimizationLevel};
}

// Fill the module with the MLIR in the inputFile.
static void loadMLIR(const std::string &inputFile,
                     mlir::OwningOpRef<ModuleOp> &module,
                     mlir::MLIRContext &context) {
  assert(inputFile.substr(inputFile.find_last_of(".") + 1) == "mlir" &&
         "Input file has incorrect extension");

  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> input =
      mlir::openInputFile(inputFile, &errorMsg);
  if (!input) {
    llvm::errs() << errorMsg << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFile << "\n";
    exit(1);
  }
}

// Generate MLIR for the input files.
static void processInputFiles(const cl::list<std::string> &inputFiles,
                              const cl::list<std::string> &inputCommandArgs,
                              mlir::MLIRContext &context,
                              mlir::OwningOpRef<ModuleOp> &module,
                              llvm::DataLayout &DL, llvm::Triple &triple,
                              const char *Argv0, bool syclIsDevice) {
  assert(!inputFiles.empty() && "inputFiles should not be empty");

  // Ensure all input files can be opened.
  std::vector<std::string> files;
  for (auto inputFile : inputFiles) {
    std::ifstream ifs(inputFile);
    if (!ifs.good()) {
      llvm::errs() << "Not able to open file: " << inputFile << "\n";
      exit(-1);
    }
    files.push_back(inputFile);
  }

  std::vector<std::string> commands;
  for (auto cmd : inputCommandArgs)
    commands.push_back(cmd);

  // Count the number of MLIR input files we might have.
  int numMLIRFiles =
      llvm::count_if(inputFiles, [](const std::string &inputFile) {
        std::string extension =
            inputFile.substr(inputFile.find_last_of(".") + 1);
        return (extension == "mlir");
      });

  // Early exit if we have a input file with a '.mlir' extension.
  if (numMLIRFiles > 0) {
    if (files.size() != 1) {
      llvm::errs()
          << "More than one input file has been provided. Only a single "
             "input MLIR file can be processed\n";
      exit(-1);
    }
    loadMLIR(files[0], module, context);
    return;
  }

  // Generate MLIR for the C/C++ files.
  std::string fn = (!syclIsDevice) ? Cfunction.getValue() : "";
  parseMLIR(Argv0, files, fn, IncludeDirs, Defines, module, triple, DL,
            commands);
}

static bool containsFunctions(mlir::gpu::GPUModuleOp deviceModule) {
  Region &rgn = deviceModule.getRegion();
  return !rgn.getOps<mlir::gpu::GPUFuncOp>().empty() ||
         !rgn.getOps<mlir::func::FuncOp>().empty();
}

static void eraseHostCode(mlir::ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Erasing host code\n");
  SmallVector<std::reference_wrapper<Operation>> toRemove;
  std::copy_if(module.begin(), module.end(), std::back_inserter(toRemove),
               [](Operation &op) { return !isa<mlir::gpu::GPUModuleOp>(op); });
  for (auto op : toRemove)
    op.get().erase();
}

int main(int argc, char **argv) {
  if (argc >= 1 && std::string(argv[1]) == "-cc1") {
    SmallVector<const char *> Argv;
    for (int i = 0; i < argc; i++)
      Argv.push_back(argv[i]);
    return ExecuteCC1Tool(Argv);
  }

  // Split up the arguments into MLIR and linkage arguments.
  SmallVector<const char *> LinkageArgs, MLIRArgs;
  const CgeistOptions Options =
      splitCommandLineOptions(argc, argv, LinkageArgs, MLIRArgs);
  assert(!MLIRArgs.empty() && "MLIRArgs should not be empty");

  // Register any command line options.
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();

  // Parse command line options.
  llvm::cl::list<std::string> inputFileNames(
      llvm::cl::Positional, llvm::cl::OneOrMore,
      llvm::cl::desc("<Specify input file>"), llvm::cl::cat(ToolOptions));

  llvm::cl::list<std::string> inputCommandArgs(
      "args", llvm::cl::Positional, llvm::cl::desc("<command arguments>"),
      llvm::cl::ZeroOrMore, llvm::cl::PositionalEatsArgs);

  int size = MLIRArgs.size();
  const char **data = MLIRArgs.data();
  InitLLVM y(size, data);
  llvm::cl::ParseCommandLineOptions(size, data);

  // Register MLIR dialects.
  mlir::MLIRContext context;
  registerDialects(context, Options.syclIsDevice());

  // Generate MLIR for the input files.
  mlir::OpBuilder Builder(&context);
  const auto loc = Builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  Builder.setInsertionPointToEnd(module->getBody());
  auto deviceModule = Builder.create<mlir::gpu::GPUModuleOp>(
      loc, MLIRASTConsumer::DeviceModuleName);

  llvm::DataLayout DL("");
  llvm::Triple triple;
  processInputFiles(inputFileNames, inputCommandArgs, context, module, DL,
                    triple, argv[0], Options.syclIsDevice());

  LLVM_DEBUG({
    llvm::dbgs() << "Initial MLIR:\n";
    module->dump();
  });

  // For now, we will work on the device code if it contains any functions and
  // on the host code otherwise.
  if (containsFunctions(deviceModule)) {
    eraseHostCode(*module);
    module.get()->setAttr(mlir::gpu::GPUDialect::getContainerModuleAttrName(),
                          Builder.getUnitAttr());
  } else
    deviceModule.erase();

  LLVM_DEBUG({
    llvm::dbgs() << "MLIR before compilation:\n";
    module->dump();
  });

  // Lower the MLIR to LLVM IR, compile the generated LLVM IR.
  return compileModule(module,
                       inputFileNames.size() == 1 ? inputFileNames[0]
                                                  : "LLVMDialectModule",
                       context, DL, triple, Options.getOptimizationLevel(),
                       LinkageArgs, argv[0]);
}
