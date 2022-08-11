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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "SYCL/SYCLOps.h.inc"
#include "SYCL/SYCLOpsTypes.h.inc"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include <fstream>

#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"
#include "Options.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
  llvm::cl::ExpandResponseFiles(Saver, &llvm::cl::TokenizeGNUCommandLine, ArgV,
                                /*MarkEOLs=*/false);
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

static int emitBinary(char *Argv0, const char *filename,
                      SmallVectorImpl<const char *> &LinkArgs, bool LinkOMP) {

  using namespace clang;
  using namespace clang::driver;
  using namespace std;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticPrinter *DiagBuffer =
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagBuffer);

  string TargetTriple;
  if (TargetTripleOpt == "")
    TargetTriple = llvm::sys::getDefaultTargetTriple();
  else
    TargetTriple = TargetTripleOpt;

  const char *binary = Argv0;
  const unique_ptr<Driver> driver(new Driver(binary, TargetTriple, Diags));
  driver->CC1Main = &ExecuteCC1Tool;
  std::vector<const char *> Argv;
  Argv.push_back(Argv0);
  // Argv.push_back("-x");
  // Argv.push_back("ir");
  Argv.push_back(filename);
  if (LinkOMP)
    Argv.push_back("-fopenmp");
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    char *chars = (char *)malloc(ResourceDir.length() + 1);
    memcpy(chars, ResourceDir.data(), ResourceDir.length());
    chars[ResourceDir.length()] = 0;
    Argv.push_back(chars);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (CUDAGPUArch != "") {
    auto a = "--cuda-gpu-arch=" + CUDAGPUArch;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (CUDAPath != "") {
    auto a = "--cuda-path=" + CUDAPath;
    char *chars = (char *)malloc(a.length() + 1);
    memcpy(chars, a.data(), a.length());
    chars[a.length()] = 0;
    Argv.push_back(chars);
  }
  if (Opt0) {
    Argv.push_back("-O0");
  }
  if (Opt1) {
    Argv.push_back("-O1");
  }
  if (Opt2) {
    Argv.push_back("-O2");
  }
  if (Opt3) {
    Argv.push_back("-O3");
  }
  if (Output != "") {
    Argv.push_back("-o");
    char *chars = (char *)malloc(Output.length() + 1);
    memcpy(chars, Output.data(), Output.length());
    chars[Output.length()] = 0;
    Argv.push_back(chars);
  }
  for (const auto *arg : LinkArgs)
    Argv.push_back(arg);

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));

  if (ResourceDir != "")
    driver->ResourceDir = ResourceDir;
  if (SysRoot != "")
    driver->SysRoot = SysRoot;
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  int Res = 0;

  driver->ExecuteCompilation(*compilation, FailingCommands);
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
      driver->generateCompilationDiagnostics(*compilation, *FailingCommand);
      break;
    }
  }
  Diags.getClient()->finish();

  return Res;
}

#include "Lib/clang-mlir.cc"

// Register MLIR Dialects.
static void registerDialects(mlir::DialectRegistry &registry) {
  mlir::registerOpenMPDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
}

// Load MLIR Dialects.
static void loadDialects(MLIRContext &context, const bool syclKernelsOnly) {
  context.disableMultithreading();
  context.getOrLoadDialect<AffineDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<DLTIDialect>();
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

  if (syclKernelsOnly) {
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

// MLIR canonicalization & cleanup.
static int canonicalize(mlir::MLIRContext &context,
                        mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);
  pm.enableVerifier(EarlyVerifier);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
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
    optPM.addPass(mlir::createAffineScalarReplacementPass());
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
      optPM.addPass(mlir::createAffineScalarReplacementPass());
  }
  if (mlir::failed(pm.run(module.get()))) {
    module->dump();
    return 4;
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    module->dump();
    return 5;
  }

  return 0;
}

// Optimize the MLIR.
static int optimize(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module) {
  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  if (DetectReduction)
    optPM.addPass(polygeist::detectReductionPass());

  if (!Opt0) {
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    optPM.addPass(mlir::createCSEPass());
    // Affine must be lowered to enable inlining
    if (RaiseToAffine)
      optPM.addPass(mlir::createLowerAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    pm.addPass(mlir::createInlinerPass());
    mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();
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
    module->dump();
    return 4;
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    module->dump();
    return 5;
  }

  return 0;
}

// CUDA specific optimization (add parallel loops around CUDA).
static int optimizeCUDA(mlir::MLIRContext &context,
                        mlir::OwningOpRef<mlir::ModuleOp> &module) {
  constexpr int unrollSize = 32;
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  if (CudaLower) {
    mlir::PassManager pm(&context);
    mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createLowerAffinePass());
    optPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    pm.addPass(polygeist::createParallelLowerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    mlir::OpPassManager &noptPM = pm.nest<mlir::func::FuncOp>();
    noptPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    noptPM.addPass(polygeist::createMem2RegPass());
    noptPM.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    pm.addPass(mlir::createInlinerPass());
    mlir::OpPassManager &noptPM2 = pm.nest<mlir::func::FuncOp>();
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
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (ParallelLICM)
        noptPM2.addPass(polygeist::createParallelLICMPass());
      else
        noptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
      noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      noptPM2.addPass(polygeist::replaceAffineCFGPass());
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (LoopUnroll)
        noptPM2.addPass(mlir::createLoopUnrollPass(unrollSize, false, true));
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      noptPM2.addPass(mlir::createCSEPass());
      noptPM2.addPass(polygeist::createMem2RegPass());
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (ParallelLICM)
        noptPM2.addPass(polygeist::createParallelLICMPass());
      else
        noptPM2.addPass(mlir::createLoopInvariantCodeMotionPass());
      noptPM2.addPass(polygeist::createRaiseSCFToAffinePass());
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      noptPM2.addPass(polygeist::replaceAffineCFGPass());
      noptPM2.addPass(
          mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (ScalarReplacement)
        noptPM2.addPass(mlir::createAffineScalarReplacementPass());
    }
    if (mlir::failed(pm.run(module.get()))) {
      module->dump();
      return 4;
    }
  }

  return 0;
}

static int finalize(mlir::MLIRContext &context,
                    mlir::OwningOpRef<mlir::ModuleOp> &module,
                    llvm::DataLayout &DL, bool &LinkOMP) {
  constexpr int unrollSize = 32;
  GreedyRewriteConfig canonicalizerConfig;
  canonicalizerConfig.maxIterations = CanonicalizeIterations;

  mlir::PassManager pm(&context);
  mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();

  if (CudaLower) {
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
        optPM.addPass(mlir::createAffineScalarReplacementPass());
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
        optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
  }
  pm.addPass(mlir::createSymbolDCEPass());

  if (EmitLLVM || !EmitAssembly || EmitOpenMPIR) {
    pm.addPass(mlir::createLowerAffinePass());
    if (InnerSerialize)
      pm.addPass(polygeist::createInnerSerializationPass());

    // pm.nest<mlir::FuncOp>().addPass(mlir::createConvertMathToLLVMPass());
    if (mlir::failed(pm.run(module.get()))) {
      module->dump();
      return 4;
    }
    mlir::PassManager pm2(&context);
    if (SCFOpenMP) {
      pm2.addPass(createConvertSCFToOpenMPPass());
    }
    pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (OpenMPOpt) {
      pm2.addPass(polygeist::createOpenMPOptPass());
      pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    }
    pm2.nest<mlir::func::FuncOp>().addPass(polygeist::createMem2RegPass());
    pm2.addPass(mlir::createCSEPass());
    pm2.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
    if (mlir::failed(pm2.run(module.get()))) {
      module->dump();
      return 4;
    }
    if (!EmitOpenMPIR) {
      module->walk([&](mlir::omp::ParallelOp) { LinkOMP = true; });
      mlir::PassManager pm3(&context);
      LowerToLLVMOptions options(&context);
      options.dataLayout = DL;
      // invalid for gemm.c init array
      // options.useBarePtrCallConv = true;
      pm3.addPass(polygeist::createConvertPolygeistToLLVMPass(options));
      // pm3.addPass(mlir::createLowerFuncToLLVMPass(options));
      pm3.addPass(mlir::createCanonicalizerPass(canonicalizerConfig, {}, {}));
      if (mlir::failed(pm3.run(module.get()))) {
        module->dump();
        return 4;
      }
    }
  } else {
    if (mlir::failed(pm.run(module.get()))) {
      module->dump();
      return 4;
    }
  }
  if (mlir::failed(mlir::verify(module.get()))) {
    module->dump();
    return 5;
  }

  return 0;
}

// Create and execute the MLIR transformations pipeline.
static int
createAndExecutePassPipeline(mlir::MLIRContext &context,
                             mlir::OwningOpRef<mlir::ModuleOp> &module,
                             llvm::DataLayout &DL, llvm::Triple &triple, bool &LinkOMP) {
  // MLIR canonicalization & cleanup.
  int rc = canonicalize(context, module);
  if (rc != 0) 
    return rc;

  // MLIR optimizations.
  rc = optimize(context, module);
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

int main(int argc, char **argv) {
  bool syclKernelsOnly = false;
  if (argc >= 1) {
    if (std::string(argv[1]) == "-cc1") {
      SmallVector<const char *> Argv;
      for (int i = 0; i < argc; i++)
        Argv.push_back(argv[i]);
      return ExecuteCC1Tool(Argv);
    }
  }
  SmallVector<const char *> LinkageArgs;
  SmallVector<const char *> MLIRArgs;
  {
    bool linkOnly = false;
    for (int i = 0; i < argc; i++) {
      StringRef ref(argv[i]);
      if (ref == "-Wl,--start-group")
        linkOnly = true;
      if (!linkOnly) {
        if (ref == "-fPIC" || ref == "-c" || ref.startswith("-fsanitize")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-L" || ref == "-l") {
          LinkageArgs.push_back(argv[i]);
          i++;
          LinkageArgs.push_back(argv[i]);
        } else if (ref.startswith("-L") || ref.startswith("-l") ||
                   ref.startswith("-Wl")) {
          LinkageArgs.push_back(argv[i]);
        } else if (ref == "-D" || ref == "-I") {
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
          syclKernelsOnly = true;
          MLIRArgs.push_back(argv[i]);
        } else if (ref == "-g") {
          LinkageArgs.push_back(argv[i]);
        } else {
          MLIRArgs.push_back(argv[i]);
        }
      } else {
        LinkageArgs.push_back(argv[i]);
      }
      if (ref == "-Wl,--end-group")
        linkOnly = false;
    }
  }
  using namespace mlir;

  std::vector<std::string> files;
  std::vector<std::string> commands;
  cl::list<std::string> inputFileName(cl::Positional, cl::OneOrMore,
                                      cl::desc("<Specify input file>"),
                                      cl::cat(toolOptions));

  cl::list<std::string> inputCommandArgs(
      "args", cl::Positional, cl::desc("<command arguments>"), cl::ZeroOrMore,
      cl::PositionalEatsArgs);

  int size = MLIRArgs.size();
  const char **data = MLIRArgs.data();
  InitLLVM y(size, data);
  cl::ParseCommandLineOptions(size, data);
  assert(inputFileName.size());
  for (auto inp : inputFileName) {
    std::ifstream inputFile(inp);
    if (!inputFile.good()) {
      outs() << "Not able to open file: " << inp << "\n";
      return -1;
    }
    files.push_back(inp);
  }
  for (auto &cmd : inputCommandArgs) {
    commands.push_back(cmd);
  }

  // Register and load MLIR Dialects.
  mlir::DialectRegistry registry;
  registerDialects(registry);

  mlir::MLIRContext context(registry);
  loadDialects(context, syclKernelsOnly);

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::ModuleOp::create(mlir::OpBuilder(&context).getUnknownLoc()));

  llvm::Triple triple;
  llvm::DataLayout DL("");
  std::string fn;

  if (!syclKernelsOnly) {
    fn = cfunction.getValue();
  }
  parseMLIR(argv[0], files, fn, includeDirs, defines, module, triple, DL,
            commands);

  if (ImmediateMLIR) {
    llvm::errs() << "<immediate: mlir>\n";
    module->dump();
    llvm::errs() << "</immediate: mlir>\n";
  }

  bool LinkOMP = FOpenMP;
  int rc = createAndExecutePassPipeline(context, module, DL, triple, LinkOMP);
  if (rc != 0)
    return rc;

  if (EmitLLVM || !EmitAssembly) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
    if (!llvmModule) {
      module->dump();
      llvm::errs() << "Failed to emit LLVM IR\n";
      return -1;
    }
    llvmModule->setDataLayout(DL);
    llvmModule->setTargetTriple(triple.getTriple());
    if (!EmitAssembly) {
      auto tmpFile =
          llvm::sys::fs::TempFile::create("/tmp/intermediate%%%%%%%.ll");
      if (!tmpFile) {
        llvm::errs() << "Failed to create temp file\n";
        return -1;
      }
      std::error_code EC;
      {
        llvm::raw_fd_ostream out(tmpFile->FD, /*shouldClose*/ false);
        out << *llvmModule << "\n";
        out.flush();
      }
      int res =
          emitBinary(argv[0], tmpFile->TmpName.c_str(), LinkageArgs, LinkOMP);
      if (tmpFile->discard()) {
        llvm::errs() << "Failed to erase temp file\n";
        return -1;
      }
      return res;
    } else if (Output == "-") {
      llvm::outs() << *llvmModule << "\n";
    } else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      out << *llvmModule << "\n";
    }

  } else {
    if (Output == "-")
      module->print(outs());
    else {
      std::error_code EC;
      llvm::raw_fd_ostream out(Output, EC);
      module->print(out);
    }
  }
  return 0;
}
