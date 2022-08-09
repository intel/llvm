//===- Options.h - cgeist command line options ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command line flags for cgeist.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_OPTIONS_H_
#define CGEIST_OPTIONS_H_

#include "llvm/Support/CommandLine.h"
#include <string>

static llvm::cl::OptionCategory toolOptions("clang to mlir - tool options");

static llvm::cl::opt<bool>
    CudaLower("cuda-lower", llvm::cl::init(false),
              llvm::cl::desc("Add parallel loops around cuda"));

static llvm::cl::opt<bool> EmitLLVM("emit-llvm", llvm::cl::init(false),
                                    llvm::cl::desc("Emit llvm"));

static llvm::cl::opt<bool> EmitOpenMPIR("emit-openmpir", llvm::cl::init(false),
                                        llvm::cl::desc("Emit OpenMP IR"));

static llvm::cl::opt<bool> EmitAssembly("S", llvm::cl::init(false),
                                        llvm::cl::desc("Emit Assembly"));

static llvm::cl::opt<bool> Opt0("O0", llvm::cl::init(false),
                                llvm::cl::desc("Opt level 0"));
static llvm::cl::opt<bool> Opt1("O1", llvm::cl::init(false),
                                llvm::cl::desc("Opt level 1"));
static llvm::cl::opt<bool> Opt2("O2", llvm::cl::init(false),
                                llvm::cl::desc("Opt level 2"));
static llvm::cl::opt<bool> Opt3("O3", llvm::cl::init(false),
                                llvm::cl::desc("Opt level 3"));

static llvm::cl::opt<bool> SCFOpenMP("scf-openmp", llvm::cl::init(true),
                                     llvm::cl::desc("Emit llvm"));

static llvm::cl::opt<bool> OpenMPOpt("openmp-opt", llvm::cl::init(true),
                                     llvm::cl::desc("Turn on openmp opt"));

static llvm::cl::opt<bool>
    ParallelLICM("parallel-licm", llvm::cl::init(true),
                 llvm::cl::desc("Turn on parallel licm"));

static llvm::cl::opt<bool>
    InnerSerialize("inner-serialize", llvm::cl::init(false),
                   llvm::cl::desc("Turn on parallel licm"));

static llvm::cl::opt<bool> ShowAST("show-ast", llvm::cl::init(false),
                                   llvm::cl::desc("Show AST"));

static llvm::cl::opt<bool> ImmediateMLIR("immediate", llvm::cl::init(false),
                                         llvm::cl::desc("Emit immediate mlir"));

static llvm::cl::opt<bool> RaiseToAffine("raise-scf-to-affine",
                                         llvm::cl::init(false),
                                         llvm::cl::desc("Raise SCF to Affine"));

static llvm::cl::opt<bool>
    ScalarReplacement("scal-rep", llvm::cl::init(true),
                      llvm::cl::desc("Raise SCF to Affine"));

static llvm::cl::opt<bool> LoopUnroll("unroll-loops", llvm::cl::init(true),
                                      llvm::cl::desc("Unroll Affine Loops"));

static llvm::cl::opt<bool>
    DetectReduction("detect-reduction", llvm::cl::init(false),
                    llvm::cl::desc("Detect reduction in inner most loop"));

static llvm::cl::opt<std::string> Standard("std", llvm::cl::init(""),
                                           llvm::cl::desc("C/C++ std"));

static llvm::cl::opt<std::string> CUDAGPUArch("cuda-gpu-arch",
                                              llvm::cl::init(""),
                                              llvm::cl::desc("CUDA GPU arch"));

static llvm::cl::opt<std::string> CUDAPath("cuda-path", llvm::cl::init(""),
                                           llvm::cl::desc("CUDA Path"));

static llvm::cl::opt<bool>
    NoCUDAInc("nocudainc", llvm::cl::init(false),
              llvm::cl::desc("Do not include CUDA headers"));

static llvm::cl::opt<bool>
    NoCUDALib("nocudalib", llvm::cl::init(false),
              llvm::cl::desc("Do not link CUDA libdevice"));

static llvm::cl::opt<std::string> Output("o", llvm::cl::init("-"),
                                         llvm::cl::desc("Output file"));

static llvm::cl::opt<std::string>
    cfunction("function", llvm::cl::desc("<Specify function>"),
              llvm::cl::init("main"), llvm::cl::cat(toolOptions));

static llvm::cl::opt<bool> FOpenMP("fopenmp", llvm::cl::init(false),
                                   llvm::cl::desc("Enable OpenMP"));

static llvm::cl::opt<std::string> ToCPU("cpuify", llvm::cl::init(""),
                                        llvm::cl::desc("Convert to cpu"));

static llvm::cl::opt<std::string> MArch("march", llvm::cl::init(""),
                                        llvm::cl::desc("Architecture"));

static llvm::cl::opt<std::string> ResourceDir("resource-dir",
                                              llvm::cl::init(""),
                                              llvm::cl::desc("Resource-dir"));

static llvm::cl::opt<std::string> SysRoot("sysroot", llvm::cl::init(""),
                                          llvm::cl::desc("sysroot"));

static llvm::cl::opt<bool>
    EarlyVerifier("early-verifier", llvm::cl::init(false),
                  llvm::cl::desc("Enable verifier ASAP"));

static llvm::cl::opt<bool> Verbose("v", llvm::cl::init(false),
                                   llvm::cl::desc("Verbose"));

static llvm::cl::list<std::string>
    includeDirs("I", llvm::cl::desc("include search path"),
                llvm::cl::cat(toolOptions));

static llvm::cl::list<std::string> defines("D", llvm::cl::desc("defines"),
                                           llvm::cl::cat(toolOptions));

static llvm::cl::list<std::string>
    Includes("include", llvm::cl::desc("includes"), llvm::cl::cat(toolOptions));

static llvm::cl::opt<std::string>
    TargetTripleOpt("target", llvm::cl::init(""),
                    llvm::cl::desc("Target triple"),
                    llvm::cl::cat(toolOptions));

static llvm::cl::opt<int> CanonicalizeIterations(
    "canonicalizeiters", llvm::cl::init(400),
    llvm::cl::desc("Number of canonicalization iterations"));

static llvm::cl::opt<std::string> McpuOpt("mcpu", llvm::cl::init(""),
                                          llvm::cl::desc("Target CPU"),
                                          llvm::cl::cat(toolOptions));

#endif /* CGEIST_OPTIONS_H_ */
