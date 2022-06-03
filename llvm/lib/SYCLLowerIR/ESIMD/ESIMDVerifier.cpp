//===---------- ESIMDVerifier.cpp - ESIMD-specific IR verification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ESIMD specific IR verification pass. So far it only
// detects invalid API calls in ESIMD context.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDVerifier.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
namespace id = itanium_demangle;

#define DEBUG_TYPE "esimd-verifier"

// A list of SYCL functions (regexps) allowed for use in ESIMD context.
static const char *LegalSYCLFunctions[] = {
    "^cl::sycl::accessor<.+>::accessor",
    "^cl::sycl::accessor<.+>::~accessor",
    "^cl::sycl::accessor<.+>::getNativeImageObj",
    "^cl::sycl::accessor<.+>::__init_esimd",
    "^cl::sycl::id<.+>::.+",
    "^cl::sycl::item<.+>::.+",
    "^cl::sycl::nd_item<.+>::.+",
    "^cl::sycl::group<.+>::.+",
    "^cl::sycl::sub_group<.+>::.+",
    "^cl::sycl::range<.+>::.+",
    "^cl::sycl::kernel_handler::.+",
    "^cl::sycl::cos<.+>",
    "^cl::sycl::sin<.+>",
    "^cl::sycl::log<.+>",
    "^cl::sycl::exp<.+>",
    "^cl::sycl::operator.+<.+>",
    "^cl::sycl::ext::oneapi::sub_group::.+",
    "^cl::sycl::ext::oneapi::experimental::spec_constant<.+>::.+",
    "^cl::sycl::ext::oneapi::experimental::this_sub_group"};

namespace {

// Simplest possible implementation of an allocator for the Itanium demangler
class SimpleAllocator {
protected:
  SmallVector<void *, 128> Ptrs;

public:
  void reset() {
    for (void *Ptr : Ptrs) {
      // Destructors are not called, but that is OK for the
      // itanium_demangle::Node subclasses
      std::free(Ptr);
    }
    Ptrs.resize(0);
  }

  template <typename T, typename... Args> T *makeNode(Args &&...args) {
    void *Ptr = std::calloc(1, sizeof(T));
    Ptrs.push_back(Ptr);
    return new (Ptr) T(std::forward<Args>(args)...);
  }

  void *allocateNodeArray(size_t sz) {
    void *Ptr = std::calloc(sz, sizeof(id::Node *));
    Ptrs.push_back(Ptr);
    return Ptr;
  }

  ~SimpleAllocator() { reset(); }
};

class ESIMDVerifierImpl {
  const Module &M;

public:
  ESIMDVerifierImpl(const Module &M) : M(M) {}

  void verify() {
    SmallPtrSet<const Function *, 8u> Visited;
    SmallVector<const Function *, 8u> Worklist;

    auto Add2Worklist = [&Worklist, &Visited](const Function *F) {
      if (Visited.insert(F).second)
        Worklist.push_back(F);
    };

    // Start with adding all ESIMD functions to the work list.
    for (const Function &F : M)
      if (F.hasMetadata("sycl_explicit_simd"))
        Add2Worklist(&F);

    // Then check ESIMD functions and all functions called from ESIMD context
    // for invalid calls.
    while (!Worklist.empty()) {
      const Function *F = Worklist.pop_back_val();
      for (const Instruction &I : instructions(F)) {
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Function *Callee = CB->getCalledFunction();
          if (!Callee)
            continue;

          // Add callee to the list to be analyzed if it is not a declaration.
          if (!Callee->isDeclaration())
            Add2Worklist(Callee);

          // Demangle called function name and check if it is legal to use this
          // function in ESIMD context.
          StringRef MangledName = Callee->getName();
          id::ManglingParser<SimpleAllocator> Parser(MangledName.begin(),
                                                     MangledName.end());
          id::Node *AST = Parser.parse();
          if (!AST || AST->getKind() != id::Node::KFunctionEncoding)
            continue;

          auto *FE = static_cast<id::FunctionEncoding *>(AST);
          const id::Node *NameNode = FE->getName();
          if (!NameNode) // Can it be null?
            continue;

          id::OutputBuffer NameBuf;
          NameNode->print(NameBuf);
          StringRef Name(NameBuf.getBuffer(), NameBuf.getCurrentPosition());

          // We are interested in functions defined in SYCL namespace, but
          // outside of ESIMD namespaces.
          if (!Name.startswith("cl::sycl::") ||
              Name.startswith("cl::sycl::detail::") ||
              Name.startswith("cl::sycl::ext::intel::esimd::") ||
              Name.startswith("cl::sycl::ext::intel::experimental::esimd::"))
            continue;

          // Check if function name matches any allowed SYCL function name.
          if (any_of(LegalSYCLFunctions, [Name](const char *LegalName) {
                Regex LegalNameRE(LegalName);
                assert(LegalNameRE.isValid() && "invalid function name regex");
                return LegalNameRE.match(Name);
              }))
            continue;

          // If not, report an error.
          std::string ErrorMsg = std::string("function '") +
                                 demangle(MangledName.str()) +
                                 "' is not supported in ESIMD context";
          F->getContext().emitError(&I, ErrorMsg);
        }
      }
    }
  }
};

} // end anonymous namespace

PreservedAnalyses ESIMDVerifierPass::run(Module &M, ModuleAnalysisManager &AM) {
  ESIMDVerifierImpl(M).verify();
  return PreservedAnalyses::all();
}

namespace {

struct ESIMDVerifier : public ModulePass {
  static char ID;

  ESIMDVerifier() : ModulePass(ID) {
    initializeESIMDVerifierPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    ESIMDVerifierImpl(M).verify();
    return false;
  }
};

} // end anonymous namespace

char ESIMDVerifier::ID = 0;

INITIALIZE_PASS_BEGIN(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                      false, false)
INITIALIZE_PASS_END(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                    false, false)

ModulePass *llvm::createESIMDVerifierPass() { return new ESIMDVerifier(); }
