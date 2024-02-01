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
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace llvm::esimd;
namespace id = itanium_demangle;

#define DEBUG_TYPE "esimd-verifier"

// A list of SYCL functions (regexps) allowed for use in ESIMD context.
// clang-format off
static const char *LegalSYCLFunctions[] = {
    "^sycl::_V1::accessor<.+>::accessor",
    "^sycl::_V1::accessor<.+>::~accessor",
    "^sycl::_V1::accessor<.+>::getQualifiedPtr",
    "^sycl::_V1::accessor<.+>::__init_esimd",
    "^sycl::_V1::address_space_cast",
    "^sycl::_V1::local_accessor<.+>::local_accessor",
    "^sycl::_V1::local_accessor<.+>::__init_esimd",
    "^sycl::_V1::local_accessor<.+>::get_pointer",
    "^sycl::_V1::local_accessor<.+>::get_multi_ptr",
    "^sycl::_V1::local_accessor_base<.+>::local_accessor_base",
    "^sycl::_V1::local_accessor_base<.+>::__init_esimd",
    "^sycl::_V1::local_accessor_base<.+>::getQualifiedPtr",
    "^sycl::_V1::local_accessor_base<.+>::getSize",
    "^sycl::_V1::local_accessor_base<.+>::operator\\[\\]",
    "^sycl::_V1::ext::oneapi::experimental::printf",
    "^sycl::_V1::id<.+>::.+",
    "^sycl::_V1::item<.+>::.+",
    "^sycl::_V1::multi_ptr<.+>::.+",
    "^sycl::_V1::nd_item<.+>::.+",
    "^sycl::_V1::group<.+>::.+",
    "^sycl::_V1::sub_group::.+",
    "^sycl::_V1::range<.+>::.+",
    "^sycl::_V1::kernel_handler::.+",
    "^sycl::_V1::cos",
    "^sycl::_V1::sin",
    "^sycl::_V1::log",
    "^sycl::_V1::exp",
    "^sycl::_V1::bit_cast<.+>",
    "^sycl::_V1::operator.+<.+>",
    "^sycl::_V1::ext::oneapi::experimental::properties",
    "^sycl::_V1::ext::oneapi::experimental::detail::ExtractProperties",
    "^sycl::_V1::ext::oneapi::sub_group::.+",
    "^sycl::_V1::ext::oneapi::experimental::spec_constant<.+>::.+",
    "^sycl::_V1::ext::oneapi::experimental::this_sub_group",
    "^sycl::_V1::ext::oneapi::experimental::uniform<.+>::.+",
    "^sycl::_V1::ext::oneapi::bfloat16::.+",
    "^sycl::_V1::ext::oneapi::experimental::if_architecture_is"};

static const char *LegalSYCLFunctionsInStatelessMode[] = {
    "^sycl::_V1::accessor<.+>::get_pointer.+",
    "^sycl::_V1::accessor<.+>::get_multi_ptr.+",
    "^sycl::_V1::accessor<.+>::getPointerAdjusted",
    "^sycl::_V1::accessor<.+>::getTotalOffset",
    "^sycl::_V1::accessor<.+>::getLinearIndex",
    "^sycl::_V1::accessor<.+>::getAccessRange",
    "^sycl::_V1::accessor<.+>::getMemoryRange",
    "^sycl::_V1::accessor<.+>::getOffset",
    "^sycl::_V1::accessor<.+>::operator\\[\\]"};
// clang-format on

namespace {

class ESIMDVerifierImpl {
  const Module &M;
  bool MayNeedForceStatelessMemModeAPI;

public:
  ESIMDVerifierImpl(const Module &M, bool MayNeedForceStatelessMemModeAPI)
      : M(M), MayNeedForceStatelessMemModeAPI(MayNeedForceStatelessMemModeAPI) {
  }

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

          // Skip local names, which are functions whose type
          // is not exposed outside of the current function,
          // such as lambdas or local classes. Note we will
          // still analyze functions called by these constructs,
          // assuming they are marked as ESIMD functions.
          if (NameNode->getKind() == id::Node::KLocalName)
            continue;

          id::OutputBuffer NameBuf;
          NameNode->print(NameBuf);
          StringRef Name(NameBuf.getBuffer(), NameBuf.getCurrentPosition());

          // We are interested in functions defined in SYCL namespace, but
          // outside of ESIMD namespaces.
          if (!Name.starts_with("sycl::_V1::") ||
              Name.starts_with("sycl::_V1::detail::") ||
              Name.starts_with("sycl::_V1::ext::intel::esimd::") ||
              Name.starts_with("sycl::_V1::ext::intel::experimental::esimd::"))
            continue;

          // Check if function name matches any allowed SYCL function name.
          auto checkLegalFunc = [Name](const char *LegalName) {
            Regex LegalNameRE(LegalName);
            assert(LegalNameRE.isValid() && "invalid function name regex");
            return LegalNameRE.match(Name);
          };
          if (any_of(LegalSYCLFunctions, checkLegalFunc) ||
              // Methods listed in LegalSYCLFunctionsInStatelessMode are
              // required to support ESIMD APIs accepting accessors in
              // stateless-only mode. Attempts to use that API with accessors
              // lowered to buffer_t will cause runtime error and thus must be
              // reported at compilation time.
              (MayNeedForceStatelessMemModeAPI &&
               any_of(LegalSYCLFunctionsInStatelessMode, checkLegalFunc)))
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
  ESIMDVerifierImpl(M, MayNeedForceStatelessMemModeAPI).verify();
  return PreservedAnalyses::all();
}

namespace {

struct ESIMDVerifier : public ModulePass {
  static char ID;
  bool MayNeedForceStatelessMemModeAPI;

  ESIMDVerifier(bool MayNeedForceStatelessMemModeAPI = true)
      : ModulePass(ID),
        MayNeedForceStatelessMemModeAPI(MayNeedForceStatelessMemModeAPI) {
    initializeESIMDVerifierPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override {
    ESIMDVerifierImpl(M, MayNeedForceStatelessMemModeAPI).verify();
    return false;
  }
};

} // end anonymous namespace

char ESIMDVerifier::ID = 0;

INITIALIZE_PASS_BEGIN(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                      false, false)
INITIALIZE_PASS_END(ESIMDVerifier, DEBUG_TYPE, "ESIMD-specific IR verifier",
                    false, false)

ModulePass *
llvm::createESIMDVerifierPass(bool MayNeedForceStatelessMemModeAPI) {
  return new ESIMDVerifier(MayNeedForceStatelessMemModeAPI);
}
