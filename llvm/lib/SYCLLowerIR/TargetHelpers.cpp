//===----------- TargetHelpers.cpp - Helpers for SYCL kernels ------------ ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for processing SYCL kernels.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;

namespace llvm {
namespace TargetHelpers {

KernelPayload::KernelPayload(Function *Kernel, MDNode *MD)
    : Kernel(Kernel), MD(MD) {}

ArchType getArchType(const Module &M) {
  return StringSwitch<ArchType>(M.getTargetTriple().c_str())
      .Case("nvptx64-nvidia-cuda", ArchType::Cuda)
      .Case("nvptx-nvidia-cuda", ArchType::Cuda)
      .Case("amdgcn-amd-amdhsa", ArchType::AMDHSA)
      .Case("amdgcn--amdhsa", ArchType::AMDHSA)
      .Default(ArchType::Unsupported);
}

std::string getAnnotationString(ArchType AT) {
  switch (AT) {
  case TargetHelpers::ArchType::Cuda:
    return std::string("nvvm.annotations");
    break;
  case TargetHelpers::ArchType::AMDHSA:
    return std::string("amdgcn.annotations");
    break;
  default:
    llvm_unreachable("Unsupported arch type.");
  }
  return std::string();
}

void populateKernels(Module &M, SmallVectorImpl<KernelPayload> &Kernels,
                     ArchType AT) {
  // Access `{amdgcn|nvvm}.annotations` to determine which functions are kernel
  // entry points.
  std::string Annotation = getAnnotationString(AT);
  auto *AnnotationMetadata = M.getNamedMetadata(Annotation);
  // No kernels in the module, early exit.
  if (!AnnotationMetadata)
    return;

  // It is possible that the annotations node contains multiple pointers to the
  // same metadata, recognise visited ones.
  SmallSet<MDNode *, 4> Visited;
  for (auto *MetadataNode : AnnotationMetadata->operands()) {
    if (Visited.contains(MetadataNode) || MetadataNode->getNumOperands() != 3)
      continue;

    Visited.insert(MetadataNode);

    // Kernel entry points are identified using metadata nodes of the form:
    //   !X = !{<function>, !"kernel", i32 1}
    auto *Type = dyn_cast<MDString>(MetadataNode->getOperand(1));
    if (!Type)
      continue;
    // Only process kernel entry points.
    if (Type->getString() != "kernel")
      continue;

    // Get a pointer to the entry point function from the metadata.
    const MDOperand &FuncOperand = MetadataNode->getOperand(0);
    if (!FuncOperand)
      continue;
    if (auto *FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand))
      if (auto *Func = dyn_cast<Function>(FuncConstant->getValue()))
        Kernels.push_back(KernelPayload(Func, MetadataNode));
  }
}

} // namespace TargetHelpers
} // namespace llvm
