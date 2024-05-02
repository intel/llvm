// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/metadata.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace compiler {
namespace utils {

uint32_t getOpenCLVersion(const llvm::Module &m) {
  if (auto *const md = m.getNamedMetadata("opencl.ocl.version")) {
    if (md->getNumOperands() == 1) {
      auto *const op = md->getOperand(0);
      if (op->getNumOperands() == 2) {
        const auto major =
            mdconst::extract<ConstantInt>(op->getOperand(0))->getZExtValue();
        const auto minor =
            mdconst::extract<ConstantInt>(op->getOperand(1))->getZExtValue();
        return (major * 100 + minor) * 1000;
      }
    }
  }
  return OpenCLC12;
}

static constexpr const char *ReqdWGSizeMD = "reqd_work_group_size";

static MDTuple *encodeVectorizationInfo(const VectorizationInfo &info,
                                        LLVMContext &Ctx) {
  auto *const i32Ty = Type::getInt32Ty(Ctx);

  return MDTuple::get(
      Ctx,
      {ConstantAsMetadata::get(ConstantInt::get(i32Ty, info.vf.getKnownMin())),
       ConstantAsMetadata::get(ConstantInt::get(i32Ty, info.vf.isScalable())),
       ConstantAsMetadata::get(ConstantInt::get(i32Ty, info.simdDimIdx)),
       ConstantAsMetadata::get(
           ConstantInt::get(i32Ty, info.IsVectorPredicated))});
}

static std::optional<VectorizationInfo> extractVectorizationInfo(MDTuple *md) {
  if (md->getNumOperands() != 4) {
    return std::nullopt;
  }
  auto *const widthMD = mdconst::extract<ConstantInt>(md->getOperand(0));
  auto *const isScalableMD = mdconst::extract<ConstantInt>(md->getOperand(1));
  auto *const simdDimIdxMD = mdconst::extract<ConstantInt>(md->getOperand(2));
  auto *const isVPMD = mdconst::extract<ConstantInt>(md->getOperand(3));

  VectorizationInfo info;

  info.vf.setKnownMin(widthMD->getZExtValue());
  info.vf.setIsScalable(isScalableMD->equalsInt(1));
  info.simdDimIdx = simdDimIdxMD->getZExtValue();
  info.IsVectorPredicated = isVPMD->equalsInt(1);

  return info;
}

static std::optional<LinkMetadataResult> parseVectorLinkMD(MDNode *mdnode) {
  if (auto info =
          extractVectorizationInfo(dyn_cast<MDTuple>(mdnode->getOperand(0)))) {
    // The Function may well be null.
    Function *vecFn = mdconst::extract_or_null<Function>(mdnode->getOperand(1));
    return LinkMetadataResult(vecFn, *info);
  }
  return std::nullopt;
}

void encodeVectorizationFailedMetadata(Function &f,
                                       const VectorizationInfo &info) {
  auto *veczInfo = encodeVectorizationInfo(info, f.getContext());
  f.addMetadata("codeplay_ca_vecz.base.fail", *veczInfo);
}

void linkOrigToVeczFnMetadata(Function &origF, Function &vectorF,
                              const VectorizationInfo &info) {
  auto *veczInfo = encodeVectorizationInfo(info, origF.getContext());
  auto *const mdTuple = MDTuple::get(
      origF.getContext(), {veczInfo, ValueAsMetadata::get(&vectorF)});
  origF.addMetadata("codeplay_ca_vecz.base", *mdTuple);
}

void linkVeczToOrigFnMetadata(Function &vectorizedF, Function &origF,
                              const VectorizationInfo &info) {
  auto *veczInfo = encodeVectorizationInfo(info, vectorizedF.getContext());
  auto *const mdTuple = MDTuple::get(origF.getContext(),
                                     {veczInfo, ValueAsMetadata::get(&origF)});
  vectorizedF.addMetadata("codeplay_ca_vecz.derived", *mdTuple);
}

static bool parseVectorizedFunctionLinkMetadata(
    Function &f, StringRef mdName,
    SmallVectorImpl<LinkMetadataResult> &results) {
  SmallVector<MDNode *, 1> nodes;

  f.getMetadata(mdName, nodes);
  if (nodes.empty()) {
    return false;
  }
  results.reserve(results.size() + nodes.size());
  for (auto *mdnode : nodes) {
    if (auto link = parseVectorLinkMD(mdnode)) {
      results.emplace_back(*link);
    } else {
      return false;
    }
  }
  return true;
}

bool parseOrigToVeczFnLinkMetadata(Function &f,
                                   SmallVectorImpl<LinkMetadataResult> &VFs) {
  return parseVectorizedFunctionLinkMetadata(f, "codeplay_ca_vecz.base", VFs);
}

std::optional<LinkMetadataResult> parseVeczToOrigFnLinkMetadata(Function &f) {
  auto *mdnode = f.getMetadata("codeplay_ca_vecz.derived");
  if (!mdnode) {
    return std::nullopt;
  }
  return parseVectorLinkMD(mdnode);
}

void dropVeczOrigMetadata(Function &f) {
  f.setMetadata("codeplay_ca_vecz.base", nullptr);
}

void dropVeczDerivedMetadata(Function &f) {
  f.setMetadata("codeplay_ca_vecz.derived", nullptr);
}

void encodeWrapperFnMetadata(Function &f, const VectorizationInfo &mainInfo,
                             std::optional<VectorizationInfo> tailInfo) {
  MDTuple *tailInfoMD = nullptr;
  auto *mainInfoMD = encodeVectorizationInfo(mainInfo, f.getContext());

  if (tailInfo) {
    tailInfoMD = encodeVectorizationInfo(*tailInfo, f.getContext());
  }

  f.setMetadata("codeplay_ca_wrapper",
                MDTuple::get(f.getContext(), {mainInfoMD, tailInfoMD}));
}

std::optional<std::pair<VectorizationInfo, std::optional<VectorizationInfo>>>
parseWrapperFnMetadata(Function &f) {
  auto *const mdnode = f.getMetadata("codeplay_ca_wrapper");
  if (!mdnode || mdnode->getNumOperands() != 2) {
    return std::nullopt;
  }

  auto *const mainTuple = dyn_cast_or_null<MDTuple>(mdnode->getOperand(0));
  if (!mainTuple) {
    return std::nullopt;
  }

  VectorizationInfo mainInfo;
  std::optional<VectorizationInfo> tailInfo;

  if (auto info = extractVectorizationInfo(mainTuple)) {
    mainInfo = *info;
  } else {
    return std::nullopt;
  }

  if (auto *const tailTuple =
          dyn_cast_or_null<MDTuple>(mdnode->getOperand(1))) {
    if (auto info = extractVectorizationInfo(tailTuple)) {
      tailInfo = info;
    }
  }

  return std::make_pair(mainInfo, tailInfo);
}

void copyFunctionMetadata(Function &fromF, Function &toF, bool includeDebug) {
  if (includeDebug) {
    toF.copyMetadata(&fromF, 0);
    return;
  }
  // Copy the metadata into the new kernel ignoring any debug info.
  SmallVector<std::pair<unsigned, MDNode *>, 5> metadata;
  fromF.getAllMetadata(metadata);

  // Iterate through the metadata and only add nodes to the new one if they
  // are not debug info.
  for (const auto &pair : metadata) {
    if (auto *nonDebug = dyn_cast_or_null<MDTuple>(pair.second)) {
      toF.setMetadata(pair.first, nonDebug);
    }
  }
}

void encodeLocalSizeMetadata(Function &f, const std::array<uint64_t, 3> &size) {
  // We may be truncating i64 to i32 but we don't expect local sizes to ever
  // exceed 32 bits.
  auto *const i32Ty = Type::getInt32Ty(f.getContext());
  auto *const mdTuple =
      MDTuple::get(f.getContext(),
                   {ConstantAsMetadata::get(ConstantInt::get(i32Ty, size[0])),
                    ConstantAsMetadata::get(ConstantInt::get(i32Ty, size[1])),
                    ConstantAsMetadata::get(ConstantInt::get(i32Ty, size[2]))});
  f.setMetadata(ReqdWGSizeMD, mdTuple);
}

std::optional<std::array<uint64_t, 3>> getLocalSizeMetadata(const Function &f) {
  if (auto *md = f.getMetadata(ReqdWGSizeMD)) {
    return std::array<uint64_t, 3>{
        mdconst::extract<ConstantInt>(md->getOperand(0))->getZExtValue(),
        mdconst::extract<ConstantInt>(md->getOperand(1))->getZExtValue(),
        mdconst::extract<ConstantInt>(md->getOperand(2))->getZExtValue()};
  }
  return std::nullopt;
}

static constexpr const char *MuxScheduledFnMD = "mux_scheduled_fn";

void dropSchedulingParameterMetadata(Function &f) {
  f.setMetadata(MuxScheduledFnMD, nullptr);
}

SmallVector<int, 4> getSchedulingParameterFunctionMetadata(const Function &f) {
  SmallVector<int, 4> idxs;
  if (auto *md = f.getMetadata(MuxScheduledFnMD)) {
    for (auto &op : md->operands()) {
      idxs.push_back(mdconst::extract<ConstantInt>(op)->getSExtValue());
    }
  }
  return idxs;
}

void setSchedulingParameterFunctionMetadata(Function &f, ArrayRef<int> idxs) {
  if (idxs.empty()) {
    return;
  }
  SmallVector<Metadata *, 4> mdOps;
  auto *const i32Ty = Type::getInt32Ty(f.getContext());
  for (auto idx : idxs) {
    mdOps.push_back(ConstantAsMetadata::get(ConstantInt::get(i32Ty, idx)));
  }
  auto *const mdOpsTuple = MDTuple::get(f.getContext(), mdOps);
  f.setMetadata(MuxScheduledFnMD, mdOpsTuple);
}

static constexpr const char *MuxSchedulingParamsMD = "mux-scheduling-params";

void setSchedulingParameterModuleMetadata(Module &m,
                                          ArrayRef<std::string> names) {
  SmallVector<Metadata *, 4> paramDebugNames;
  for (const auto &name : names) {
    paramDebugNames.push_back(MDString::get(m.getContext(), name));
  }
  auto *const md = m.getOrInsertNamedMetadata(MuxSchedulingParamsMD);
  md->clearOperands();
  md->addOperand(MDNode::get(m.getContext(), paramDebugNames));
}

NamedMDNode *getSchedulingParameterModuleMetadata(const Module &m) {
  return m.getNamedMetadata(MuxSchedulingParamsMD);
}

std::optional<unsigned> isSchedulingParameter(const Function &f, unsigned idx) {
  if (auto *md = f.getMetadata(MuxScheduledFnMD)) {
    for (const auto &op : enumerate(md->operands())) {
      auto paramIdx = mdconst::extract<ConstantInt>(op.value())->getSExtValue();
      if (paramIdx >= 0 && (unsigned)paramIdx == idx) {
        return op.index();
      }
    }
  }
  return std::nullopt;
}

// Uses the format of a metadata node directly applied to a function.
std::optional<std::array<uint64_t, 3>> parseRequiredWGSMetadata(
    const Function &f) {
  if (auto mdnode = f.getMetadata(ReqdWGSizeMD)) {
    std::array<uint64_t, 3> wgs = {0, 1, 1};
    assert(mdnode->getNumOperands() >= 1 && mdnode->getNumOperands() <= 3 &&
           "Unsupported number of operands in reqd_work_group_size");
    for (const auto &[idx, op] : enumerate(mdnode->operands())) {
      wgs[idx] = mdconst::extract<ConstantInt>(op)->getZExtValue();
    }
    return wgs;
  }
  return std::nullopt;
}

// Uses the format of a metadata node that's a part of the opencl.kernels node.
std::optional<std::array<uint64_t, 3>> parseRequiredWGSMetadata(
    const MDNode &node) {
  for (uint32_t i = 1; i < node.getNumOperands(); ++i) {
    MDNode *const subNode = cast<MDNode>(node.getOperand(i));
    MDString *const operandName = cast<MDString>(subNode->getOperand(0));
    if (operandName->getString() == ReqdWGSizeMD) {
      auto *const op0 = mdconst::extract<ConstantInt>(subNode->getOperand(1));
      auto *const op1 = mdconst::extract<ConstantInt>(subNode->getOperand(2));
      auto *const op2 = mdconst::extract<ConstantInt>(subNode->getOperand(3));
      // KLOCWORK "UNINIT.STACK.ARRAY.MUST" possible false positive
      // Initialization of looks like an uninitialized access to Klocwork
      std::array<uint64_t, 3> wgs = {
          {op0->getZExtValue(), op1->getZExtValue(), op2->getZExtValue()}};
      return wgs;
    }
  }
  return std::nullopt;
}

std::optional<uint32_t> parseMaxWorkDimMetadata(const Function &f) {
  if (auto *mdnode = f.getMetadata("max_work_dim")) {
    auto *op0 = mdconst::extract<ConstantInt>(mdnode->getOperand(0));
    return op0->getZExtValue();
  }

  return std::nullopt;
}

void populateKernelList(Module &m, SmallVectorImpl<KernelInfo> &results) {
  // Construct list of kernels from metadata, if present.
  if (auto *md = m.getNamedMetadata("opencl.kernels")) {
    for (uint32_t i = 0, e = md->getNumOperands(); i < e; ++i) {
      MDNode *const kernelNode = md->getOperand(i);
      ValueAsMetadata *vmdKernel =
          cast<ValueAsMetadata>(kernelNode->getOperand(0));
      KernelInfo info{vmdKernel->getValue()->getName()};
      if (auto wgs = parseRequiredWGSMetadata(*kernelNode)) {
        info.ReqdWGSize = *wgs;
      }
      results.push_back(info);
    }
    return;
  }

  // No metadata - assume all functions with the SPIR_KERNEL calling
  // convention are kernels.
  for (auto &f : m) {
    if (f.hasName() && f.getCallingConv() == CallingConv::SPIR_KERNEL) {
      KernelInfo info(f.getName());
      if (auto wgs = parseRequiredWGSMetadata(f)) {
        info.ReqdWGSize = *wgs;
      }
      results.push_back(info);
    }
  }
}

void replaceKernelInOpenCLKernelsMetadata(Function &fromF, Function &toF,
                                          Module &M) {
  // update the kernel metadata
  if (auto *const namedMD = M.getNamedMetadata("opencl.kernels")) {
    for (auto *md : namedMD->operands()) {
      if (md && md->getOperand(0) == ValueAsMetadata::get(&fromF)) {
        md->replaceOperandWith(0, ValueAsMetadata::get(&toF));
      }
    }
  }
}

static constexpr const char *ReqdSGSizeMD = "intel_reqd_sub_group_size";

void encodeReqdSubgroupSizeMetadata(Function &f, uint32_t size) {
  auto *const i32Ty = Type::getInt32Ty(f.getContext());
  auto *const mdTuple = MDTuple::get(
      f.getContext(), ConstantAsMetadata::get(ConstantInt::get(i32Ty, size)));
  f.setMetadata(ReqdSGSizeMD, mdTuple);
}

std::optional<uint32_t> getReqdSubgroupSize(const Function &f) {
  if (auto *md = f.getMetadata(ReqdSGSizeMD)) {
    return mdconst::extract<ConstantInt>(md->getOperand(0))->getZExtValue();
  }
  return std::nullopt;
}

}  // namespace utils
}  // namespace compiler
