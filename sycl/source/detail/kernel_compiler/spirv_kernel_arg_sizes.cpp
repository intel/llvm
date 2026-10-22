//==-- spirv_kernel_arg_sizes.cpp  Kernel argument sizes from SPIR-V ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_compiler/spirv_kernel_arg_sizes.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace {

// The subset of the SPIR-V specification needed to determine the layout of
// kernel parameters.
namespace spv {
constexpr uint32_t MagicNumber = 0x07230203;
constexpr uint32_t MagicNumberSwapped = 0x03022307;
constexpr size_t HeaderWords = 5;

enum Op : uint32_t {
  OpMemoryModel = 14,
  OpEntryPoint = 15,
  OpTypeInt = 21,
  OpTypeFloat = 22,
  OpTypeVector = 23,
  OpTypeArray = 28,
  OpTypeStruct = 30,
  OpTypePointer = 32,
  OpConstant = 43,
  OpFunction = 54,
  OpFunctionParameter = 55,
  OpFunctionEnd = 56,
  OpDecorate = 71,
  OpMemberDecorate = 72,
};

constexpr uint32_t AddressingModelPhysical32 = 1;
constexpr uint32_t AddressingModelPhysical64 = 2;
constexpr uint32_t ExecutionModelKernel = 6;
constexpr uint32_t StorageClassWorkgroup = 4;
constexpr uint32_t DecorationCPacked = 10;
constexpr uint32_t DecorationFuncParamAttr = 38;
constexpr uint32_t DecorationOffset = 35;
constexpr uint32_t FunctionParameterAttributeByVal = 2;
} // namespace spv

struct Layout {
  size_t Size;
  size_t Align;
};

struct TypeInfo {
  uint32_t Opcode = 0;
  // Operands following the result id.
  std::vector<uint32_t> Operands;
};

class SPIRVModuleInfo {
public:
  bool parse(const std::byte *Data, size_t Size);
  KernelNameToArgSizesMap getKernelArgSizes() const;

private:
  std::optional<Layout> getLayout(uint32_t TypeId) const;
  std::optional<size_t> getParamSize(uint32_t ParamId,
                                     uint32_t ParamTypeId) const;
  std::optional<size_t> getPointerSize() const;

  uint32_t MAddressingModel = 0;
  std::vector<std::pair<std::string, uint32_t>> MKernelEntryPoints;
  std::unordered_map<uint32_t, TypeInfo> MTypes;
  std::unordered_map<uint32_t, uint64_t> MConstants;
  // Function id -> (parameter type id, parameter id) in parameter order.
  std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>
      MFunctionParams;
  std::set<uint32_t> MByValParams;
  std::set<uint32_t> MPackedStructs;
  // Struct id -> member index -> explicit offset.
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>
      MMemberOffsets;
};

bool SPIRVModuleInfo::parse(const std::byte *Data, size_t Size) {
  if (!Data || Size % sizeof(uint32_t) != 0 ||
      Size < spv::HeaderWords * sizeof(uint32_t))
    return false;

  std::vector<uint32_t> Words(Size / sizeof(uint32_t));
  std::memcpy(Words.data(), Data, Size);
  if (Words[0] == spv::MagicNumberSwapped) {
    for (uint32_t &W : Words)
      W = ((W & 0xFFu) << 24) | ((W & 0xFF00u) << 8) | ((W >> 8) & 0xFF00u) |
          (W >> 24);
  } else if (Words[0] != spv::MagicNumber) {
    return false;
  }

  uint32_t CurrentFunction = 0;
  for (size_t I = spv::HeaderWords; I < Words.size();) {
    const uint32_t WordCount = Words[I] >> 16;
    const uint32_t Opcode = Words[I] & 0xFFFFu;
    if (WordCount == 0 || I + WordCount > Words.size())
      return false;
    const uint32_t *Ops = Words.data() + I + 1;
    const uint32_t NumOps = WordCount - 1;
    I += WordCount;

    switch (Opcode) {
    case spv::OpMemoryModel:
      if (NumOps >= 1)
        MAddressingModel = Ops[0];
      break;
    case spv::OpEntryPoint: {
      if (NumOps < 3 || Ops[0] != spv::ExecutionModelKernel)
        break;
      // The name is a nul-terminated string packed into the following words.
      const char *NameBegin = reinterpret_cast<const char *>(&Ops[2]);
      const size_t MaxLen = (NumOps - 2) * sizeof(uint32_t);
      const size_t Len = strnlen(NameBegin, MaxLen);
      if (Len == MaxLen)
        return false;
      MKernelEntryPoints.emplace_back(std::string(NameBegin, Len), Ops[1]);
      break;
    }
    case spv::OpDecorate:
      if (NumOps >= 3 && Ops[1] == spv::DecorationFuncParamAttr &&
          Ops[2] == spv::FunctionParameterAttributeByVal)
        MByValParams.insert(Ops[0]);
      else if (NumOps >= 2 && Ops[1] == spv::DecorationCPacked)
        MPackedStructs.insert(Ops[0]);
      break;
    case spv::OpMemberDecorate:
      if (NumOps >= 4 && Ops[2] == spv::DecorationOffset)
        MMemberOffsets[Ops[0]][Ops[1]] = Ops[3];
      break;
    case spv::OpTypeInt:
    case spv::OpTypeFloat:
    case spv::OpTypeVector:
    case spv::OpTypeArray:
    case spv::OpTypeStruct:
    case spv::OpTypePointer:
      // Other types (images, samplers, untyped pointers, ...) are not
      // recorded, so parameters of those types get an unknown size.
      if (NumOps >= 1)
        MTypes[Ops[0]] = TypeInfo{Opcode, {Ops + 1, Ops + NumOps}};
      break;
    case spv::OpConstant:
      if (NumOps == 3)
        MConstants[Ops[1]] = Ops[2];
      else if (NumOps == 4)
        MConstants[Ops[1]] = Ops[2] | (static_cast<uint64_t>(Ops[3]) << 32);
      break;
    case spv::OpFunction:
      if (NumOps >= 2) {
        CurrentFunction = Ops[1];
        MFunctionParams[CurrentFunction];
      }
      break;
    case spv::OpFunctionParameter:
      if (CurrentFunction == 0 || NumOps < 2)
        return false;
      MFunctionParams[CurrentFunction].emplace_back(Ops[0], Ops[1]);
      break;
    case spv::OpFunctionEnd:
      CurrentFunction = 0;
      break;
    default:
      break;
    }
  }
  return true;
}

std::optional<size_t> SPIRVModuleInfo::getPointerSize() const {
  if (MAddressingModel == spv::AddressingModelPhysical64)
    return 8;
  if (MAddressingModel == spv::AddressingModelPhysical32)
    return 4;
  return std::nullopt;
}

static size_t alignTo(size_t Value, size_t Align) {
  return (Value + Align - 1) / Align * Align;
}

// Computes the layout of a type as defined by the OpenCL C ABI, which is the
// layout the host has to use for by-value kernel arguments.
std::optional<Layout> SPIRVModuleInfo::getLayout(uint32_t TypeId) const {
  auto It = MTypes.find(TypeId);
  if (It == MTypes.end())
    return std::nullopt;
  const TypeInfo &Type = It->second;
  const std::vector<uint32_t> &Ops = Type.Operands;

  switch (Type.Opcode) {
  case spv::OpTypeInt:
  case spv::OpTypeFloat: {
    if (Ops.empty() || Ops[0] == 0 || Ops[0] % 8 != 0)
      return std::nullopt;
    const size_t Bytes = Ops[0] / 8;
    return Layout{Bytes, Bytes};
  }
  case spv::OpTypeVector: {
    if (Ops.size() < 2)
      return std::nullopt;
    std::optional<Layout> Elem = getLayout(Ops[0]);
    if (!Elem)
      return std::nullopt;
    // 3-component vectors have the size and alignment of 4-component ones.
    const size_t NumElems = Ops[1] == 3 ? 4 : Ops[1];
    const size_t Bytes = Elem->Size * NumElems;
    return Layout{Bytes, Bytes};
  }
  case spv::OpTypeArray: {
    if (Ops.size() < 2)
      return std::nullopt;
    std::optional<Layout> Elem = getLayout(Ops[0]);
    auto LengthIt = MConstants.find(Ops[1]);
    if (!Elem || LengthIt == MConstants.end())
      return std::nullopt;
    return Layout{Elem->Size * LengthIt->second, Elem->Align};
  }
  case spv::OpTypeStruct: {
    const bool IsPacked = MPackedStructs.count(TypeId) != 0;
    auto OffsetsIt = MMemberOffsets.find(TypeId);
    const std::unordered_map<uint32_t, uint32_t> *ExplicitOffsets =
        OffsetsIt != MMemberOffsets.end() ? &OffsetsIt->second : nullptr;
    if (ExplicitOffsets && ExplicitOffsets->size() != Ops.size())
      return std::nullopt;

    size_t Size = 0;
    size_t Align = 1;
    for (uint32_t MemberIdx = 0; MemberIdx < Ops.size(); ++MemberIdx) {
      std::optional<Layout> Member = getLayout(Ops[MemberIdx]);
      if (!Member)
        return std::nullopt;
      const size_t MemberAlign = IsPacked ? 1 : Member->Align;
      size_t Offset = alignTo(Size, MemberAlign);
      if (ExplicitOffsets)
        Offset = ExplicitOffsets->at(MemberIdx);
      Size = std::max(Size, Offset + Member->Size);
      Align = std::max(Align, MemberAlign);
    }
    return Layout{alignTo(Size, Align), Align};
  }
  case spv::OpTypePointer: {
    std::optional<size_t> PtrSize = getPointerSize();
    if (!PtrSize)
      return std::nullopt;
    return Layout{*PtrSize, *PtrSize};
  }
  default:
    return std::nullopt;
  }
}

std::optional<size_t>
SPIRVModuleInfo::getParamSize(uint32_t ParamId, uint32_t ParamTypeId) const {
  auto It = MTypes.find(ParamTypeId);
  if (It == MTypes.end())
    return std::nullopt;
  const TypeInfo &Type = It->second;
  if (Type.Opcode != spv::OpTypePointer) {
    std::optional<Layout> L = getLayout(ParamTypeId);
    return L ? std::optional<size_t>{L->Size} : std::nullopt;
  }

  if (Type.Operands.size() < 2)
    return std::nullopt;
  // Aggregates passed by value are represented as pointers to the aggregate.
  if (MByValParams.count(ParamId)) {
    std::optional<Layout> L = getLayout(Type.Operands[1]);
    return L ? std::optional<size_t>{L->Size} : std::nullopt;
  }
  // For local memory the argument size is the size of the allocation.
  if (Type.Operands[0] == spv::StorageClassWorkgroup)
    return std::nullopt;
  return getPointerSize();
}

KernelNameToArgSizesMap SPIRVModuleInfo::getKernelArgSizes() const {
  KernelNameToArgSizesMap Result;
  for (const auto &[Name, FunctionId] : MKernelEntryPoints) {
    auto ParamsIt = MFunctionParams.find(FunctionId);
    if (ParamsIt == MFunctionParams.end())
      continue;
    KernelArgSizesT Sizes;
    Sizes.reserve(ParamsIt->second.size());
    for (const auto &[ParamTypeId, ParamId] : ParamsIt->second)
      Sizes.push_back(getParamSize(ParamId, ParamTypeId));
    Result.emplace(Name, std::move(Sizes));
  }
  return Result;
}

} // namespace

KernelNameToArgSizesMap getSPIRVKernelArgSizes(const std::byte *Data,
                                               size_t Size) {
  SPIRVModuleInfo Module;
  if (!Module.parse(Data, Size))
    return {};
  return Module.getKernelArgSizes();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
