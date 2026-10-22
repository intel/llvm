//==--- SPIRVKernelArgSizes.cpp --- SPIR-V kernel argument size checks ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Kernels built from SPIR-V carry no compile-time parameter information, so the
// runtime reads the parameter sizes from the module and rejects set_arg values
// of a different size instead of letting the backend read garbage.

#include <detail/kernel_compiler/spirv_kernel_arg_sizes.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/experimental/raw_kernel_arg.hpp>
#include <sycl/kernel_bundle.hpp>

#include <cstring>

namespace syclex = sycl::ext::oneapi::experimental;
using sycl::detail::getSPIRVKernelArgSizes;
using sycl::detail::KernelArgSizesT;
using sycl::detail::KernelNameToArgSizesMap;

namespace {

namespace op {
constexpr uint32_t MemoryModel = 14;
constexpr uint32_t EntryPoint = 15;
constexpr uint32_t TypeVoid = 19;
constexpr uint32_t TypeInt = 21;
constexpr uint32_t TypeFloat = 22;
constexpr uint32_t TypeVector = 23;
constexpr uint32_t TypeSampler = 26;
constexpr uint32_t TypeArray = 28;
constexpr uint32_t TypeStruct = 30;
constexpr uint32_t TypePointer = 32;
constexpr uint32_t TypeFunction = 33;
constexpr uint32_t Constant = 43;
constexpr uint32_t Function = 54;
constexpr uint32_t FunctionParameter = 55;
constexpr uint32_t FunctionEnd = 56;
constexpr uint32_t Decorate = 71;
constexpr uint32_t MemberDecorate = 72;
} // namespace op

constexpr uint32_t AddressingPhysical32 = 1;
constexpr uint32_t AddressingPhysical64 = 2;
constexpr uint32_t ExecModelGLCompute = 5;
constexpr uint32_t ExecModelKernel = 6;
constexpr uint32_t StorageWorkgroup = 4;
constexpr uint32_t StorageCrossWorkgroup = 5;
constexpr uint32_t StorageFunction = 7;
constexpr uint32_t DecorationCPacked = 10;
constexpr uint32_t DecorationOffset = 35;
constexpr uint32_t DecorationFuncParamAttr = 38;
constexpr uint32_t ParamAttrByVal = 2;

class SPIRVModuleBuilder {
public:
  SPIRVModuleBuilder() : MWords{0x07230203, 0x00010000, 0, 1000, 0} {}

  SPIRVModuleBuilder &add(uint32_t Opcode, std::vector<uint32_t> Operands) {
    MWords.push_back(static_cast<uint32_t>(Operands.size() + 1) << 16 | Opcode);
    MWords.insert(MWords.end(), Operands.begin(), Operands.end());
    return *this;
  }

  SPIRVModuleBuilder &addEntryPoint(uint32_t ExecModel, uint32_t FunctionId,
                                    const std::string &Name) {
    std::vector<uint32_t> Operands{ExecModel, FunctionId};
    std::vector<uint32_t> NameWords(Name.size() / sizeof(uint32_t) + 1, 0);
    std::memcpy(NameWords.data(), Name.data(), Name.size());
    Operands.insert(Operands.end(), NameWords.begin(), NameWords.end());
    return add(op::EntryPoint, Operands);
  }

  // Adds `void Name(Params...)`, where each element of Params is a pair of
  // parameter type id and parameter id.
  SPIRVModuleBuilder &
  addFunction(uint32_t VoidId, uint32_t FunctionId, uint32_t FunctionTypeId,
              const std::vector<std::pair<uint32_t, uint32_t>> &Params) {
    add(op::Function, {VoidId, FunctionId, 0, FunctionTypeId});
    for (auto [TypeId, ParamId] : Params)
      add(op::FunctionParameter, {TypeId, ParamId});
    return add(op::FunctionEnd, {});
  }

  const std::vector<uint32_t> &words() const { return MWords; }

  std::vector<std::byte> bytes() const {
    std::vector<std::byte> Bytes(MWords.size() * sizeof(uint32_t));
    std::memcpy(Bytes.data(), MWords.data(), Bytes.size());
    return Bytes;
  }

private:
  std::vector<uint32_t> MWords;
};

KernelNameToArgSizesMap parse(const std::vector<std::byte> &Bytes) {
  return getSPIRVKernelArgSizes(Bytes.data(), Bytes.size());
}

// Type and value ids used by the modules below.
enum Id : uint32_t {
  Void = 1,
  I8,
  I32,
  I64,
  F16,
  F32,
  V3F32,
  ConstOne,
  ArrI64x1,
  IdInner,
  IdOuter,
  PaddedStruct,
  PackedStruct,
  OffsetStruct,
  GlobalPtr,
  LocalPtr,
  ByValIdPtr,
  ByValPaddedPtr,
  ByValPackedPtr,
  ByValOffsetPtr,
  Sampler,
  FnType,
  Kernel,
  Param0,
};

// Declares the types shared by the modules below.
void addTypes(SPIRVModuleBuilder &B) {
  B.add(op::TypeVoid, {Void})
      .add(op::TypeInt, {I8, 8, 0})
      .add(op::TypeInt, {I32, 32, 0})
      .add(op::TypeInt, {I64, 64, 0})
      .add(op::TypeFloat, {F16, 16})
      .add(op::TypeFloat, {F32, 32})
      .add(op::TypeVector, {V3F32, F32, 3})
      .add(op::Constant, {I32, ConstOne, 1})
      .add(op::TypeArray, {ArrI64x1, I64, ConstOne})
      // The shape of sycl::id<1> after lowering: struct { struct { i64[1] } }.
      .add(op::TypeStruct, {IdInner, ArrI64x1})
      .add(op::TypeStruct, {IdOuter, IdInner})
      .add(op::TypeStruct, {PaddedStruct, I8, I32})
      .add(op::TypeStruct, {PackedStruct, I8, I32})
      .add(op::TypeStruct, {OffsetStruct, I32, I32})
      .add(op::TypePointer, {GlobalPtr, StorageCrossWorkgroup, I32})
      .add(op::TypePointer, {LocalPtr, StorageWorkgroup, I32})
      .add(op::TypePointer, {ByValIdPtr, StorageFunction, IdOuter})
      .add(op::TypePointer, {ByValPaddedPtr, StorageFunction, PaddedStruct})
      .add(op::TypePointer, {ByValPackedPtr, StorageFunction, PackedStruct})
      .add(op::TypePointer, {ByValOffsetPtr, StorageFunction, OffsetStruct})
      .add(op::TypeSampler, {Sampler})
      .add(op::TypeFunction, {FnType, Void});
}

// Builds a module with a single kernel `Name` taking the given parameter
// types. ByVal parameters are decorated as aggregates passed by value.
std::vector<std::byte>
makeKernelModule(const std::string &Name,
                 const std::vector<uint32_t> &ParamTypes,
                 const std::vector<uint32_t> &ByValParams = {},
                 uint32_t AddressingModel = AddressingPhysical64,
                 uint32_t ExecModel = ExecModelKernel) {
  SPIRVModuleBuilder B;
  B.add(op::MemoryModel, {AddressingModel, 2});
  B.addEntryPoint(ExecModel, Kernel, Name);
  for (uint32_t ParamIdx : ByValParams)
    B.add(op::Decorate,
          {Param0 + ParamIdx, DecorationFuncParamAttr, ParamAttrByVal});
  B.add(op::Decorate, {PackedStruct, DecorationCPacked});
  B.add(op::MemberDecorate, {OffsetStruct, 0, DecorationOffset, 0});
  B.add(op::MemberDecorate, {OffsetStruct, 1, DecorationOffset, 12});
  addTypes(B);

  std::vector<std::pair<uint32_t, uint32_t>> Params;
  for (uint32_t I = 0; I < ParamTypes.size(); ++I)
    Params.emplace_back(ParamTypes[I], Param0 + I);
  B.addFunction(Void, Kernel, FnType, Params);
  return B.bytes();
}

} // namespace

TEST(SPIRVKernelArgSizes, ScalarAndVectorParams) {
  auto Sizes = parse(makeKernelModule("k", {I8, I32, I64, F16, F32, V3F32}));
  ASSERT_EQ(Sizes.count("k"), 1u);
  EXPECT_EQ(Sizes["k"], (KernelArgSizesT{1, 4, 8, 2, 4, 16}));
}

TEST(SPIRVKernelArgSizes, PointerParams) {
  auto Sizes = parse(makeKernelModule("k", {GlobalPtr, LocalPtr}));
  // The size of a local memory argument is the allocation size, so it cannot
  // be checked against the parameter type.
  EXPECT_EQ(Sizes["k"], (KernelArgSizesT{8, std::nullopt}));

  auto Sizes32 = parse(makeKernelModule("k", {GlobalPtr}, /*ByValParams=*/{},
                                        AddressingPhysical32));
  EXPECT_EQ(Sizes32["k"], (KernelArgSizesT{4}));
}

TEST(SPIRVKernelArgSizes, ByValAggregateParams) {
  auto Sizes = parse(makeKernelModule(
      "k", {ByValIdPtr, ByValPaddedPtr, ByValPackedPtr, ByValOffsetPtr},
      /*ByValParams=*/{0, 1, 2, 3}));
  EXPECT_EQ(Sizes["k"], (KernelArgSizesT{8, 8, 5, 16}));
}

TEST(SPIRVKernelArgSizes, UnknownTypes) {
  auto Sizes = parse(makeKernelModule("k", {Sampler, I32}));
  EXPECT_EQ(Sizes["k"], (KernelArgSizesT{std::nullopt, 4}));
}

TEST(SPIRVKernelArgSizes, NameFillingWholeWords) {
  // A name whose length is a multiple of 4 is followed by a word of zeros.
  auto Sizes = parse(makeKernelModule("kern", {I32}));
  EXPECT_EQ(Sizes["kern"], (KernelArgSizesT{4}));
}

TEST(SPIRVKernelArgSizes, NonKernelEntryPointsIgnored) {
  auto Sizes =
      parse(makeKernelModule("k", {I32}, /*ByValParams=*/{},
                             AddressingPhysical64, ExecModelGLCompute));
  EXPECT_TRUE(Sizes.empty());
}

TEST(SPIRVKernelArgSizes, ByteSwappedModule) {
  std::vector<std::byte> Bytes = makeKernelModule("k", {I32, I64});
  for (size_t I = 0; I < Bytes.size(); I += sizeof(uint32_t))
    std::reverse(Bytes.begin() + I, Bytes.begin() + I + sizeof(uint32_t));
  auto Sizes = parse(Bytes);
  EXPECT_EQ(Sizes["k"], (KernelArgSizesT{4, 8}));
}

TEST(SPIRVKernelArgSizes, MalformedModules) {
  std::vector<std::byte> Valid = makeKernelModule("k", {I32});

  EXPECT_TRUE(getSPIRVKernelArgSizes(nullptr, 0).empty());

  std::vector<std::byte> BadMagic = Valid;
  BadMagic[0] = std::byte{0};
  EXPECT_TRUE(parse(BadMagic).empty());

  std::vector<std::byte> Unaligned = Valid;
  Unaligned.pop_back();
  EXPECT_TRUE(parse(Unaligned).empty());

  // An instruction whose word count runs past the end of the module.
  std::vector<std::byte> Truncated = Valid;
  const uint32_t TruncatedInst = 4u << 16 | op::TypeInt;
  const auto *TruncatedBytes =
      reinterpret_cast<const std::byte *>(&TruncatedInst);
  Truncated.insert(Truncated.end(), TruncatedBytes,
                   TruncatedBytes + sizeof(TruncatedInst));
  EXPECT_TRUE(parse(Truncated).empty());
}

namespace {

constexpr const char *KernelName = "spirv_kernel";

ur_result_t redefinedProgramGetInfo(void *pParams) {
  auto *Params = reinterpret_cast<ur_program_get_info_params_t *>(pParams);
  if (*Params->ppropName == UR_PROGRAM_INFO_NUM_KERNELS) {
    if (*Params->ppPropValue)
      *static_cast<size_t *>(*Params->ppPropValue) = 1;
    if (*Params->ppPropSizeRet)
      **Params->ppPropSizeRet = sizeof(size_t);
    return UR_RESULT_SUCCESS;
  }
  if (*Params->ppropName == UR_PROGRAM_INFO_KERNEL_NAMES) {
    const size_t Size = std::strlen(KernelName) + 1;
    if (*Params->ppPropValue)
      std::memcpy(*Params->ppPropValue, KernelName, Size);
    if (*Params->ppPropSizeRet)
      **Params->ppPropSizeRet = Size;
    return UR_RESULT_SUCCESS;
  }
  return sycl::unittest::MockAdapter::mock_urProgramGetInfo(pParams);
}

ur_result_t redefinedKernelGetInfo(void *pParams) {
  auto *Params = reinterpret_cast<ur_kernel_get_info_params_t *>(pParams);
  if (*Params->ppropName == UR_KERNEL_INFO_FUNCTION_NAME) {
    const size_t Size = std::strlen(KernelName) + 1;
    if (*Params->ppPropValue)
      std::memcpy(*Params->ppPropValue, KernelName, Size);
    if (*Params->ppPropSizeRet)
      **Params->ppPropSizeRet = Size;
    return UR_RESULT_SUCCESS;
  }
  if (*Params->ppPropSizeRet)
    **Params->ppPropSizeRet = 0;
  return UR_RESULT_SUCCESS;
}

class SPIRVKernelArgSizeCheck : public ::testing::Test {
protected:
  void SetUp() override {
    mock::getCallbacks().set_replace_callback("urProgramGetInfo",
                                              &redefinedProgramGetInfo);
    mock::getCallbacks().set_replace_callback("urKernelGetInfo",
                                              &redefinedKernelGetInfo);
  }

  // Kernel parameters: global int pointer, sycl::id<1> by value, int, local
  // int pointer.
  sycl::kernel getKernel() {
    std::vector<std::byte> Spv =
        makeKernelModule(KernelName, {GlobalPtr, ByValIdPtr, I32, LocalPtr},
                         /*ByValParams=*/{1});
    auto Src = syclex::create_kernel_bundle_from_source(
        Q.get_context(), syclex::source_language::spirv, Spv);
    auto Exe = syclex::build(Src);
    return Exe.ext_oneapi_get_kernel(KernelName);
  }

  void setValidArgs(sycl::handler &CGH) {
    CGH.set_arg(0, Ptr);
    CGH.set_arg(1, sycl::id<1>{0});
    CGH.set_arg(2, 0);
    CGH.set_arg(3, sycl::local_accessor<int, 1>{sycl::range<1>{16}, CGH});
  }

  sycl::unittest::UrMock<> Mock;
  sycl::queue Q;
  int *Ptr = reinterpret_cast<int *>(0x1000);
};

void expectArgSizeError(const std::function<void()> &F,
                        const std::string &Expected) {
  try {
    F();
    FAIL() << "Expected an errc::kernel_argument exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::kernel_argument);
    EXPECT_NE(std::string(E.what()).find(Expected), std::string::npos)
        << E.what();
  }
}

} // namespace

TEST_F(SPIRVKernelArgSizeCheck, MatchingSizesAreAccepted) {
  sycl::kernel K = getKernel();
  EXPECT_NO_THROW(Q.submit([&](sycl::handler &CGH) {
    setValidArgs(CGH);
    CGH.parallel_for(sycl::range<1>{1}, K);
  }));
}

TEST_F(SPIRVKernelArgSizeCheck, UndersizedValueIsRejected) {
  sycl::kernel K = getKernel();
  // The literal 0 is a 4-byte int, while the kernel expects an 8-byte id<1>.
  expectArgSizeError(
      [&] {
        Q.submit([&](sycl::handler &CGH) {
          setValidArgs(CGH);
          CGH.set_arg(1, 0);
          CGH.parallel_for(sycl::range<1>{1}, K);
        });
      },
      "argument 1 of kernel 'spirv_kernel' is 4 bytes, but the kernel "
      "expects 8 bytes");
}

TEST_F(SPIRVKernelArgSizeCheck, OversizedValueIsRejected) {
  sycl::kernel K = getKernel();
  expectArgSizeError(
      [&] {
        Q.submit([&](sycl::handler &CGH) {
          setValidArgs(CGH);
          CGH.set_arg(2, int64_t{0});
          CGH.parallel_for(sycl::range<1>{1}, K);
        });
      },
      "argument 2 of kernel 'spirv_kernel' is 8 bytes, but the kernel "
      "expects 4 bytes");
}

TEST_F(SPIRVKernelArgSizeCheck, RawKernelArgIsChecked) {
  sycl::kernel K = getKernel();
  int64_t Value = 0;
  expectArgSizeError(
      [&] {
        Q.submit([&](sycl::handler &CGH) {
          setValidArgs(CGH);
          CGH.set_arg(1, syclex::raw_kernel_arg(&Value, 2));
          CGH.parallel_for(sycl::range<1>{1}, K);
        });
      },
      "argument 1 of kernel 'spirv_kernel' is 2 bytes");
}

TEST_F(SPIRVKernelArgSizeCheck, GraphNodeIsChecked) {
  sycl::kernel K = getKernel();
  syclex::command_graph Graph{Q.get_context(), Q.get_device()};
  expectArgSizeError(
      [&] {
        Graph.add([&](sycl::handler &CGH) {
          setValidArgs(CGH);
          CGH.set_arg(1, 0);
          CGH.parallel_for(sycl::range<1>{1}, K);
        });
      },
      "argument 1 of kernel 'spirv_kernel' is 4 bytes");
}
