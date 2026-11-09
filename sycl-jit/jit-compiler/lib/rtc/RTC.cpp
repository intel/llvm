//===- RTC.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RTC.h"
#include "JITBinaryInfo.h"
#include "helper/ErrorHelper.h"
#include "rtc/DeviceCompilation.h"
#include "translation/SPIRVLLVMTranslation.h"
#include "translation/Translation.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TimeProfiler.h>

#include <clang/Options/Options.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#ifdef SYCL_JIT_HAVE_SPIRV_TOOLS
#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.h>
#endif

using namespace jit_compiler;

JIT_EXPORT_SYMBOL RTCHashResult calculateHash(InMemoryFile SourceFile,
                                              View<InMemoryFile> IncludeFiles,
                                              View<const char *> UserArgs,
                                              BinaryFormat Format) {
  llvm::opt::InputArgList UserArgList;
  if (auto Error = parseUserArgs(UserArgs).moveInto(UserArgList)) {
    return errorTo<RTCHashResult>(std::move(Error),
                                  "Parsing of user arguments failed",
                                  /*IsHash=*/false);
  }

  auto Start = std::chrono::high_resolution_clock::now();
  std::string Hash;
  if (auto Error = calculateHash(SourceFile, IncludeFiles, UserArgList, Format)
                       .moveInto(Hash)) {
    return errorTo<RTCHashResult>(std::move(Error), "Hashing failed",
                                  /*IsHash=*/false);
  }
  auto Stop = std::chrono::high_resolution_clock::now();

  if (UserArgList.hasArg(clang::options::OPT_ftime_trace_EQ)) {
    std::chrono::duration<double, std::milli> HashTime = Stop - Start;
    llvm::dbgs() << "Hashing of " << SourceFile.Path << " took "
                 << int(HashTime.count()) << " ms\n";
  }

  return RTCHashResult{Hash.c_str()};
}

JIT_EXPORT_SYMBOL RTCResult compileSYCL(InMemoryFile SourceFile,
                                        View<InMemoryFile> IncludeFiles,
                                        View<const char *> UserArgs,
                                        View<char> CachedIR, bool SaveIR,
                                        BinaryFormat Format) {
  llvm::LLVMContext Context;
  std::string BuildLog;
  configureDiagnostics(Context, BuildLog);

  llvm::opt::InputArgList UserArgList;
  if (auto Error = parseUserArgs(UserArgs).moveInto(UserArgList)) {
    return errorTo<RTCResult>(std::move(Error),
                              "Parsing of user arguments failed",
                              RTCResult::RTCErrorCode::INVALID);
  }

  llvm::StringRef TraceFileName;
  if (auto *Arg =
          UserArgList.getLastArg(clang::options::OPT_ftime_trace_EQ)) {
    TraceFileName = Arg->getValue();
    unsigned Granularity =
        500; // microseconds. Same default as in `clang::FrontendOptions`.
    if (auto *Arg = UserArgList.getLastArg(
            clang::options::OPT_ftime_trace_granularity_EQ)) {
      if (!llvm::to_integer(Arg->getValue(), Granularity)) {
        return errorTo<RTCResult>(
            llvm::createStringError("invalid integral value '" +
                                    std::string(Arg->getValue()) + "' in '" +
                                    Arg->getAsString(UserArgList) + "'"),
            "Invalid time trace granularity", RTCResult::RTCErrorCode::INVALID);
      }
    }
    bool Verbose =
        UserArgList.hasArg(clang::options::OPT_ftime_trace_verbose);

    llvm::timeTraceProfilerInitialize(Granularity, /*ProcName=*/"sycl-rtc",
                                      Verbose);
  }

  std::unique_ptr<llvm::Module> Module;

  if (CachedIR.size() > 0) {
    llvm::StringRef IRStr{CachedIR.begin(), CachedIR.size()};
    std::unique_ptr<llvm::MemoryBuffer> IRBuf =
        llvm::MemoryBuffer::getMemBuffer(IRStr, /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);
    if (auto Error = llvm::parseBitcodeFile(*IRBuf, Context).moveInto(Module)) {
      // Not a fatal error, we'll just compile the source string normally.
      BuildLog.append(formatError(std::move(Error),
                                  "Loading of cached device code failed"));
    }
  }

  bool FromSource = !Module;
  if (FromSource) {
    if (auto Error = compileDeviceCode(SourceFile, IncludeFiles, UserArgList,
                                       BuildLog, Context, Format)
                         .moveInto(Module)) {
      return errorTo<RTCResult>(std::move(Error), "Device compilation failed");
    }
  }

  RTCDeviceCodeIR IR;
  if (SaveIR && FromSource) {
    std::string BCString;
    llvm::raw_string_ostream BCStream{BCString};
    llvm::WriteBitcodeToFile(*Module, BCStream);
    IR = RTCDeviceCodeIR{BCString.data(), BCString.data() + BCString.size()};
  }

  if (auto Error =
          linkDeviceLibraries(*Module, UserArgList, BuildLog, Format)) {
    return errorTo<RTCResult>(std::move(Error), "Device linking failed");
  }

  auto PostLinkResultOrError = performPostLink(std::move(Module), UserArgList);
  if (!PostLinkResultOrError) {
    return errorTo<RTCResult>(PostLinkResultOrError.takeError(),
                              "Post-link phase failed");
  }
  auto [BundleInfo, Modules] = std::move(*PostLinkResultOrError);

  for (auto [DevImgInfo, Module] :
       llvm::zip_equal(BundleInfo.DevImgInfos, Modules)) {
    if (auto Error =
            Translator::translate(*Module, JITContext::getInstance(), Format)
                .moveInto(DevImgInfo.BinaryInfo)) {
      return errorTo<RTCResult>(std::move(Error), "SPIR-V translation failed");
    }
  }

  encodeBuildOptions(BundleInfo, UserArgList);

  if (llvm::timeTraceProfilerEnabled()) {
    auto Error = llvm::timeTraceProfilerWrite(
        TraceFileName, /*FallbackFileName=*/"trace.json");
    llvm::timeTraceProfilerCleanup();
    if (Error) {
      return errorTo<RTCResult>(std::move(Error), "Trace file writing failed");
    }
  }

  return RTCResult{std::move(BundleInfo), std::move(IR), BuildLog.c_str()};
}

JIT_EXPORT_SYMBOL void destroyBinary(BinaryAddress Address) {
  JITContext::getInstance().destroyBinary(Address);
}

#ifdef SYCL_JIT_HAVE_SPIRV_TOOLS
namespace {
// Computes argument sizes following the OpenCL C layout rules that
// SPIR-V kernels are compiled with. Types whose layout is not certain are not
// recorded, so parameters of such types get size 0 and are not validated.
struct KernelArgSizeCollector {
  struct Layout {
    uint64_t Size, Align;
    bool IsPointer = false;
    uint32_t Pointee = 0;
    uint32_t StorageClass = 0;
  };
  uint64_t PointerSize = 0;
  bool Unsupported = false;
  std::unordered_map<uint32_t, std::string> EntryPoints;
  std::unordered_set<uint32_t> ByValParams;
  // Packed structs and types with explicit offsets or strides.
  std::unordered_set<uint32_t> NonNaturalLayout;
  std::unordered_map<uint32_t, uint64_t> Constants;
  std::unordered_map<uint32_t, Layout> Types;
  std::vector<std::pair<std::string, std::vector<uint64_t>>> Kernels;
  bool InKernel = false;

  const Layout *find(uint32_t Id) const {
    auto It = Types.find(Id);
    return It == Types.end() ? nullptr : &It->second;
  }

  uint64_t paramSize(uint32_t TypeId, uint32_t ParamId) const {
    const Layout *Ty = find(TypeId);
    if (Ty && Ty->IsPointer) {
      // USM pointers are passed for CrossWorkgroup pointers.
      if (!ByValParams.count(ParamId))
        return Ty->StorageClass == SpvStorageClassCrossWorkgroup ? Ty->Size : 0;
      const Layout *Pointee = find(Ty->Pointee);
      return Pointee ? Pointee->Size : 0;
    }
    return Ty ? Ty->Size : 0;
  }

  // SPIRV-Tools has validated the length and operand grammar of the
  // instruction, so all operands required by its opcode are present.
  void handle(const spv_parsed_instruction_t &I) {
    const uint32_t *Op = I.words + 1;
    const size_t NumOps = I.num_words - 1u;
    if (I.opcode != SpvOpFunctionParameter)
      InKernel = false;

    switch (I.opcode) {
    case SpvOpMemoryModel:
      PointerSize = Op[0] == SpvAddressingModelPhysical64   ? 8
                    : Op[0] == SpvAddressingModelPhysical32 ? 4
                                                            : 0;
      break;
    case SpvOpEntryPoint:
      if (Op[0] == SpvExecutionModelKernel)
        EntryPoints[Op[1]] = reinterpret_cast<const char *>(Op + 2);
      break;
    case SpvOpDecorate:
      if (Op[1] == SpvDecorationFuncParamAttr && NumOps > 2 &&
          Op[2] == SpvFunctionParameterAttributeByVal)
        ByValParams.insert(Op[0]);
      else if (Op[1] == SpvDecorationCPacked ||
               Op[1] == SpvDecorationArrayStride)
        NonNaturalLayout.insert(Op[0]);
      break;
    case SpvOpMemberDecorate:
      if (Op[2] == SpvDecorationOffset)
        NonNaturalLayout.insert(Op[0]);
      break;
    case SpvOpGroupDecorate:
    case SpvOpGroupMemberDecorate:
      Unsupported = true;
      break;
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
      // Odd widths (arbitrary precision integers) have no host-side layout.
      if (Op[1] % 8 == 0 && (Op[1] / 8 & (Op[1] / 8 - 1)) == 0)
        Types[Op[0]] = {Op[1] / 8u, Op[1] / 8u};
      break;
    case SpvOpTypeVector:
      // 3-element vectors are laid out like 4-element ones.
      if (const Layout *Elem = find(Op[1])) {
        const uint64_t Size = Elem->Size * (Op[2] == 3 ? 4 : Op[2]);
        Types[Op[0]] = {Size, Size};
      }
      break;
    case SpvOpTypeArray: {
      const Layout *Elem = find(Op[1]);
      auto Length = Constants.find(Op[2]);
      if (Elem && Length != Constants.end() && !NonNaturalLayout.count(Op[0]))
        Types[Op[0]] = {Elem->Size * Length->second, Elem->Align};
      break;
    }
    case SpvOpTypeStruct: {
      if (NumOps < 2 || NonNaturalLayout.count(Op[0]))
        break;
      Layout Struct{0, 1};
      for (size_t M = 1; M < NumOps; ++M) {
        const Layout *Member = find(Op[M]);
        if (!Member)
          return;
        Struct.Size =
            (Struct.Size + Member->Align - 1) / Member->Align * Member->Align +
            Member->Size;
        Struct.Align = std::max(Struct.Align, Member->Align);
      }
      Struct.Size =
          (Struct.Size + Struct.Align - 1) / Struct.Align * Struct.Align;
      Types[Op[0]] = Struct;
      break;
    }
    case SpvOpTypeForwardPointer:
    case SpvOpTypePointer:
    case SpvOpTypeUntypedPointerKHR:
      if (PointerSize)
        Types[Op[0]] = {PointerSize, PointerSize, true,
                        I.opcode == SpvOpTypePointer ? Op[2] : 0, Op[1]};
      break;
    case SpvOpConstant:
      // Only integer constants are needed, as array lengths.
      if (find(I.type_id))
        Constants[I.result_id] =
            NumOps > 3 ? Op[2] | uint64_t{Op[3]} << 32 : Op[2];
      break;
    case SpvOpFunction:
      if (auto It = EntryPoints.find(I.result_id); It != EntryPoints.end()) {
        Kernels.emplace_back(It->second, std::vector<uint64_t>{});
        InKernel = true;
      }
      break;
    case SpvOpFunctionParameter:
      if (InKernel)
        Kernels.back().second.push_back(paramSize(I.type_id, I.result_id));
      break;
    default:
      break;
    }
  }
};
} // namespace
#endif // SYCL_JIT_HAVE_SPIRV_TOOLS

JIT_EXPORT_SYMBOL DynArray<SPIRVKernelInfo>
getSPIRVKernelInfo([[maybe_unused]] View<uint8_t> SPIRVBinary) {
#ifdef SYCL_JIT_HAVE_SPIRV_TOOLS
  if (SPIRVBinary.size() % sizeof(uint32_t) != 0)
    return {};
  // spvBinaryParse needs aligned words.
  std::vector<uint32_t> Words(SPIRVBinary.size() / sizeof(uint32_t));
  std::memcpy(Words.data(), SPIRVBinary.begin(), SPIRVBinary.size());

  KernelArgSizeCollector Collector;
  spv_context Context = spvContextCreate(SPV_ENV_UNIVERSAL_1_6);
  spv_diagnostic Diagnostic = nullptr;
  const spv_result_t Status = spvBinaryParse(
      Context, &Collector, Words.data(), Words.size(), /*parse_header=*/nullptr,
      [](void *Collector, const spv_parsed_instruction_t *I) {
        static_cast<KernelArgSizeCollector *>(Collector)->handle(*I);
        return SPV_SUCCESS;
      },
      &Diagnostic);
  spvDiagnosticDestroy(Diagnostic);
  spvContextDestroy(Context);
  if (Status != SPV_SUCCESS || Collector.Unsupported)
    return {};

  DynArray<SPIRVKernelInfo> Result{Collector.Kernels.size()};
  SPIRVKernelInfo *Info = Result.begin();
  for (const auto &[Name, ArgSizes] : Collector.Kernels) {
    Info->Name = std::string_view{Name};
    Info->ArgSizes = DynArray<uint64_t>{ArgSizes.begin(), ArgSizes.end()};
    ++Info;
  }
  return Result;
#else
  return {};
#endif
}
