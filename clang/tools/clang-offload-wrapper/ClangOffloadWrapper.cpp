//===-- clang-offload-wrapper/ClangOffloadWrapper.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the offload wrapper tool. It takes offload target binaries
/// as input and creates wrapper bitcode file containing target binaries
/// packaged as data. Wrapper bitcode also includes initialization code which
/// registers target binaries in offloading runtime at program startup.
///
//===----------------------------------------------------------------------===//

#include "SymPropReader.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/TargetParser/Triple.h"
#ifndef NDEBUG
#include "llvm/IR/Verifier.h"
#endif // NDEBUG
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VCSRevision.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>

// For device image compression.
#include <llvm/Support/Compression.h>

#define OPENMP_OFFLOAD_IMAGE_VERSION "1.0"

using namespace llvm;
using namespace llvm::object;

// Fields in the binary descriptor which are made available to SYCL runtime
// by the offload wrapper. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
static constexpr char COL_CODE[] = "Code";
static constexpr char COL_SYM[] = "Symbols";
static constexpr char COL_PROPS[] = "Properties";
static constexpr char COL_MANIFEST[] = "Manifest";

// Offload models supported by this tool. The support basically means mapping
// a string representation given at the command line to a value from this
// enum.
enum OffloadKind {
  Unknown = 0,
  Host,
  OpenMP,
  HIP,
  SYCL,
  First = Host,
  Last = SYCL
};

namespace llvm {
template <> struct DenseMapInfo<OffloadKind> {
  static inline OffloadKind getEmptyKey() {
    return static_cast<OffloadKind>(DenseMapInfo<unsigned>::getEmptyKey());
  }

  static inline OffloadKind getTombstoneKey() {
    return static_cast<OffloadKind>(DenseMapInfo<unsigned>::getTombstoneKey());
  }

  static unsigned getHashValue(const OffloadKind &Val) {
    return DenseMapInfo<unsigned>::getHashValue(static_cast<unsigned>(Val));
  }

  static bool isEqual(const OffloadKind &LHS, const OffloadKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadWrapperCategory("clang-offload-wrapper options");

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("Output filename"),
                                   cl::value_desc("filename"),
                                   cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    SymPropBCFiles("sym-prop-bc-files", cl::Optional,
                   cl::desc("File with list of wrapped BC input files that "
                            "will be used to supply symbols and properties."),
                   cl::value_desc("filename"),
                   cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> Verbose("v", cl::desc("verbose output"),
                             cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<input files>"),
                                    cl::cat(ClangOffloadWrapperCategory));

// CLI options for device image compression.
static cl::opt<bool> OffloadCompressDevImgs(
    "offload-compress", cl::init(false), cl::Optional,
    cl::desc("Enable device image compression using ZSTD."),
    cl::cat(ClangOffloadWrapperCategory));

static cl::opt<int>
    OffloadCompressLevel("offload-compression-level", cl::init(10),
                         cl::Optional,
                         cl::desc("ZSTD Compression level. Default: 10"),
                         cl::cat(ClangOffloadWrapperCategory));

static cl::opt<int>
    OffloadCompressThreshold("offload-compression-threshold", cl::init(512),
                             cl::Optional,
                             cl::desc("Threshold (in bytes) over which to "
                                      "compress images. Default: 512"),
                             cl::cat(ClangOffloadWrapperCategory));

// Binary image formats supported by this tool. The support basically means
// mapping string representation given at the command line to a value from this
// enum. No format checking is performed.
enum BinaryImageFormat {
  none,   // image kind is not determined
  native, // image kind is native
  // portable image kinds go next
  spirv,          // SPIR-V
  llvmbc,         // LLVM bitcode
  compressed_none // compressed image with unknown format
};

/// Sets offload kind.
static cl::list<OffloadKind> Kinds(
    "kind", cl::desc("offload kind:"), cl::OneOrMore,
    cl::values(clEnumValN(Unknown, "unknown", "unknown"),
               clEnumValN(Host, "host", "host"),
               clEnumValN(OpenMP, "openmp", "OpenMP"),
               clEnumValN(HIP, "hip", "HIP"), clEnumValN(SYCL, "sycl", "SYCL")),
    cl::cat(ClangOffloadWrapperCategory));

/// Sets binary image format.
static cl::list<BinaryImageFormat>
    Formats("format", cl::desc("device binary image formats:"), cl::ZeroOrMore,
            cl::values(clEnumVal(none, "not set"),
                       clEnumVal(native, "unknown or native"),
                       clEnumVal(spirv, "SPIRV binary"),
                       clEnumVal(llvmbc, "LLVMIR bitcode")),
            cl::cat(ClangOffloadWrapperCategory));

/// Sets offload target.
static cl::list<std::string> Targets("target", cl::ZeroOrMore,
                                     cl::desc("offload target triple"),
                                     cl::cat(ClangOffloadWrapperCategory));

/// Sets compile options for device binary image.
static cl::list<std::string>
    CompileOptions("compile-opts", cl::ZeroOrMore,
                   cl::desc("compile options passed to the offload runtime"),
                   cl::cat(ClangOffloadWrapperCategory));

/// Sets link options for device binary image.
static cl::list<std::string>
    LinkOptions("link-opts", cl::ZeroOrMore,
                cl::desc("link options passed to the offload runtime"),
                cl::cat(ClangOffloadWrapperCategory));

/// Sets the name of the file containing offload function entries
static cl::list<std::string> Entries(
    "entries", cl::ZeroOrMore,
    cl::desc("File listing all offload function entries, SYCL offload only"),
    cl::value_desc("filename"), cl::cat(ClangOffloadWrapperCategory));

/// Sets the name of the file containing arbitrary properties for current device
/// binary image
static cl::list<std::string> Properties(
    "properties", cl::ZeroOrMore,
    cl::desc("File listing device binary image properties, SYCL offload only"),
    cl::value_desc("filename"), cl::cat(ClangOffloadWrapperCategory));

/// Specifies the target triple of the host wrapper.
static cl::opt<std::string> Target(
    "host", cl::Optional,
    cl::desc("Target triple for the output module. If omitted, the host\n"
             "triple is used."),
    cl::value_desc("triple"), cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> EmitRegFuncs("emit-reg-funcs", cl::NotHidden,
                                  cl::init(true), cl::Optional,
                                  cl::desc("Emit [un-]registration functions"),
                                  cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string> DescriptorName(
    "desc-name", cl::Optional, cl::init("descriptor"),
    cl::desc(
        "Specifies offload descriptor symbol name: '.<offload kind>.<name>',\n"
        "and makes it globally visible"),
    cl::value_desc("name"), cl::cat(ClangOffloadWrapperCategory));

// clang-format off
/// batch mode - All input files are treated as a table file.  One table file per target.
///            - Table files consist of a table of filenames that provide
///            - Code, Symbols, Properties, etc.
static cl::opt<bool> BatchMode(
    "batch", cl::NotHidden, cl::init(false), cl::Optional,
    cl::desc("All input files are treated as a table file.  One table file per target.\n"
             "Table files consist of a table of filenames that provide\n"
             "Code, Symbols, Properties, etc.\n"
             "Example input table file in batch mode:\n"
             "  [Code|Symbols|Properties|Manifest]\n"
             "  a_0.bc|a_0.sym|a_0.props|a_0.mnf\n"
             "  a_1.bin|||\n"
             "Example usage:\n"
             "  clang-offload-wrapper -batch -host=x86_64-unknown-linux-gnu\n"
             "    -kind=openmp -target=spir64_gen table1.txt\n"
             "    -kind=openmp -target=spir64     table2.txt"),
    cl::cat(ClangOffloadWrapperCategory));
// clang-format on

static StringRef offloadKindToString(OffloadKind Kind) {
  switch (Kind) {
  case OffloadKind::Unknown:
    return "unknown";
  case OffloadKind::Host:
    return "host";
  case OffloadKind::OpenMP:
    return "openmp";
  case OffloadKind::HIP:
    return "hip";
  case OffloadKind::SYCL:
    return "sycl";
  }
  llvm_unreachable("bad offload kind");

  return "<ERROR>";
}

static StringRef formatToString(BinaryImageFormat Fmt) {
  switch (Fmt) {
  case BinaryImageFormat::none:
    return "none";
  case BinaryImageFormat::spirv:
    return "spirv";
  case BinaryImageFormat::llvmbc:
    return "llvmbc";
  case BinaryImageFormat::native:
    return "native";
  case BinaryImageFormat::compressed_none:
    return "compressed_none";
  }
  llvm_unreachable("bad format");

  return "<ERROR>";
}

static cl::opt<bool> SaveTemps(
    "save-temps",
    cl::desc("Save temporary files that may be produced by the tool. "
             "This option forces print-out of the temporary files' names."),
    cl::Hidden);

static cl::opt<bool> AddOpenMPOffloadNotes(
    "add-omp-offload-notes",
    cl::desc("Add LLVMOMPOFFLOAD ELF notes to ELF device images."), cl::Hidden);

namespace {

/// Implements binary image information collecting and wrapping it in a host
/// bitcode file.
class BinaryWrapper {
public:
  /// Represents a single image to wrap.
  class Image {
  public:
    Image(const llvm::StringRef File_, const llvm::StringRef Manif_,
          const llvm::StringRef Tgt_, BinaryImageFormat Fmt_,
          const llvm::StringRef CompileOpts_, const llvm::StringRef LinkOpts_,
          const llvm::StringRef EntriesFile_, const llvm::StringRef PropsFile_)
        : File(File_.str()), Manif(Manif_.str()), Tgt(Tgt_.str()), Fmt(Fmt_),
          CompileOpts(CompileOpts_.str()), LinkOpts(LinkOpts_.str()),
          EntriesFile(EntriesFile_.str()), PropsFile(PropsFile_.str()) {}

    /// Name of the file with actual contents
    const std::string File;
    /// Name of the manifest file
    const std::string Manif;
    /// Offload target architecture
    const std::string Tgt;
    /// Format
    const BinaryImageFormat Fmt;
    /// Compile options
    const std::string CompileOpts;
    /// Link options
    const std::string LinkOpts;
    /// File listing contained entries
    const std::string EntriesFile;
    /// File with properties
    const std::string PropsFile;

    friend raw_ostream &operator<<(raw_ostream &Out, const Image &Img);
  };

private:
  using SameKindPack = llvm::SmallVector<std::unique_ptr<Image>, 4>;

  LLVMContext C;
  Module M;

  StructType *EntryTy = nullptr;
  StructType *ImageTy = nullptr;
  StructType *DescTy = nullptr;

  // SYCL image and binary descriptor types have diverged from libomptarget
  // definitions, but presumably they will converge in future. So, these SYCL
  // specific types should be removed if/when this happens.
  StructType *SyclImageTy = nullptr;
  StructType *SyclDescTy = nullptr;
  StructType *SyclPropSetTy = nullptr;
  StructType *SyclPropTy = nullptr;
  PointerType *PtrTy = nullptr;

  /// Records all added device binary images per offload kind.
  llvm::DenseMap<OffloadKind, std::unique_ptr<SameKindPack>> Packs;
  /// Records all created memory buffers for safe auto-gc
  llvm::SmallVector<std::unique_ptr<MemoryBuffer>, 4> AutoGcBufs;

public:
  void addImage(const OffloadKind Kind, llvm::StringRef File,
                llvm::StringRef Manif, llvm::StringRef Tgt,
                const BinaryImageFormat Fmt, llvm::StringRef CompileOpts,
                llvm::StringRef LinkOpts, llvm::StringRef EntriesFile,
                llvm::StringRef PropsFile) {
    std::unique_ptr<SameKindPack> &Pack = Packs[Kind];
    if (!Pack)
      Pack.reset(new SameKindPack());
    Pack->emplace_back(std::make_unique<Image>(
        File, Manif, Tgt, Fmt, CompileOpts, LinkOpts, EntriesFile, PropsFile));
  }

  std::string ToolName;
  std::string ObjcopyPath;
  // Temporary file names that may be created during adding notes
  // to ELF offload images. Use -save-temps to keep them and also
  // see their names. A temporary file's name includes the name
  // of the original input ELF image, so you can easily match
  // them, if you have multiple inputs.
  std::vector<std::string> TempFiles;

private:
  std::unique_ptr<SymPropReader> MySymPropReader;

  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerTypeSize(getPtrTy())) {
    case 4u:
      return Type::getInt32Ty(C);
    case 8u:
      return Type::getInt64Ty(C);
    }
    llvm_unreachable("unsupported pointer type size");
  }

  SmallVector<Constant *, 2> getSizetConstPair(size_t First, size_t Second) {
    return SmallVector<Constant *, 2>{ConstantInt::get(getSizeTTy(), First),
                                      ConstantInt::get(getSizeTTy(), Second)};
  }

  std::pair<Constant *, Constant *>
  addStructArrayToModule(ArrayRef<Constant *> ArrayData, Type *ElemTy) {

    auto *PtrTy = getPtrTy();

    if (ArrayData.size() == 0) {
      auto *NullPtr = Constant::getNullValue(PtrTy);
      return std::make_pair(NullPtr, NullPtr);
    }
    assert(ElemTy == ArrayData[0]->getType() && "elem type mismatch");
    auto *Arr =
        ConstantArray::get(ArrayType::get(ElemTy, ArrayData.size()), ArrayData);
    auto *ArrGlob = new GlobalVariable(M, Arr->getType(), true,
                                       GlobalVariable::InternalLinkage, Arr,
                                       "__sycl_offload_prop_sets_arr");
    if (Verbose)
      errs() << "  global added: " << ArrGlob->getName() << "\n";

    auto *ArrB = ConstantExpr::getGetElementPtr(
        ArrGlob->getValueType(), ArrGlob, getSizetConstPair(0u, 0u));
    auto *ArrE =
        ConstantExpr::getGetElementPtr(ArrGlob->getValueType(), ArrGlob,
                                       getSizetConstPair(0, ArrayData.size()));
    return std::pair<Constant *, Constant *>(ArrB, ArrE);
  }

  // struct __tgt_offload_entry {
  //   void *addr;
  //   char *name;
  //   size_t size;
  //   int32_t flags;
  //   int32_t reserved;
  // };
  StructType *getEntryTy() {
    if (!EntryTy)
      EntryTy = StructType::create("__tgt_offload_entry", getPtrTy(),
                                   getPtrTy(), getSizeTTy(),
                                   Type::getInt32Ty(C), Type::getInt32Ty(C));
    return EntryTy;
  }

  // struct __tgt_device_image {
  //   void *ImageStart;
  //   void *ImageEnd;
  //   __tgt_offload_entry *EntriesBegin;
  //   __tgt_offload_entry *EntriesEnd;
  // };
  StructType *getDeviceImageTy() {
    if (!ImageTy)
      ImageTy = StructType::create("__tgt_device_image", getPtrTy(), getPtrTy(),
                                   getPtrTy(), getPtrTy());
    return ImageTy;
  }

  // struct __tgt_bin_desc {
  //   int32_t NumDeviceImages;
  //   __tgt_device_image *DeviceImages;
  //   __tgt_offload_entry *HostEntriesBegin;
  //   __tgt_offload_entry *HostEntriesEnd;
  // };
  StructType *getBinDescTy() {
    if (!DescTy)
      DescTy = StructType::create("__tgt_bin_desc", Type::getInt32Ty(C),
                                  getPtrTy(), getPtrTy(), getPtrTy());
    return DescTy;
  }

  // DeviceImageStructVersion change log:
  // -- version 2: updated to PI 1.2 binary image format
  const uint16_t DeviceImageStructVersion = 2;

  // typedef enum {
  //   PI_PROPERTY_TYPE_INT32,
  //   PI_PROPERTY_TYPE_STRING
  // } pi_property_type;
  //
  // struct _pi_device_binary_property_struct {
  //   char *Name;
  //   void *ValAddr;
  //   pi_property_type Type;
  //   uint64_t ValSize;
  // };

  StructType *getSyclPropTy() {
    if (!SyclPropTy) {
      SyclPropTy = StructType::create(
          {
              getPtrTy(),          // Name
              getPtrTy(),          // ValAddr
              Type::getInt32Ty(C), // Type
              Type::getInt64Ty(C)  // ValSize
          },
          "_pi_device_binary_property_struct");
    }
    return SyclPropTy;
  }

  // struct _pi_device_binary_property_set_struct {
  //   char *Name;
  //   _pi_device_binary_property_struct* PropertiesBegin;
  //   _pi_device_binary_property_struct* PropertiesEnd;
  // };

  StructType *getSyclPropSetTy() {
    if (!SyclPropSetTy) {
      SyclPropSetTy = StructType::create(
          {
              getPtrTy(), // Name
              getPtrTy(), // PropertiesBegin
              getPtrTy()  // PropertiesEnd
          },
          "_pi_device_binary_property_set_struct");
    }
    return SyclPropSetTy;
  }

  // SYCL specific image descriptor type.
  //  struct __tgt_device_image {
  //    /// version of this structure - for backward compatibility;
  //    /// all modifications which change order/type/offsets of existing fields
  //    /// should increment the version.
  //    uint16_t Version;
  //    /// the kind of offload model the image employs.
  //    uint8_t OffloadKind;
  //    /// format of the image data - SPIRV, LLVMIR bitcode,...
  //    uint8_t Format;
  //    /// null-terminated string representation of the device's target
  //    /// architecture
  //    const char *DeviceTargetSpec;
  //    /// a null-terminated string; target- and compiler-specific options
  //    /// which are suggested to use to "compile" program at runtime
  //    const char *CompileOptions;
  //    /// a null-terminated string; target- and compiler-specific options
  //    /// which are suggested to use to "link" program at runtime
  //    const char *LinkOptions;
  //    /// Pointer to the manifest data start
  //    const unsigned char *ManifestStart;
  //    /// Pointer to the manifest data end
  //    const unsigned char *ManifestEnd;
  //    /// Pointer to the device binary image start
  //    void *ImageStart;
  //    /// Pointer to the device binary image end
  //    void *ImageEnd;
  //    /// the entry table
  //    __tgt_offload_entry *EntriesBegin;
  //    __tgt_offload_entry *EntriesEnd;
  //    _pi_device_binary_property_set_struct PropertySetBegin;
  //    _pi_device_binary_property_set_struct PropertySetEnd;
  //  };
  //

  StructType *getSyclDeviceImageTy() {
    if (!SyclImageTy) {
      SyclImageTy = StructType::create(
          {
              Type::getInt16Ty(C), // Version
              Type::getInt8Ty(C),  // OffloadKind
              Type::getInt8Ty(C),  // Format
              getPtrTy(),          // DeviceTargetSpec
              getPtrTy(),          // CompileOptions
              getPtrTy(),          // LinkOptions
              getPtrTy(),          // ManifestStart
              getPtrTy(),          // ManifestEnd
              getPtrTy(),          // ImageStart
              getPtrTy(),          // ImageEnd
              getPtrTy(),          // EntriesBegin
              getPtrTy(),          // EntriesEnd
              getPtrTy(),          // PropertySetBegin
              getPtrTy()           // PropertySetEnd
          },
          "__tgt_device_image");
    }
    return SyclImageTy;
  }

  PointerType *getPtrTy() {
    PointerType *&Ty = PtrTy;
    Ty = Ty ? Ty : PointerType::getUnqual(C);
    return Ty;
  }

  const uint16_t BinDescStructVersion = 1;

  // SYCL specific binary descriptor type.
  // struct __tgt_bin_desc {
  //   /// version of this structure - for backward compatibility;
  //   /// all modifications which change order/type/offsets of existing fields
  //   /// should increment the version.
  //   uint16_t Version;
  //   uint16_t NumDeviceImages;
  //   __tgt_device_image *DeviceImages;
  //   /// the offload entry table
  //   __tgt_offload_entry *HostEntriesBegin;
  //   __tgt_offload_entry *HostEntriesEnd;
  // };
  StructType *getSyclBinDescTy() {
    if (!SyclDescTy) {
      SyclDescTy = StructType::create(
          {
              Type::getInt16Ty(C), // Version
              Type::getInt16Ty(C), // NumDeviceImages
              getPtrTy(),          // DeviceImages
              getPtrTy(),          // HostEntriesBegin
              getPtrTy()           // HostEntriesEnd
          },
          "__tgt_bin_desc");
    }
    return SyclDescTy;
  }

  Expected<MemoryBuffer *> loadFile(llvm::StringRef Name) {
    auto InputOrErr = MemoryBuffer::getFileOrSTDIN(Name);

    if (auto EC = InputOrErr.getError())
      return createFileError(Name, EC);

    AutoGcBufs.emplace_back(std::move(InputOrErr.get()));
    return AutoGcBufs.back().get();
  }

  Function *addDeclarationForNativeCPU(StringRef Name) {
    static FunctionType *NativeCPUFuncTy =
        FunctionType::get(Type::getVoidTy(C), {getPtrTy(), getPtrTy()}, false);
    static FunctionType *NativeCPUBuiltinTy =
        FunctionType::get(getPtrTy(), {getPtrTy()}, false);
    FunctionType *FTy;
    if (Name.starts_with("__dpcpp_nativecpu"))
      FTy = NativeCPUBuiltinTy;
    else
      FTy = NativeCPUFuncTy;
    auto FCalle = M.getOrInsertFunction(
        sycl::utils::addSYCLNativeCPUSuffix(Name).str(), FTy);
    Function *F = dyn_cast<Function>(FCalle.getCallee());
    if (F == nullptr)
      report_fatal_error("Unexpected callee");
    return F;
  }

  Expected<std::pair<Constant *, Constant *>>
  addDeclarationsForNativeCPU(StringRef EntriesFile) {
    Expected<MemoryBuffer *> MBOrErr = loadFile(EntriesFile);
    if (!MBOrErr)
      return MBOrErr.takeError();
    MemoryBuffer *MB = *MBOrErr;
    // the Native CPU PI Plug-in expects the BinaryStart field to point to an
    // array of struct nativecpu_entry {
    //   char *kernelname;
    //   unsigned char *kernel_ptr;
    // };
    StructType *NCPUEntryT =
        StructType::create({getPtrTy(), getPtrTy()}, "__nativecpu_entry");
    SmallVector<Constant *, 5> NativeCPUEntries;
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
      auto *NewDecl = addDeclarationForNativeCPU(*LI);
      NativeCPUEntries.push_back(ConstantStruct::get(
          NCPUEntryT,
          {addStringToModule(*LI, "__ncpu_function_name"), NewDecl}));
    }

    // Add an empty entry that we use as end iterator
    static auto *NativeCPUEndStr =
        addStringToModule("__nativecpu_end", "__ncpu_end_str");
    auto *NullPtr = llvm::ConstantPointerNull::get(getPtrTy());
    NativeCPUEntries.push_back(
        ConstantStruct::get(NCPUEntryT, {NativeCPUEndStr, NullPtr}));

    // Create the constant array containing the {kernel name, function pointers}
    // pairs
    ArrayType *ATy = ArrayType::get(NCPUEntryT, NativeCPUEntries.size());
    Constant *CA = ConstantArray::get(ATy, NativeCPUEntries);
    auto *GVar = new GlobalVariable(M, CA->getType(), true,
                                    GlobalVariable::InternalLinkage, CA,
                                    "__sycl_native_cpu_decls");
    auto *Begin = ConstantExpr::getGetElementPtr(GVar->getValueType(), GVar,
                                                 getSizetConstPair(0u, 0u));
    auto *End = ConstantExpr::getGetElementPtr(
        GVar->getValueType(), GVar,
        getSizetConstPair(0u, NativeCPUEntries.size()));
    return std::make_pair(Begin, End);
  }

  // Adds a global readonly variable that is initialized by given data to the
  // module.
  GlobalVariable *addGlobalArrayVariable(const Twine &Name,
                                         ArrayRef<char> Initializer,
                                         const Twine &Section = "") {
    auto *Arr = ConstantDataArray::get(C, Initializer);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    if (Verbose)
      errs() << "  global added: " << Var->getName() << "\n";
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    SmallVector<char, 32u> NameBuf;
    auto SectionName = Section.toStringRef(NameBuf);
    if (!SectionName.empty())
      Var->setSection(SectionName);
    return Var;
  }

  // Adds given buffer as a global variable into the module and returns a pair
  // of pointers that point to the beginning and end of the variable.
  std::pair<Constant *, Constant *>
  addArrayToModule(ArrayRef<char> Buf, const Twine &Name,
                   const Twine &Section = "") {
    auto *Var = addGlobalArrayVariable(Name, Buf, Section);
    auto *ImageB = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                  getSizetConstPair(0u, 0u));
    auto *ImageE = ConstantExpr::getGetElementPtr(
        Var->getValueType(), Var, getSizetConstPair(0u, Buf.size()));
    return std::make_pair(ImageB, ImageE);
  }

  // Adds given data buffer as constant byte array and returns a constant
  // pointer to it. The pointer type does not carry size information.
  Constant *addRawDataToModule(ArrayRef<char> Data, const Twine &Name) {
    auto *Var = addGlobalArrayVariable(Name, Data);
    auto *DataPtr = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                   getSizetConstPair(0u, 0u));
    return DataPtr;
  }

  // Creates all necessary data objects for the given image and returns a pair
  // of pointers that point to the beginning and end of the global variable that
  // contains the image data.
  std::pair<Constant *, Constant *>
  addDeviceImageToModule(ArrayRef<char> Buf, const Twine &Name,
                         OffloadKind Kind, StringRef TargetTriple) {
    // Create global variable for the image data.
    return addArrayToModule(Buf, Name,
                            TargetTriple.empty()
                                ? ""
                                : "__CLANG_OFFLOAD_BUNDLE__" +
                                      offloadKindToString(Kind) + "-" +
                                      TargetTriple);
  }

  // Creates a global variable of const char* type and creates an
  // initializer that initializes it with given string (with added null
  // terminator). Returns a link-time constant pointer (constant expr) to that
  // variable.
  Constant *addStringToModule(StringRef Str, const Twine &Name) {
    auto *Arr = ConstantDataArray::getString(C, Str);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    if (Verbose)
      errs() << "  global added: " << Var->getName() << "\n";
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    Constant *ZeroZero[] = {Zero, Zero};
    return ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
  }

  // Creates an array of __tgt_offload_entry that contains function info
  // for the given image. Returns a pair of pointers to the beginning and end
  // of the array, or a pair of nullptrs in case the entries file wasn't
  // specified.
  Expected<std::pair<Constant *, Constant *>>
  addSYCLOffloadEntriesToModule(StringRef EntriesFile) {
    if (EntriesFile.empty() && !MySymPropReader) {
      auto *NullPtr = Constant::getNullValue(getPtrTy());
      return std::pair<Constant *, Constant *>(NullPtr, NullPtr);
    }

    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    auto *i32Zero = ConstantInt::get(Type::getInt32Ty(C), 0u);
    auto *NullPtr = Constant::getNullValue(getPtrTy());

    std::vector<Constant *> EntriesInits;
    // Only the name field is used for SYCL now, others are for future OpenMP
    // compatibility and new SYCL features
    if (MySymPropReader) {
      for (uint64_t i = 0; i < MySymPropReader->getNumEntries(); i++)
        EntriesInits.push_back(ConstantStruct::get(
            getEntryTy(), NullPtr,
            addStringToModule(MySymPropReader->getEntryName(i),
                              "__sycl_offload_entry_name"),
            Zero, i32Zero, i32Zero));
    } else {
      Expected<MemoryBuffer *> MBOrErr = loadFile(EntriesFile);
      if (!MBOrErr)
        return MBOrErr.takeError();
      MemoryBuffer *MB = *MBOrErr;

      for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI)
        EntriesInits.push_back(ConstantStruct::get(
            getEntryTy(), NullPtr,
            addStringToModule(*LI, "__sycl_offload_entry_name"), Zero, i32Zero,
            i32Zero));
    }

    auto *Arr = ConstantArray::get(
        ArrayType::get(getEntryTy(), EntriesInits.size()), EntriesInits);
    auto *Entries = new GlobalVariable(M, Arr->getType(), true,
                                       GlobalVariable::InternalLinkage, Arr,
                                       "__sycl_offload_entries_arr");
    if (Verbose)
      errs() << "  global added: " << Entries->getName() << "\n";

    auto *EntriesB = ConstantExpr::getGetElementPtr(
        Entries->getValueType(), Entries, getSizetConstPair(0u, 0u));
    auto *EntriesE = ConstantExpr::getGetElementPtr(
        Entries->getValueType(), Entries,
        getSizetConstPair(0u, EntriesInits.size()));
    return std::pair<Constant *, Constant *>(EntriesB, EntriesE);
  }

  Expected<std::pair<Constant *, Constant *>>
  addSYCLPropertySetToModule(const llvm::util::PropertySet &PropSet) {
    std::vector<Constant *> PropInits;

    for (const auto &Prop : PropSet) {
      Constant *PropName = addStringToModule(Prop.first, "prop");
      Constant *PropValAddr = nullptr;
      Constant *PropType =
          ConstantInt::get(Type::getInt32Ty(C), Prop.second.getType());
      Constant *PropValSize = nullptr;

      switch (Prop.second.getType()) {
      case llvm::util::PropertyValue::UINT32: {
        // for known scalar types ValAddr is null, ValSize keeps the value
        PropValAddr = Constant::getNullValue(getPtrTy());
        PropValSize =
            ConstantInt::get(Type::getInt64Ty(C), Prop.second.asUint32());
        break;
      }
      case llvm::util::PropertyValue::BYTE_ARRAY: {
        const char *Ptr =
            reinterpret_cast<const char *>(Prop.second.asRawByteArray());
        uint64_t Size = Prop.second.getRawByteArraySize();
        PropValSize = ConstantInt::get(Type::getInt64Ty(C), Size);
        PropValAddr = addRawDataToModule(ArrayRef<char>(Ptr, Size), "prop_val");
        break;
      }
      default:
        llvm_unreachable_internal("unsupported property");
      }
      PropInits.push_back(ConstantStruct::get(
          getSyclPropTy(), PropName, PropValAddr, PropType, PropValSize));
    }
    return addStructArrayToModule(PropInits, getSyclPropTy());
  }

  // Given in-memory representation of a set of property sets, inserts it into
  // the wrapper object file. In-object representation is given below.
  //
  // column is a contiguous area of the wrapper object file;
  // relative location of columns can be arbitrary
  //
  //                             _pi_device_binary_property_struct
  // _pi_device_binary_property_set_struct   |
  //                     |                   |
  //                     v                   v
  // ...             # ...                # ...
  // PropSetsBegin--># Name0        +----># Name_00
  // PropSetsEnd--+  # PropsBegin0--+     # ValAddr_00
  // ...          |  # PropseEnd0------+  # Type_00
  //              |  # Name1           |  # ValSize_00
  //              |  # PropsBegin1---+ |  # ...
  //              |  # PropseEnd1--+ | |  # Name_0n
  //              +-># ...         | | |  # ValAddr_0n
  //                 #             | | |  # Type_0n
  //                 #             | | |  # ValSize_0n
  //                 #             | | +-># ...
  //                 #             | |    # ...
  //                 #             | +---># Name_10
  //                 #             |      # ValAddr_10
  //                 #             |      # Type_10
  //                 #             |      # ValSize_10
  //                 #             |      # ...
  //                 #             |      # Name_1m
  //                 #             |      # ValAddr_1m
  //                 #             |      # Type_1m
  //                 #             |      # ValSize_1m
  //                 #             +-----># ...
  //                 #                    #
  // Returns a pair of pointers to the beginning and end of the property set
  // array, or a pair of nullptrs in case the properties file wasn't specified.
  Expected<std::pair<Constant *, Constant *>>
  tformSYCLPropertySetRegistryFileToIR(StringRef PropRegistryFile) {

    std::unique_ptr<llvm::util::PropertySetRegistry> PropRegistry;

    if (MySymPropReader) {
      PropRegistry = MySymPropReader->getPropRegistry();
    } else {
      if (PropRegistryFile.empty()) {
        auto *NullPtr = Constant::getNullValue(getPtrTy());
        return std::pair<Constant *, Constant *>(NullPtr, NullPtr);
      }
      // load the property registry file
      Expected<MemoryBuffer *> MBOrErr = loadFile(PropRegistryFile);
      if (!MBOrErr)
        return MBOrErr.takeError();
      MemoryBuffer *MB = *MBOrErr;
      Expected<std::unique_ptr<llvm::util::PropertySetRegistry>> PropRegistryE =
          llvm::util::PropertySetRegistry::read(MB);
      if (!PropRegistryE)
        return PropRegistryE.takeError();
      std::unique_ptr<llvm::util::PropertySetRegistry> &PropRegistryFromFile =
          PropRegistryE.get();
      PropRegistry = std::move(PropRegistryFromFile);
    }

    std::vector<Constant *> PropSetsInits;

    // transform all property sets to IR and get the middle column image into
    // the PropSetsInits
    for (const auto &PropSet : *PropRegistry) {
      // create content in the rightmost column and get begin/end pointers
      Expected<std::pair<Constant *, Constant *>> Props =
          addSYCLPropertySetToModule(PropSet.second);
      if (!Props)
        return Props.takeError();
      // get the next the middle column element
      auto *Category = addStringToModule(PropSet.first, "SYCL_PropSetName");
      PropSetsInits.push_back(ConstantStruct::get(
          getSyclPropSetTy(), Category, Props.get().first, Props.get().second));
    }
    // now get content for the leftmost column - create the top-level
    // PropertySetsBegin/PropertySetsBegin entries and return pointers to them
    return addStructArrayToModule(PropSetsInits, getSyclPropSetTy());
  }

public:
    MemoryBuffer *addELFNotes(MemoryBuffer *Buf, StringRef OriginalFileName);

private:
  /// Creates binary descriptor for the given device images. Binary descriptor
  /// is an object that is passed to the offloading runtime at program startup
  /// and it describes all device images available in the executable or shared
  /// library. It is defined as follows
  ///
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__start_omp_offloading_entries;
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__stop_omp_offloading_entries;
  ///
  /// static const char Image0[] = { <Bufs.front() contents> };
  ///  ...
  /// static const char ImageN[] = { <Bufs.back() contents> };
  ///
  /// static const __tgt_device_image Images[] = {
  ///   {
  ///     Image0,                            /*ImageStart*/
  ///     Image0 + sizeof(Image0),           /*ImageEnd*/
  ///     __start_omp_offloading_entries,    /*EntriesBegin*/
  ///     __stop_omp_offloading_entries      /*EntriesEnd*/
  ///   },
  ///   ...
  ///   {
  ///     ImageN,                            /*ImageStart*/
  ///     ImageN + sizeof(ImageN),           /*ImageEnd*/
  ///     __start_omp_offloading_entries,    /*EntriesBegin*/
  ///     __stop_omp_offloading_entries      /*EntriesEnd*/
  ///   }
  /// };
  ///
  /// static const __tgt_bin_desc BinDesc = {
  ///   sizeof(Images) / sizeof(Images[0]),  /*NumDeviceImages*/
  ///   Images,                              /*DeviceImages*/
  ///   __start_omp_offloading_entries,      /*HostEntriesBegin*/
  ///   __stop_omp_offloading_entries        /*HostEntriesEnd*/
  /// };
  ///
  /// Global variable that represents BinDesc is returned.
  Expected<GlobalVariable *> createBinDesc(OffloadKind Kind,
                                           SameKindPack &Pack) {
    const std::string OffloadKindTag =
        (Twine(".") + offloadKindToString(Kind) + Twine("_offloading.")).str();

    Constant *EntriesB = nullptr, *EntriesE = nullptr;

    if (Kind != OffloadKind::SYCL) {
      // Create external begin/end symbols for the offload entries table.
      auto *EntriesStart = new GlobalVariable(
          M, getEntryTy(), /*isConstant*/ true, GlobalValue::ExternalLinkage,
          /*Initializer*/ nullptr, "__start_omp_offloading_entries");
      EntriesStart->setVisibility(GlobalValue::HiddenVisibility);
      auto *EntriesStop = new GlobalVariable(
          M, getEntryTy(), /*isConstant*/ true, GlobalValue::ExternalLinkage,
          /*Initializer*/ nullptr, "__stop_omp_offloading_entries");
      EntriesStop->setVisibility(GlobalValue::HiddenVisibility);

      // We assume that external begin/end symbols that we have created above
      // will be defined by the linker. But linker will do that only if linker
      // inputs have section with "omp_offloading_entries" name which is not
      // guaranteed. So, we just create dummy zero sized object in the offload
      // entries section to force linker to define those symbols.
      auto *DummyInit =
          ConstantAggregateZero::get(ArrayType::get(getEntryTy(), 0u));
      auto *DummyEntry = new GlobalVariable(
          M, DummyInit->getType(), true, GlobalVariable::ExternalLinkage,
          DummyInit, "__dummy.omp_offloading.entry");
      DummyEntry->setSection("omp_offloading_entries");
      DummyEntry->setVisibility(GlobalValue::HiddenVisibility);

      EntriesB = EntriesStart;
      EntriesE = EntriesStop;

      if (Verbose) {
        errs() << "  global added: " << EntriesStart->getName() << "\n";
        errs() << "  global added: " << EntriesStop->getName() << "\n";
      }
    } else {
      // Host entry table is not used in SYCL
      EntriesB = Constant::getNullValue(getPtrTy());
      EntriesE = Constant::getNullValue(getPtrTy());
    }

    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    auto *NullPtr = Constant::getNullValue(getPtrTy());
    Constant *ZeroZero[] = {Zero, Zero};

    // Create initializer for the images array.
    SmallVector<Constant *, 4u> ImagesInits;
    unsigned ImgId = 0;

    for (const auto &ImgPtr : Pack) {
      const BinaryWrapper::Image &Img = *(ImgPtr.get());
      if (Verbose)
        errs() << "adding image: offload kind=" << offloadKindToString(Kind)
               << Img << "\n";
      auto *Fver =
          ConstantInt::get(Type::getInt16Ty(C), DeviceImageStructVersion);
      auto *Fknd = ConstantInt::get(Type::getInt8Ty(C), Kind);
      auto *Ffmt = ConstantInt::get(Type::getInt8Ty(C), Img.Fmt);
      auto *Ftgt = addStringToModule(
          Img.Tgt, Twine(OffloadKindTag) + Twine("target.") + Twine(ImgId));
      auto *Foptcompile = addStringToModule(
          Img.CompileOpts,
          Twine(OffloadKindTag) + Twine("opts.compile.") + Twine(ImgId));
      auto *Foptlink = addStringToModule(Img.LinkOpts, Twine(OffloadKindTag) +
                                                           Twine("opts.link.") +
                                                           Twine(ImgId));
      std::pair<Constant *, Constant *> FMnf;

      if (MySymPropReader)
        MySymPropReader->getNextDeviceImageInitializer();

      if (Img.Manif.empty()) {
        // no manifest - zero out the fields
        FMnf = std::make_pair(NullPtr, NullPtr);
      } else {
        Expected<MemoryBuffer *> MnfOrErr = loadFile(Img.Manif);
        if (!MnfOrErr)
          return MnfOrErr.takeError();
        MemoryBuffer *Mnf = *MnfOrErr;
        FMnf = addArrayToModule(
            ArrayRef<char>(Mnf->getBufferStart(), Mnf->getBufferSize()),
            Twine(OffloadKindTag) + Twine(ImgId) + Twine(".manifest"));
      }

      Expected<std::pair<Constant *, Constant *>> PropSets =
          tformSYCLPropertySetRegistryFileToIR(Img.PropsFile);
      if (!PropSets)
        return PropSets.takeError();

      if (Img.File.empty())
        return createStringError(errc::invalid_argument,
                                 "image file name missing");
      Expected<MemoryBuffer *> BinOrErr = loadFile(Img.File);
      if (!BinOrErr)
        return BinOrErr.takeError();
      MemoryBuffer *Bin = *BinOrErr;
      if (Img.File != "-" && Kind == OffloadKind::OpenMP &&
          AddOpenMPOffloadNotes) {
        // Adding ELF notes for STDIN is not supported yet.
        Bin = addELFNotes(Bin, Img.File);
      }
      std::pair<Constant *, Constant *> Fbin;
      if (Triple(Img.Tgt).isNativeCPU()) {
        auto FBinOrErr = addDeclarationsForNativeCPU(Img.EntriesFile);
        if (!FBinOrErr)
          return FBinOrErr.takeError();
        Fbin = *FBinOrErr;
      } else {

        // If '--offload-compress' option is specified and zstd is not
        // available, throw an error.
        if (OffloadCompressDevImgs && !llvm::compression::zstd::isAvailable()) {
          return createStringError(
              inconvertibleErrorCode(),
              "'--offload-compress' is specified but the compiler is "
              "built without zstd support.\n"
              "If you are using a custom DPC++ build, please refer to "
              "https://github.com/intel/llvm/blob/sycl/sycl/doc/"
              "GetStartedGuide.md#build-dpc-toolchain-with-device-image-"
              "compression-support"
              " for more information on how to build with zstd support.");
        }

        // Don't compress if the user explicitly specifies the binary image
        // format or if the image is smaller than OffloadCompressThreshold
        // bytes.
        if (Kind != OffloadKind::SYCL || !OffloadCompressDevImgs ||
            Img.Fmt != BinaryImageFormat::none ||
            !llvm::compression::zstd::isAvailable() ||
            static_cast<int>(Bin->getBufferSize()) < OffloadCompressThreshold) {
          Fbin = addDeviceImageToModule(
              ArrayRef<char>(Bin->getBufferStart(), Bin->getBufferSize()),
              Twine(OffloadKindTag) + Twine(ImgId) + Twine(".data"), Kind,
              Img.Tgt);
        } else {

          // Compress the image using zstd.
          SmallVector<uint8_t, 512> CompressedBuffer;
#if LLVM_ENABLE_EXCEPTIONS
          try {
#endif
            llvm::compression::zstd::compress(
                ArrayRef<unsigned char>(
                    (const unsigned char *)(Bin->getBufferStart()),
                    Bin->getBufferSize()),
                CompressedBuffer, OffloadCompressLevel);
#if LLVM_ENABLE_EXCEPTIONS
          } catch (const std::exception &ex) {
            return createStringError(inconvertibleErrorCode(),
                                     std::string("Failed to compress the device image: \n") +
                                     std::string(ex.what()));
          }
#endif
          if (Verbose)
            errs() << "[Compression] Original image size: "
                   << Bin->getBufferSize() << "\n"
                   << "[Compression] Compressed image size: "
                   << CompressedBuffer.size() << "\n"
                   << "[Compression] Compression level used: "
                   << OffloadCompressLevel << "\n";

          // Add the compressed image to the module.
          Fbin = addDeviceImageToModule(
              ArrayRef<char>((const char *)CompressedBuffer.data(),
                             CompressedBuffer.size()),
              Twine(OffloadKindTag) + Twine(ImgId) + Twine(".data"), Kind,
              Img.Tgt);

          // Change image format to compressed_none.
          Ffmt = ConstantInt::get(Type::getInt8Ty(C),
                                  BinaryImageFormat::compressed_none);
        }
      }

      if (Kind == OffloadKind::SYCL) {
        // For SYCL image offload entries are defined here, by wrapper, so
        // those are created per image
        Expected<std::pair<Constant *, Constant *>> EntriesOrErr =
            addSYCLOffloadEntriesToModule(Img.EntriesFile);
        if (!EntriesOrErr)
          return EntriesOrErr.takeError();
        std::pair<Constant *, Constant *> ImageEntriesPtrs = *EntriesOrErr;
        ImagesInits.push_back(ConstantStruct::get(
            getSyclDeviceImageTy(), Fver, Fknd, Ffmt, Ftgt, Foptcompile,
            Foptlink, FMnf.first, FMnf.second, Fbin.first, Fbin.second,
            ImageEntriesPtrs.first, ImageEntriesPtrs.second,
            PropSets.get().first, PropSets.get().second));
      } else
        ImagesInits.push_back(ConstantStruct::get(
            getDeviceImageTy(), Fbin.first, Fbin.second, EntriesB, EntriesE));

      if (EmitRegFuncs) {
        // Create an object that holds <address, size> pair for the device image
        // and put it into a .tgtimg section. This section can be used for
        // finding and extracting all device images from the fat binary after
        // linking.
        Type *IntPtrTy = M.getDataLayout().getIntPtrType(C);
        auto *ImgInfoArr = ConstantArray::get(
            ArrayType::get(IntPtrTy, 2),
            {ConstantExpr::getPointerCast(Fbin.first, IntPtrTy),
             ConstantInt::get(IntPtrTy, Bin->getBufferSize())});
        auto *ImgInfoVar = new GlobalVariable(
            M, ImgInfoArr->getType(), /*isConstant*/ true,
            GlobalVariable::InternalLinkage, ImgInfoArr,
            Twine(OffloadKindTag) + Twine(ImgId) + Twine(".info"));
        ImgInfoVar->setAlignment(
            MaybeAlign(M.getDataLayout().getTypeStoreSize(IntPtrTy) * 2u));
        ImgInfoVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
        ImgInfoVar->setSection(".tgtimg");

        // Add image info to the used list to force it to be emitted to the
        // object.
        appendToUsed(M, ImgInfoVar);
      }

      ImgId++;
    }

    // Then create images array.
    auto *ImagesData =
        Kind == OffloadKind::SYCL
            ? ConstantArray::get(
                  ArrayType::get(getSyclDeviceImageTy(), ImagesInits.size()),
                  ImagesInits)
            : ConstantArray::get(
                  ArrayType::get(getDeviceImageTy(), ImagesInits.size()),
                  ImagesInits);

    auto *Images =
        new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                           GlobalValue::InternalLinkage, ImagesData,
                           Twine(OffloadKindTag) + "device_images");
    if (Verbose)
      errs() << "  global added: " << Images->getName() << "\n";
    Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *ImagesB = ConstantExpr::getGetElementPtr(Images->getValueType(),
                                                   Images, ZeroZero);

    // And finally create the binary descriptor object.
    auto *DescInit =
        Kind == OffloadKind::SYCL
            ? ConstantStruct::get(
                  getSyclBinDescTy(),
                  ConstantInt::get(Type::getInt16Ty(C), BinDescStructVersion),
                  ConstantInt::get(Type::getInt16Ty(C), ImagesInits.size()),
                  ImagesB, EntriesB, EntriesE)
            : ConstantStruct::get(
                  getBinDescTy(),
                  ConstantInt::get(Type::getInt32Ty(C), ImagesInits.size()),
                  ImagesB, EntriesB, EntriesE);

    GlobalValue::LinkageTypes Lnk = DescriptorName.getNumOccurrences() > 0
                                        ? GlobalValue::ExternalLinkage
                                        : GlobalValue::InternalLinkage;
    auto *Res = new GlobalVariable(
        M, DescInit->getType(), /*isConstant*/ true, Lnk, DescInit,
        Twine(OffloadKindTag) + Twine(DescriptorName));
    if (Verbose)
      errs() << "  global added: " << Res->getName() << "\n";
    return Res;
  }

  void createRegisterFunction(OffloadKind Kind, GlobalVariable *BinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func =
        Function::Create(FuncTy, GlobalValue::InternalLinkage,
                         offloadKindToString(Kind) + ".descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get RegFuncName function declaration.
    auto *RegFuncTy = FunctionType::get(Type::getVoidTy(C), getPtrTy(),
                                        /*isVarArg=*/false);
    FunctionCallee RegFuncC =
        M.getOrInsertFunction(Kind == OffloadKind::SYCL ? "__sycl_register_lib"
                                                        : "__tgt_register_lib",
                              RegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFuncC, BinDesc);
    Builder.CreateRetVoid();

    // Add this function to constructors.
    // Set priority to 1 so that __tgt_register_lib is executed AFTER
    // __tgt_register_requires (we want to know what requirements have been
    // asked for before we load a libomptarget plugin so that by the time the
    // plugin is loaded it can report how many devices there are which can
    // satisfy these requirements).
    appendToGlobalCtors(M, Func, /*Priority*/ 1);
  }

  void createUnregisterFunction(OffloadKind Kind, GlobalVariable *BinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func =
        Function::Create(FuncTy, GlobalValue::InternalLinkage,
                         offloadKindToString(Kind) + ".descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get UnregFuncName function declaration.
    auto *UnRegFuncTy = FunctionType::get(Type::getVoidTy(C), getPtrTy(),
                                          /*isVarArg=*/false);
    FunctionCallee UnRegFuncC = M.getOrInsertFunction(
        Kind == OffloadKind::SYCL ? "__sycl_unregister_lib"
                                  : "__tgt_unregister_lib",
        UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFuncC, BinDesc);
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    // Match priority of __tgt_register_lib
    appendToGlobalDtors(M, Func, /*Priority*/ 1);
  }

public:
  BinaryWrapper(StringRef Target, StringRef ToolName,
                StringRef SymPropBCFiles = "")
      : M("offload.wrapper.object", C), ToolName(ToolName) {

    if (!SymPropBCFiles.empty())
      MySymPropReader =
          std::make_unique<SymPropReader>(SymPropBCFiles, ToolName);

    M.setTargetTriple(Triple(Target));
    // Look for llvm-objcopy in the same directory, from which
    // clang-offload-wrapper is invoked. This helps OpenMP offload
    // LIT tests.

    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *P = (void *)(intptr_t)&Help;
    std::string COWPath = sys::fs::getMainExecutable(ToolName.str().c_str(), P);
    if (!COWPath.empty()) {
      auto COWDir = sys::path::parent_path(COWPath);
      ErrorOr<std::string> ObjcopyPathOrErr =
          sys::findProgramByName("llvm-objcopy", {COWDir});
      if (ObjcopyPathOrErr) {
        ObjcopyPath = *ObjcopyPathOrErr;
        return;
      }

      // Otherwise, look through PATH environment.
    }

    ErrorOr<std::string> ObjcopyPathOrErr =
        sys::findProgramByName("llvm-objcopy");
    if (!ObjcopyPathOrErr) {
      WithColor::warning(errs(), ToolName)
          << "cannot find llvm-objcopy[.exe] in PATH; ELF notes cannot be "
             "added.\n";
      return;
    }

    ObjcopyPath = *ObjcopyPathOrErr;
  }

  BinaryWrapper(const BinaryWrapper &BW) = delete;
  BinaryWrapper &operator=(const BinaryWrapper &BW) = delete;

  ~BinaryWrapper() {
    if (TempFiles.empty())
      return;

    StringRef ToolNameRef(ToolName);
    auto warningOS = [ToolNameRef]() -> raw_ostream & {
      return WithColor::warning(errs(), ToolNameRef);
    };

    for (auto &F : TempFiles) {
      if (SaveTemps) {
        warningOS() << "keeping temporary file " << F << "\n";
        continue;
      }

      auto EC = sys::fs::remove(F, false);
      if (EC)
        warningOS() << "cannot remove temporary file " << F << ": "
                    << EC.message().c_str() << "\n";
    }
  }

  Expected<const Module *> wrap() {
    for (auto &X : Packs) {
      OffloadKind Kind = X.first;
      SameKindPack *Pack = X.second.get();
      Expected<GlobalVariable *> DescOrErr = createBinDesc(Kind, *Pack);
      if (!DescOrErr)
        return DescOrErr.takeError();

      if (EmitRegFuncs) {
        GlobalVariable *Desc = *DescOrErr;
        createRegisterFunction(Kind, Desc);
        createUnregisterFunction(Kind, Desc);
      }
    }
    return &M;
  }
};

  // The whole function body is misaligned just to simplify
  // conflict resolutions with llorg.
  MemoryBuffer *BinaryWrapper::addELFNotes(
      MemoryBuffer *Buf,
      StringRef OriginalFileName) {
    // Cannot add notes, if llvm-objcopy is not available.
    //
    // I did not find a clean way to add a new notes section into an existing
    // ELF file. llvm-objcopy seems to recreate a new ELF from scratch,
    // and we just try to use llvm-objcopy here.
    if (ObjcopyPath.empty())
      return Buf;

    StringRef ToolNameRef(ToolName);

    // Helpers to emit warnings.
    auto warningOS = [ToolNameRef]() -> raw_ostream & {
      return WithColor::warning(errs(), ToolNameRef);
    };
    auto handleErrorAsWarning = [&warningOS](Error E) {
      logAllUnhandledErrors(std::move(E), warningOS());
    };

    Expected<std::unique_ptr<ObjectFile>> BinOrErr =
        ObjectFile::createELFObjectFile(Buf->getMemBufferRef(),
                                        /*InitContent=*/false);
    if (Error E = BinOrErr.takeError()) {
      consumeError(std::move(E));
      // This warning is questionable, but let it be here,
      // assuming that most OpenMP offload models use ELF offload images.
      warningOS() << OriginalFileName
                  << " is not an ELF image, so notes cannot be added to it.\n";
      return Buf;
    }

    // If we fail to add the note section, we just pass through the original
    // ELF image for wrapping. At some point we should enforce the note section
    // and start emitting errors vs warnings.
    endianness Endianness;
    if (isa<ELF64LEObjectFile>(BinOrErr->get()) ||
        isa<ELF32LEObjectFile>(BinOrErr->get())) {
      Endianness = endianness::little;
    } else if (isa<ELF64BEObjectFile>(BinOrErr->get()) ||
               isa<ELF32BEObjectFile>(BinOrErr->get())) {
      Endianness = endianness::big;
    } else {
      warningOS() << OriginalFileName
                  << " is an ELF image of unrecognized format.\n";
      return Buf;
    }

    // Create temporary file for the data of a new SHT_NOTE section.
    // We fill it in with data and then pass to llvm-objcopy invocation
    // for reading.
    Twine NotesFileModel = OriginalFileName + Twine(".elfnotes.%%%%%%%.tmp");
    Expected<sys::fs::TempFile> NotesTemp =
        sys::fs::TempFile::create(NotesFileModel);
    if (Error E = NotesTemp.takeError()) {
      handleErrorAsWarning(createFileError(NotesFileModel, std::move(E)));
      return Buf;
    }
    TempFiles.push_back(NotesTemp->TmpName);

    // Create temporary file for the updated ELF image.
    // This is an empty file that we pass to llvm-objcopy invocation
    // for writing.
    Twine ELFFileModel = OriginalFileName + Twine(".elfwithnotes.%%%%%%%.tmp");
    Expected<sys::fs::TempFile> ELFTemp =
        sys::fs::TempFile::create(ELFFileModel);
    if (Error E = ELFTemp.takeError()) {
      handleErrorAsWarning(createFileError(ELFFileModel, std::move(E)));
      return Buf;
    }
    TempFiles.push_back(ELFTemp->TmpName);

    // Keep the new ELF image file to reserve the name for the future
    // llvm-objcopy invocation.
    std::string ELFTmpFileName = ELFTemp->TmpName;
    if (Error E = ELFTemp->keep(ELFTmpFileName)) {
      handleErrorAsWarning(createFileError(ELFTmpFileName, std::move(E)));
      return Buf;
    }

    // Write notes to the *elfnotes*.tmp file.
    raw_fd_ostream NotesOS(NotesTemp->FD, false);

    struct NoteTy {
      // Note name is a null-terminated "LLVMOMPOFFLOAD".
      std::string Name;
      // Note type defined in llvm/include/llvm/BinaryFormat/ELF.h.
      uint32_t Type = 0;
      // Each note has type-specific associated data.
      std::string Desc;

      NoteTy(std::string &&Name, uint32_t Type, std::string &&Desc)
          : Name(std::move(Name)), Type(Type), Desc(std::move(Desc)) {}
    };

    // So far we emit just three notes.
    SmallVector<NoteTy, 3> Notes;
    // Version of the offload image identifying the structure of the ELF image.
    // Version 1.0 does not have any specific requirements.
    // We may come up with some structure that has to be honored by all
    // offload implementations in future (e.g. to let libomptarget
    // get some information from the offload image).
    Notes.emplace_back("LLVMOMPOFFLOAD", ELF::NT_LLVM_OPENMP_OFFLOAD_VERSION,
                       OPENMP_OFFLOAD_IMAGE_VERSION);
    // This is a producer identification string. We are LLVM!
    Notes.emplace_back("LLVMOMPOFFLOAD", ELF::NT_LLVM_OPENMP_OFFLOAD_PRODUCER,
                       "LLVM");
    // This is a producer version. Use the same format that is used
    // by clang to report the LLVM version.
    Notes.emplace_back("LLVMOMPOFFLOAD",
                       ELF::NT_LLVM_OPENMP_OFFLOAD_PRODUCER_VERSION,
                       LLVM_VERSION_STRING
#ifdef LLVM_REVISION
                       " " LLVM_REVISION
#endif
    );

    // Return the amount of padding required for a blob of N bytes
    // to be aligned to Alignment bytes.
    auto getPadAmount = [](uint32_t N, uint32_t Alignment) -> uint32_t {
      uint32_t Mod = (N % Alignment);
      if (Mod == 0)
        return 0;
      return Alignment - Mod;
    };
    auto emitPadding = [&getPadAmount](raw_ostream &OS, uint32_t Size) {
      for (uint32_t I = 0; I < getPadAmount(Size, 4); ++I)
        OS << '\0';
    };

    // Put notes into the file.
    for (auto &N : Notes) {
      assert(!N.Name.empty() && "We should not create notes with empty names.");
      // Name must be null-terminated.
      if (N.Name.back() != '\0')
        N.Name += '\0';
      uint32_t NameSz = N.Name.size();
      uint32_t DescSz = N.Desc.size();
      // A note starts with three 4-byte values:
      //   NameSz
      //   DescSz
      //   Type
      // These three fields are endian-sensitive.
      support::endian::write<uint32_t>(NotesOS, NameSz, Endianness);
      support::endian::write<uint32_t>(NotesOS, DescSz, Endianness);
      support::endian::write<uint32_t>(NotesOS, N.Type, Endianness);
      // Next, we have a null-terminated Name padded to a 4-byte boundary.
      NotesOS << N.Name;
      emitPadding(NotesOS, NameSz);
      if (DescSz == 0)
        continue;
      // Finally, we have a descriptor, which is an arbitrary flow of bytes.
      NotesOS << N.Desc;
      emitPadding(NotesOS, DescSz);
    }
    NotesOS.flush();

    // Keep the notes file.
    std::string NotesTmpFileName = NotesTemp->TmpName;
    if (Error E = NotesTemp->keep(NotesTmpFileName)) {
      handleErrorAsWarning(createFileError(NotesTmpFileName, std::move(E)));
      return Buf;
    }

    // Run llvm-objcopy like this:
    //   llvm-objcopy --add-section=.note.openmp=<notes-tmp-file-name> \
    //       <orig-file-name> <elf-tmp-file-name>
    //
    // This will add a SHT_NOTE section on top of the original ELF.
    std::vector<StringRef> Args;
    Args.push_back(ObjcopyPath);
    std::string Option("--add-section=.note.openmp=" + NotesTmpFileName);
    Args.push_back(Option);
    Args.push_back("--no-verify-note-sections");
    Args.push_back(OriginalFileName);
    Args.push_back(ELFTmpFileName);
    bool ExecutionFailed = false;
    std::string ErrMsg;
    (void)sys::ExecuteAndWait(ObjcopyPath, Args,
                              /*Env=*/std::nullopt, /*Redirects=*/{},
                              /*SecondsToWait=*/0,
                              /*MemoryLimit=*/0, &ErrMsg, &ExecutionFailed);

    if (ExecutionFailed) {
      warningOS() << ErrMsg << "\n";
      return Buf;
    }

    // Substitute the original ELF with new one.
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(ELFTmpFileName);
    if (!BufOrErr) {
      handleErrorAsWarning(
          createFileError(ELFTmpFileName, BufOrErr.getError()));
      return Buf;
    }

    AutoGcBufs.emplace_back(std::move(*BufOrErr));
    return AutoGcBufs.back().get();
  }

llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                              const BinaryWrapper::Image &Img) {
  Out << "\n{\n";
  Out << "  file     = " << Img.File << "\n";
  Out << "  manifest = " << (Img.Manif.empty() ? "-" : Img.Manif) << "\n";
  Out << "  format   = " << formatToString(Img.Fmt) << "\n";
  Out << "  target   = " << (Img.Tgt.empty() ? "-" : Img.Tgt) << "\n";
  Out << "  compile options  = "
      << (Img.CompileOpts.empty() ? "-" : Img.CompileOpts) << "\n";
  Out << "  link options     = " << (Img.LinkOpts.empty() ? "-" : Img.LinkOpts)
      << "\n";
  Out << "}\n";
  return Out;
}

// Helper class to order elements of multiple cl::list option lists according to
// the sequence they occurred on the command line. Each cl::list defines a
// separate options "class" to identify which class current options belongs to.
// The ID of a class is simply the ordinal of its corresponding cl::list object
// as passed to the constructor. Typical usage:
//  do {
//    ID = ArgSeq.next();
//
//    switch (ID) {
//    case -1: // Done
//      break;
//    case 0: // An option from the cl::list which came first in the constructor
//      (*(ArgSeq.template get<0>())); // get the option value
//      break;
//    case 1: // An option from the cl::list which came second in the
//    constructor
//      (*(ArgSeq.template get<1>())); // get the option value
//      break;
//    ...
//    default:
//      llvm_unreachable("bad option class ID");
//    }
//  } while (ID != -1);
//
template <typename... Tys> class ListArgsSequencer {
private:
  /// The class ID of current option
  int Cur = -1;

  /// Class IDs of all options from all lists. Filled in the constructor.
  /// Can also be seen as a map from command line position to the option class
  /// ID. If there is no option participating in one of the sequenced lists at
  /// given position, then it is mapped to -1 marker value.
  std::unique_ptr<std::vector<int>> OptListIDs;

  using tuple_of_iters_t = std::tuple<typename Tys::iterator...>;

  template <size_t I>
  using iter_t = typename std::tuple_element<I, tuple_of_iters_t>::type;

  /// Tuple of all lists' iterators pointing to "previous" option value -
  /// before latest next() was called
  tuple_of_iters_t Prevs;

  /// Holds "current" iterators - after next()
  tuple_of_iters_t Iters;

public:
  /// The only constructor.
  /// Sz   - total number of options on the command line
  /// Args - the cl::list objects to sequence elements of
  ListArgsSequencer(size_t Sz, Tys &... Args)
      : Prevs(Args.end()...), Iters(Args.begin()...) {
    // make OptListIDs big enough to hold IDs of all options coming from the
    // command line and initialize all IDs to default class -1
    OptListIDs.reset(new std::vector<int>(Sz, -1));
    // map command line positions where sequenced options occur to appropriate
    // class IDs
    addLists<sizeof...(Tys) - 1, 0>(Args...);
  }

  ListArgsSequencer() = delete;

  /// Advances to the next option in the sequence. Returns the option class ID
  /// or -1 when all lists' elements have been iterated over.
  int next() {
    size_t Sz = OptListIDs->size();

    if ((Cur > 0) && (((size_t)Cur) >= Sz))
      return -1;
    while ((((size_t)++Cur) < Sz) && (cur() == -1))
      ;

    if (((size_t)Cur) < Sz)
      inc<sizeof...(Tys) - 1>();
    return ((size_t)Cur) >= Sz ? -1 : cur();
  }

  /// Retrieves the value of current option. ID must match is the option class
  /// returned by next(), otherwise compile error can happen or incorrect option
  /// value will be retrieved.
  template <int ID> decltype(std::get<ID>(Prevs)) get() {
    return std::get<ID>(Prevs);
  }

private:
  int cur() {
    assert(Cur >= 0 && ((size_t)Cur) < OptListIDs->size());
    return (*OptListIDs)[Cur];
  }

  // clang-format off
  template <int MAX, int ID, typename XTy, typename... XTys>
      std::enable_if_t<ID < MAX> addLists(XTy &Arg, XTys &...Args) {
    // clang-format on
    addListImpl<ID>(Arg);
    addLists<MAX, ID + 1>(Args...);
  }

  template <int MAX, int ID, typename XTy>
  std::enable_if_t<ID == MAX> addLists(XTy &Arg) {
    addListImpl<ID>(Arg);
  }

  /// Does the actual sequencing of options found in given list.
  template <int ID, typename T> void addListImpl(T &L) {
    // iterate via all occurrences of an option of given list class
    for (auto It = L.begin(); It != L.end(); It++) {
      // calculate its sequential position in the command line
      unsigned Pos = L.getPosition(It - L.begin());
      assert((*OptListIDs)[Pos] == -1);
      // ... and fill the corresponding spot in the list with the class ID
      (*OptListIDs)[Pos] = ID;
    }
  }

  template <int N> void incImpl() {
    if (cur() == -1)
      return;
    if (N == cur()) {
      std::get<N>(Prevs) = std::get<N>(Iters);
      std::get<N>(Iters)++;
    }
  }

  template <int N> std::enable_if_t<N != 0> inc() {
    incImpl<N>();
    inc<N - 1>();
  }

  template <int N> std::enable_if_t<N == 0> inc() { incImpl<N>(); }
};

} // anonymous namespace

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadWrapperCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-wrapper") << '\n';
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to create a wrapper bitcode for offload target binaries.\n"
      "Takes offload target binaries and optional manifest files as input\n"
      "and produces bitcode file containing target binaries packaged as data\n"
      "and initialization code which registers target binaries in the offload\n"
      "runtime. Manifest files format and contents are not restricted and are\n"
      "a subject of agreement between the device compiler and the native\n"
      "runtime for that device. When present, manifest file name should\n"
      "immediately follow the corresponding device image filename on the\n"
      "command line. Options annotating a device binary have effect on all\n"
      "subsequent input, until redefined.\n"
      "\n"
      "For example:\n"
      "  clang-offload-wrapper                   \\\n"
      "      -host x86_64-pc-linux-gnu           \\\n"
      "      -kind=sycl                          \\\n"
      "        -target=spir64                    \\\n"
      "          -format=spirv                   \\\n"
      "          -compile-opts=-g                \\\n"
      "          -link-opts=-cl-denorms-are-zero \\\n"
      "          -entries=sym.txt                \\\n"
      "          -properties=props.txt           \\\n"
      "          a.spv                           \\\n"
      "          a_mf.txt                        \\\n"
      "        -target=xxx                       \\\n"
      "          -format=native                  \\\n"
      "          -compile-opts=\"\"                \\\n"
      "          -link-opts=\"\"                   \\\n"
      "          -entries=\"\"                     \\\n"
      "          -properties=\"\"                  \\\n"
      "          b.bin                           \\\n"
      "          b_mf.txt                        \\\n"
      "      -kind=openmp                        \\\n"
      "          c.bin\\n"
      "\n"
      "This command generates an x86 wrapper object (.bc) enclosing the\n"
      "following tuples describing a single device binary each:\n"
      "\n"
      "|offload|target|data  |data |manifest|compile|entries|properties|...|\n"
      "|  kind |      |format|     |        |options|       |          |...|\n"
      "|-------|------|------|-----|--------|-------|-------|----------|---|\n"
      "|sycl   |spir64|spirv |a.spv|a_mf.txt|  -g   |sym.txt|props.txt |...|\n"
      "|sycl   |xxx   |native|b.bin|b_mf.txt|       |       |          |...|\n"
      "|openmp |xxx   |native|c.bin|        |       |       |          |...|\n"
      "\n"
      "|...|    link            |\n"
      "|...|    options         |\n"
      "|---|--------------------|\n"
      "|...|-cl-denorms-are-zero|\n"
      "|...|                    |\n"
      "|...|                    |\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }
  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
  };
  if (Target.empty()) {
    Target = sys::getProcessTriple();
    if (Verbose)
      errs() << "warning: -" << Target.ArgStr << " option is omitted, using "
             << "host triple '" << Target << "'\n";
  }
  if (Triple(Target).getArch() == Triple::UnknownArch) {
    reportError(createStringError(
        errc::invalid_argument, "'" + Target + "': unsupported target triple"));
    return 1;
  }
  if (!SymPropBCFiles.empty() && Entries.size()) {
    reportError(createStringError(errc::invalid_argument,
                                  "Entry points cannot be provided by both "
                                  "-sym-prop-bc-files and -entries"));
    return 1;
  }
  if (!SymPropBCFiles.empty() && Properties.size()) {
    reportError(createStringError(errc::invalid_argument,
                                  "Properties cannot be provided by both "
                                  "-sym-prop-bc-files and -properties"));
    return 1;
  }

  // Construct BinaryWrapper::Image instances based on command line args and
  // add them to the wrapper

  BinaryWrapper Wr(Target, argv[0], SymPropBCFiles);
  OffloadKind Knd = OffloadKind::Unknown;
  llvm::StringRef Tgt = "";
  BinaryImageFormat Fmt = BinaryImageFormat::none;
  llvm::StringRef CompileOpts = "";
  llvm::StringRef LinkOpts = "";
  llvm::StringRef EntriesFile = "";
  llvm::StringRef PropsFile = "";
  llvm::SmallVector<llvm::StringRef, 2> CurInputGroup;

  ListArgsSequencer<decltype(Inputs), decltype(Kinds), decltype(Formats),
                    decltype(Targets), decltype(CompileOptions),
                    decltype(LinkOptions), decltype(Entries),
                    decltype(Properties)>
      ArgSeq((size_t)argc, Inputs, Kinds, Formats, Targets, CompileOptions,
             LinkOptions, Entries, Properties);
  int ID = -1;

  do {
    ID = ArgSeq.next();

    // ID != 0 signal that a new image(s) must be added
    if (ID != 0) {
      // create an image instance using current state
      if (CurInputGroup.size() > 2) {
        reportError(
            createStringError(errc::invalid_argument,
                              "too many inputs for a single binary image, "
                              "<binary file> <manifest file>{opt}expected"));
        return 1;
      }
      if (CurInputGroup.size() != 0) {
        if (BatchMode) {
          // transform the batch job (a table of filenames) into a series of
          // 'Wr.addImage' operations for each record in the table
          assert(CurInputGroup.size() == 1 && "1 input in batch mode expected");
          StringRef BatchFile = CurInputGroup[0];
          Expected<std::unique_ptr<util::SimpleTable>> TPtr =
              util::SimpleTable::read(BatchFile);
          if (!TPtr) {
            reportError(TPtr.takeError());
            return 1;
          }
          const util::SimpleTable &T = *TPtr->get();

          // iterate via records
          for (const auto &Row : T.rows()) {
            Wr.addImage(Knd, Row.getCell(COL_CODE),
                        Row.getCell(COL_MANIFEST, ""), Tgt, Fmt, CompileOpts,
                        LinkOpts, Row.getCell(COL_SYM, ""),
                        Row.getCell(COL_PROPS, ""));
          }
        } else {
          if (Knd == OffloadKind::Unknown) {
            reportError(createStringError(errc::invalid_argument,
                                          "offload model not set"));
            return 1;
          }
          StringRef File = CurInputGroup[0];
          StringRef Manif = CurInputGroup.size() > 1 ? CurInputGroup[1] : "";
          Wr.addImage(Knd, File, Manif, Tgt, Fmt, CompileOpts, LinkOpts,
                      EntriesFile, PropsFile);
        }
        CurInputGroup.clear();
      }
    }
    switch (ID) {
    case -1: // Done
      break;
    case 0: // Inputs
      CurInputGroup.push_back(*(ArgSeq.template get<0>()));
      break;
    case 1: // Kinds
      Knd = *(ArgSeq.template get<1>());
      break;
    case 2: // Formats
      Fmt = *(ArgSeq.template get<2>());
      break;
    case 3: // Targets
      Tgt = *(ArgSeq.template get<3>());
      break;
    case 4: // CompileOptions
      CompileOpts = *(ArgSeq.template get<4>());
      break;
    case 5: // LinkOptions
      LinkOpts = *(ArgSeq.template get<5>());
      break;
    case 6: // Entries
      EntriesFile = *(ArgSeq.template get<6>());
      break;
    case 7: // Properties
      PropsFile = *(ArgSeq.template get<7>());
      break;
    default:
      llvm_unreachable("bad option class ID");
    }
  } while (ID != -1);

  // Create the output file to write the resulting bitcode to.
  std::error_code EC;
  ToolOutputFile Out(Output, EC, sys::fs::OF_None);
  if (EC) {
    reportError(createFileError(Output, EC));
    return 1;
  }

  // Create a wrapper for device binaries.
  Expected<const Module *> ModOrErr = Wr.wrap();
  if (!ModOrErr) {
    reportError(ModOrErr.takeError());
    return 1;
  }

#ifndef NDEBUG
  verifyModule(*ModOrErr.get(), &llvm::errs());
#endif

  // And write its bitcode to the file.
  WriteBitcodeToFile(**ModOrErr, Out.os());
  if (Out.os().has_error()) {
    reportError(createFileError(Output, Out.os().error()));
    return 1;
  }

  // Success.
  Out.keep();
  return 0;
}
