//===- SYCLOffloadWrapper.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a part of wrapping functionality related to SYCL part.
// All implementation details were inspired by previous implementation in
// clang-offload-wrapper tool.
// OffloadWrapper accepts device images, corresponding entries, properties,
// and additional optional options. Then it wraps this into a LLVM IR
// Module accordingly to the expected format listed in
//  sycl/include/sycl/detail/pi.h
//===----------------------------------------------------------------------===//

#include "SYCLOffloadWrapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <memory>
#include <string>
#include <utility>

using namespace llvm;
using namespace llvm::util;

namespace {

/// Note: Returned values are a part of ABI. If you want to change them
/// then coordinate it with SYCL Runtime.
int8_t binaryImageFormatToInt8(SYCLBinaryImageFormat Format) {
  switch (Format) {
  case SYCLBinaryImageFormat::BIF_None:
    return 0;
  case SYCLBinaryImageFormat::BIF_Native:
    return 1;
  case SYCLBinaryImageFormat::BIF_SPIRV:
    return 2;
  case SYCLBinaryImageFormat::BIF_LLVMBC:
    return 3;
  default:
    llvm_unreachable("unexpected SYCLBinaryImageFormat");
  }
}

/// Wrapper helper class that creates all LLVM IRs wrapping given images.
/// Note: All created structures, "_pi_device_*", "__sycl_*" and "__tgt*" names
/// in this implementation are aligned with "sycl/include/sycl/detail/pi.h".
/// If you want to change anything then you MUST coordinate changes with SYCL
/// Runtime ABI.
struct Wrapper {
  Module &M;
  LLVMContext &C;
  SYCLWrappingOptions Options;

  StructType *SyclPropTy = nullptr;
  StructType *SyclPropSetTy = nullptr;
  StructType *EntryTy = nullptr;
  StructType *SyclDeviceImageTy = nullptr;
  StructType *SyclBinDescTy = nullptr;

  Wrapper(Module &M, const SYCLWrappingOptions &Options)
      : M(M), C(M.getContext()), Options(Options) {

    SyclPropTy = getSyclPropTy();
    SyclPropSetTy = getSyclPropSetTy();
    EntryTy = offloading::getEntryTy(M);
    SyclDeviceImageTy = getSyclDeviceImageTy();
    SyclBinDescTy = getSyclBinDescTy();
  }

  /// Creates structure corresponding to:
  /// \code
  ///  struct _pi_device_binary_property_struct {
  ///    char *Name;
  ///    void *ValAddr;
  ///    uint32_t Type;
  ///    uint64_t ValSize;
  ///  };
  /// \endcode
  StructType *getSyclPropTy() {
    return StructType::create({PointerType::getUnqual(C),
                               PointerType::getUnqual(C), Type::getInt32Ty(C),
                               Type::getInt64Ty(C)},
                              "_pi_device_binary_property_struct");
  }

  /// Creates a structure corresponding to:
  /// \code
  ///  struct _pi_device_binary_property_set_struct {
  ///    char *Name;
  ///    _pi_device_binary_property_struct* PropertiesBegin;
  ///    _pi_device_binary_property_struct* PropertiesEnd;
  ///  };
  /// \endcode
  StructType *getSyclPropSetTy() {
    return StructType::create({PointerType::getUnqual(C),
                               PointerType::getUnqual(C),
                               PointerType::getUnqual(C)},
                              "_pi_device_binary_property_set_struct");
  }

  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerSize()) {
    case 4:
      return Type::getInt32Ty(C);
    case 8:
      return Type::getInt64Ty(C);
    }
    llvm_unreachable("unsupported pointer type size");
  }

  SmallVector<Constant *, 2> getSizetConstPair(size_t First, size_t Second) {
    IntegerType *SizeTTy = getSizeTTy();
    return SmallVector<Constant *, 2>{ConstantInt::get(SizeTTy, First),
                                      ConstantInt::get(SizeTTy, Second)};
  }

  // TODO: Drop Version in favor of the Version in Binary Descriptor.
  // TODO: Drop Manifest fields.
  /// Creates a structure corresponding to:
  /// SYCL specific image descriptor type.
  /// \code
  /// struct __sycl.tgt_device_image {
  ///   // version of this structure - for backward compatibility;
  ///   // all modifications which change order/type/offsets of existing fields
  ///   // should increment the version.
  ///   uint16_t Version;
  ///   // the kind of offload model the image employs.
  ///   uint8_t OffloadKind;
  ///   // format of the image data - SPIRV, LLVMIR bitcode, etc
  ///   uint8_t Format;
  ///   // null-terminated string representation of the device's target
  ///   // architecture
  ///   const char *DeviceTargetSpec;
  ///   // a null-terminated string; target- and compiler-specific options
  ///   // which are suggested to use to "compile" program at runtime
  ///   const char *CompileOptions;
  ///   // a null-terminated string; target- and compiler-specific options
  ///   // which are suggested to use to "link" program at runtime
  ///   const char *LinkOptions;
  ///   // Pointer to the manifest data start
  ///   const unsigned char *ManifestStart;
  ///   // Pointer to the manifest data end
  ///   const unsigned char *ManifestEnd;
  ///   // Pointer to the device binary image start
  ///   void *ImageStart;
  ///   // Pointer to the device binary image end
  ///   void *ImageEnd;
  ///   // the entry table
  ///   __tgt_offload_entry *EntriesBegin;
  ///   __tgt_offload_entry *EntriesEnd;
  ///   _pi_device_binary_property_set_struct *PropertySetBegin;
  ///   _pi_device_binary_property_set_struct *PropertySetEnd;
  /// };
  /// \endcode
  StructType *getSyclDeviceImageTy() {
    return StructType::create(
        {
            Type::getInt16Ty(C),       // Version
            Type::getInt8Ty(C),        // OffloadKind
            Type::getInt8Ty(C),        // Format
            PointerType::getUnqual(C), // DeviceTargetSpec
            PointerType::getUnqual(C), // CompileOptions
            PointerType::getUnqual(C), // LinkOptions
            PointerType::getUnqual(C), // ManifestStart
            PointerType::getUnqual(C), // ManifestEnd
            PointerType::getUnqual(C), // ImageStart
            PointerType::getUnqual(C), // ImageEnd
            PointerType::getUnqual(C), // EntriesBegin
            PointerType::getUnqual(C), // EntriesEnd
            PointerType::getUnqual(C), // PropertySetBegin
            PointerType::getUnqual(C)  // PropertySetEnd
        },
        "__sycl.tgt_device_image");
  }

  /// Creates a structure for SYCL specific binary descriptor type. Corresponds
  /// to:
  ///
  /// \code
  ///  struct __sycl.tgt_bin_desc {
  ///    // version of this structure - for backward compatibility;
  ///    // all modifications which change order/type/offsets of existing fields
  ///    // should increment the version.
  ///    uint16_t Version;
  ///    uint16_t NumDeviceImages;
  ///    __sycl.tgt_device_image *DeviceImages;
  ///    // the offload entry table
  ///    __tgt_offload_entry *HostEntriesBegin;
  ///    __tgt_offload_entry *HostEntriesEnd;
  ///  };
  /// \endcode
  StructType *getSyclBinDescTy() {
    return StructType::create(
        {Type::getInt16Ty(C), Type::getInt16Ty(C), PointerType::getUnqual(C),
         PointerType::getUnqual(C), PointerType::getUnqual(C)},
        "__sycl.tgt_bin_desc");
  }

  Function *addDeclarationForNativeCPU(StringRef Name) {
    FunctionType *NativeCPUFuncTy = FunctionType::get(
        Type::getVoidTy(C),
        {PointerType::getUnqual(C), PointerType::getUnqual(C)}, false);
    FunctionType *NativeCPUBuiltinTy = FunctionType::get(
        PointerType::getUnqual(C), {PointerType::getUnqual(C)}, false);
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

  std::pair<Constant *, Constant *>
  addDeclarationsForNativeCPU(std::string Entries) {
    auto *NullPtr = llvm::ConstantPointerNull::get(PointerType::getUnqual(C));
    if (Entries.empty())
      return {NullPtr, NullPtr};

    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(Entries);
    // the Native CPU PI Plug-in expects the BinaryStart field to point to an
    // array of struct nativecpu_entry {
    //   char *kernelname;
    //   unsigned char *kernel_ptr;
    // };
    StructType *NCPUEntryT = StructType::create(
        {PointerType::getUnqual(C), PointerType::getUnqual(C)},
        "__nativecpu_entry");
    SmallVector<Constant *, 5> NativeCPUEntries;
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI) {
      auto *NewDecl = addDeclarationForNativeCPU(*LI);
      NativeCPUEntries.push_back(ConstantStruct::get(
          NCPUEntryT,
          {addStringToModule(*LI, "__ncpu_function_name"), NewDecl}));
    }

    // Add an empty entry that we use as end iterator
    auto *NativeCPUEndStr =
        addStringToModule("__nativecpu_end", "__ncpu_end_str");
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
                                                 getSizetConstPair(0, 0));
    auto *End = ConstantExpr::getGetElementPtr(
        GVar->getValueType(), GVar,
        getSizetConstPair(0, NativeCPUEntries.size()));
    return std::make_pair(Begin, End);
  }

  /// Adds a global readonly variable that is initialized by given
  /// \p Initializer to the module.
  GlobalVariable *addGlobalArrayVariable(const Twine &Name,
                                         ArrayRef<char> Initializer,
                                         const Twine &Section = "") {
    auto *Arr = ConstantDataArray::get(M.getContext(), Initializer);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    SmallVector<char, 32> NameBuf;
    auto SectionName = Section.toStringRef(NameBuf);
    if (!SectionName.empty())
      Var->setSection(SectionName);
    return Var;
  }

  /// Adds given \p Buf as a global variable into the module.
  /// \returns Pair of pointers that point at the beginning and the end of the
  /// variable.
  std::pair<Constant *, Constant *>
  addArrayToModule(ArrayRef<char> Buf, const Twine &Name,
                   const Twine &Section = "") {
    auto *Var = addGlobalArrayVariable(Name, Buf, Section);
    auto *ImageB = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                  getSizetConstPair(0, 0));
    auto *ImageE = ConstantExpr::getGetElementPtr(
        Var->getValueType(), Var, getSizetConstPair(0, Buf.size()));
    return std::make_pair(ImageB, ImageE);
  }

  /// Adds given \p Data as constant byte array in the module.
  /// \returns Constant pointer to the added data. The pointer type does not
  /// carry size information.
  Constant *addRawDataToModule(ArrayRef<char> Data, const Twine &Name) {
    auto *Var = addGlobalArrayVariable(Name, Data);
    auto *DataPtr = ConstantExpr::getGetElementPtr(Var->getValueType(), Var,
                                                   getSizetConstPair(0, 0));
    return DataPtr;
  }

  /// Creates necessary data objects for the given image and returns a pair
  /// of pointers that point to the beginning and end of the global variable
  /// that contains the image data.
  ///
  /// \returns Pair of pointers that point at the
  /// beginning and at end of the global variable that contains the image data.
  std::pair<Constant *, Constant *>
  addDeviceImageToModule(ArrayRef<char> Buf, const Twine &Name,
                         StringRef TargetTriple) {
    // Create global variable for the image data.
    return addArrayToModule(Buf, Name,
                            TargetTriple.empty() ? "" : TargetTriple);
  }

  /// Creates a global variable of const char* type and creates an
  /// initializer that initializes it with \p Str.
  ///
  /// \returns Link-time constant pointer (constant expr) to that
  /// variable.
  Constant *addStringToModule(StringRef Str, const Twine &Name) {
    auto *Arr = ConstantDataArray::getString(C, Str);
    auto *Var = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    auto *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    return ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
  }

  /// Creates a global variable of array of structs and initializes
  /// it with the given values in \p ArrayData.
  ///
  /// \returns Pair of Constants that point at array content.
  /// If \p ArrayData is empty then a returned pair contains nullptrs.
  std::pair<Constant *, Constant *>
  addStructArrayToModule(ArrayRef<Constant *> ArrayData, Type *ElemTy) {
    if (ArrayData.empty()) {
      auto *PtrTy = ElemTy->getPointerTo();
      auto *NullPtr = Constant::getNullValue(PtrTy);
      return std::make_pair(NullPtr, NullPtr);
    }

    assert(ElemTy == ArrayData[0]->getType() && "elem type mismatch");
    auto *Arr =
        ConstantArray::get(ArrayType::get(ElemTy, ArrayData.size()), ArrayData);
    auto *ArrGlob = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                       GlobalVariable::InternalLinkage, Arr,
                                       "__sycl_offload_prop_sets_arr");
    auto *ArrB = ConstantExpr::getGetElementPtr(
        ArrGlob->getValueType(), ArrGlob, getSizetConstPair(0, 0));
    auto *ArrE =
        ConstantExpr::getGetElementPtr(ArrGlob->getValueType(), ArrGlob,
                                       getSizetConstPair(0, ArrayData.size()));
    return std::pair<Constant *, Constant *>(ArrB, ArrE);
  }

  /// Creates a global variable that is initiazed with the given \p Entries.
  ///
  /// \returns Pair of Constants that point at entries content.
  std::pair<Constant *, Constant *>
  addOffloadEntriesToModule(StringRef Entries) {
    if (Entries.empty()) {
      auto *NullPtr = Constant::getNullValue(PointerType::getUnqual(C));
      return std::pair<Constant *, Constant *>(NullPtr, NullPtr);
    }

    auto *Zero = ConstantInt::get(getSizeTTy(), 0);
    auto *I32Zero = ConstantInt::get(Type::getInt32Ty(C), 0);
    auto *NullPtr = Constant::getNullValue(PointerType::getUnqual(C));

    SmallVector<Constant *> EntriesInits;
    std::unique_ptr<MemoryBuffer> MB = MemoryBuffer::getMemBuffer(Entries);
    for (line_iterator LI(*MB); !LI.is_at_eof(); ++LI)
      EntriesInits.push_back(ConstantStruct::get(
          EntryTy, NullPtr, addStringToModule(*LI, "__sycl_offload_entry_name"),
          Zero, I32Zero, I32Zero));

    auto *Arr = ConstantArray::get(ArrayType::get(EntryTy, EntriesInits.size()),
                                   EntriesInits);
    auto *EntriesGV = new GlobalVariable(M, Arr->getType(), /*isConstant*/ true,
                                         GlobalVariable::InternalLinkage, Arr,
                                         "__sycl_offload_entries_arr");

    auto *EntriesB = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV, getSizetConstPair(0, 0));
    auto *EntriesE = ConstantExpr::getGetElementPtr(
        EntriesGV->getValueType(), EntriesGV,
        getSizetConstPair(0, EntriesInits.size()));
    return std::pair<Constant *, Constant *>(EntriesB, EntriesE);
  }

  /// Creates a global variable that is initialized with the \p PropSet.
  ///
  /// \returns Pair of Constants that point at properties content.
  std::pair<Constant *, Constant *>
  addPropertySetToModule(const PropertySet &PropSet) {
    SmallVector<Constant *> PropInits;
    for (const auto &Prop : PropSet) {
      Constant *PropName = addStringToModule(Prop.first(), "prop");
      Constant *PropValAddr = nullptr;
      Constant *PropType =
          ConstantInt::get(Type::getInt32Ty(C), Prop.second.getType());
      Constant *PropValSize = nullptr;

      switch (Prop.second.getType()) {
      case llvm::util::PropertyValue::UINT32: {
        // for known scalar types ValAddr is null, ValSize keeps the value
        PropValAddr = Constant::getNullValue(PointerType::getUnqual(C));
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
      PropInits.push_back(ConstantStruct::get(SyclPropTy, PropName, PropValAddr,
                                              PropType, PropValSize));
    }
    return addStructArrayToModule(PropInits, SyclPropTy);
  }

  /// Creates a global variable that holds encoded given \p PropRegistry.
  /// In-object representation is demonstated below.
  ///
  /// column is a contiguous area of the wrapper object file;
  /// relative location of columns can be arbitrary
  ///
  /// \code
  ///                             _pi_device_binary_property_struct
  /// _pi_device_binary_property_set_struct   |
  ///                     |                   |
  ///                     v                   v
  /// ...             # ...                # ...
  /// PropSetsBegin--># Name0        +----># Name_00
  /// PropSetsEnd--+  # PropsBegin0--+     # ValAddr_00
  /// ...          |  # PropseEnd0------+  # Type_00
  ///              |  # Name1           |  # ValSize_00
  ///              |  # PropsBegin1---+ |  # ...
  ///              |  # PropseEnd1--+ | |  # Name_0n
  ///              +-># ...         | | |  # ValAddr_0n
  ///                 #             | | |  # Type_0n
  ///                 #             | | |  # ValSize_0n
  ///                 #             | | +-># ...
  ///                 #             | |    # ...
  ///                 #             | +---># Name_10
  ///                 #             |      # ValAddr_10
  ///                 #             |      # Type_10
  ///                 #             |      # ValSize_10
  ///                 #             |      # ...
  ///                 #             |      # Name_1m
  ///                 #             |      # ValAddr_1m
  ///                 #             |      # Type_1m
  ///                 #             |      # ValSize_1m
  ///                 #             +-----># ...
  ///                 #                    #
  /// \endcode
  ///
  /// \returns Pair of pointers to the beginning and end of the property set
  /// array, or a pair of nullptrs in case the properties file wasn't specified.
  std::pair<Constant *, Constant *>
  addPropertySetRegistry(const PropertySetRegistry &PropRegistry) {
    // transform all property sets to IR and get the middle column image into
    // the PropSetsInits
    SmallVector<Constant *> PropSetsInits;
    for (const auto &PropSet : PropRegistry) {
      // create content in the rightmost column and get begin/end pointers
      std::pair<Constant *, Constant *> Props =
          addPropertySetToModule(PropSet.second);
      // get the next the middle column element
      auto *Category = addStringToModule(PropSet.first(), "SYCL_PropSetName");
      PropSetsInits.push_back(ConstantStruct::get(SyclPropSetTy, Category,
                                                  Props.first, Props.second));
    }
    // now get content for the leftmost column - create the top-level
    // PropertySetsBegin/PropertySetsBegin entries and return pointers to them
    return addStructArrayToModule(PropSetsInits, SyclPropSetTy);
  }

  /// Emits a global array that contains \p Address and \P Size. Also add
  /// it into llvm.used to force it to be emitted in the object file.
  void emitRegistrationFunctions(Constant *Address, size_t Size, Twine ImageID,
                                 StringRef OffloadKindTag) {
    Type *IntPtrTy = M.getDataLayout().getIntPtrType(C);
    auto *ImgInfoArr =
        ConstantArray::get(ArrayType::get(IntPtrTy, 2),
                           {ConstantExpr::getPointerCast(Address, IntPtrTy),
                            ConstantInt::get(IntPtrTy, Size)});
    auto *ImgInfoVar = new GlobalVariable(
        M, ImgInfoArr->getType(), true, GlobalVariable::InternalLinkage,
        ImgInfoArr, Twine(OffloadKindTag) + ImageID + ".info");
    ImgInfoVar->setAlignment(
        MaybeAlign(M.getDataLayout().getTypeStoreSize(IntPtrTy) * 2u));
    ImgInfoVar->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    ImgInfoVar->setSection(".tgtimg");

    // Add image info to the used list to force it to be emitted to the
    // object.
    appendToUsed(M, ImgInfoVar);
  }

  Constant *wrapImage(const SYCLImage &Image, Twine ImageID,
                      StringRef OffloadKindTag) {
    auto *NullPtr = Constant::getNullValue(PointerType::getUnqual(C));
    // DeviceImageStructVersion change log:
    // -- version 2: updated to PI 1.2 binary image format
    constexpr uint16_t DeviceImageStructVersion = 2;
    constexpr uint8_t SYCLOffloadKind = 4; // Corresponds to SYCL
    auto *Version =
        ConstantInt::get(Type::getInt16Ty(C), DeviceImageStructVersion);
    auto *Kind = ConstantInt::get(Type::getInt8Ty(C), SYCLOffloadKind);
    auto *Format = ConstantInt::get(Type::getInt8Ty(C),
                                    binaryImageFormatToInt8(Image.Format));
    auto *Target = addStringToModule(Image.Target, Twine(OffloadKindTag) +
                                                       "target." + ImageID);
    auto *CompileOptions =
        addStringToModule(Options.CompileOptions,
                          Twine(OffloadKindTag) + "opts.compile." + ImageID);
    auto *LinkOptions = addStringToModule(
        Options.LinkOptions, Twine(OffloadKindTag) + "opts.link." + ImageID);

    std::pair<Constant *, Constant *> PropSets =
        addPropertySetRegistry(Image.PropertyRegistry);

    std::pair<Constant *, Constant *> Binary;
    if (Image.Target == "native_cpu")
      Binary = addDeclarationsForNativeCPU(Image.Entries);
    else {
      Binary = addDeviceImageToModule(
          Image.Image, Twine(OffloadKindTag) + ImageID + ".data", Image.Target);
    }

    // TODO: Manifests are going to be removed.
    // Note: Manifests are deprecated but corresponding nullptr fields should
    // remain to comply ABI.
    std::pair<Constant *, Constant *> Manifests = {NullPtr, NullPtr};

    // For SYCL image offload entries are defined here, by wrapper, so
    // those are created per image
    std::pair<Constant *, Constant *> ImageEntriesPtrs =
        addOffloadEntriesToModule(Image.Entries);
    Constant *WrappedImage = ConstantStruct::get(
        SyclDeviceImageTy, Version, Kind, Format, Target, CompileOptions,
        LinkOptions, Manifests.first, Manifests.second, Binary.first,
        Binary.second, ImageEntriesPtrs.first, ImageEntriesPtrs.second,
        PropSets.first, PropSets.second);

    if (Options.EmitRegistrationFunctions)
      emitRegistrationFunctions(Binary.first, Image.Image.size(), ImageID,
                                OffloadKindTag);

    return WrappedImage;
  }

  GlobalVariable *combineWrappedImages(ArrayRef<Constant *> WrappedImages,
                                       StringRef OffloadKindTag) {
    auto *ImagesData = ConstantArray::get(
        ArrayType::get(SyclDeviceImageTy, WrappedImages.size()), WrappedImages);
    auto *ImagesGV =
        new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                           GlobalValue::InternalLinkage, ImagesData,
                           Twine(OffloadKindTag) + "device_images");
    ImagesGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *Zero = ConstantInt::get(getSizeTTy(), 0);
    Constant *ZeroZero[] = {Zero, Zero};
    auto *ImagesB = ConstantExpr::getGetElementPtr(ImagesGV->getValueType(),
                                                   ImagesGV, ZeroZero);

    // And finally create the binary descriptor object.
    Constant *EntriesB = Constant::getNullValue(PointerType::getUnqual(C));
    Constant *EntriesE = Constant::getNullValue(PointerType::getUnqual(C));
    static constexpr uint16_t BinDescStructVersion = 1;
    auto *DescInit = ConstantStruct::get(
        SyclBinDescTy,
        ConstantInt::get(Type::getInt16Ty(C), BinDescStructVersion),
        ConstantInt::get(Type::getInt16Ty(C), WrappedImages.size()), ImagesB,
        EntriesB, EntriesE);

    return new GlobalVariable(M, DescInit->getType(), /*isConstant*/ true,
                              GlobalValue::InternalLinkage, DescInit,
                              Twine(OffloadKindTag) + "descriptor");
  }

  /// Creates binary descriptor for the given device images. Binary descriptor
  /// is an object that is passed to the offloading runtime at program startup
  /// and it describes all device images available in the executable or shared
  /// library. It is defined as follows:
  ///
  /// \code
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__start_offloading_entries0;
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__stop_offloading_entries0;
  /// ...
  ///
  /// __attribute__((visibility("hidden")))
  /// extern const char *CompileOptions0 = "...";
  /// ...
  /// __attribute__((visibility("hidden")))
  /// extern const char *LinkOptions0 = "...";
  /// ...
  ///
  /// static const char Image0[] = { <Bufs.front() contents> };
  ///  ...
  /// static const char ImageN[] = { <Bufs.back() contents> };
  ///
  /// static constexpr uint16_t Version = 2;
  /// static constexpr uint16_t OffloadKind = 4; // SYCL
  ///
  /// static const __sycl.tgt_device_image Images[] = {
  ///   {
  ///     Version,                      /*Version*/
  ///     OffloadKind,                  // Kind of offload model.
  ///     Format,                       // format of the image - SPIRV, LLVMIR
  ///                                   // bc, etc
  //      NULL,                         /*DeviceTargetSpec*/
  ///     CompileOptions0,              /*CompileOptions0*/
  ///     LinkOptions0,                 /*LinkOptions0*/
  ///     NULL,                         /*ManifestStart*/
  ///     NULL,                         /*ManifestEnd*/
  ///     Image0,                       /*ImageStart*/
  ///     Image0 + sizeof(Image0),      /*ImageEnd*/
  ///     __start_offloading_entries0,  /*EntriesBegin*/
  ///     __stop_offloading_entries0,   /*EntriesEnd*/
  ///     PropertySetBegin0,            /*EntriesEnd*/
  ///     PropertySetEnd0               /*EntriesEnd*/
  ///   },
  ///   ...
  /// };
  ///
  /// static const __sycl.tgt_bin_desc FatbinDesc = {
  ///   Version,                             /*Version*/
  ///   sizeof(Images) / sizeof(Images[0]),  /*NumDeviceImages*/
  ///   Images,                              /*DeviceImages*/
  ///   __start_offloading_entries,      /*HostEntriesBegin*/
  ///   __stop_offloading_entries        /*HostEntriesEnd*/
  /// };
  /// \endcode
  ///
  /// \returns Global variable that represents FatbinDesc.
  GlobalVariable *createFatbinDesc(SmallVector<SYCLImage> &Images) {
    const char *OffloadKindTag = ".sycl_offloading.";
    SmallVector<Constant *> WrappedImages;
    WrappedImages.reserve(Images.size());
    for (size_t i = 0; i != Images.size(); ++i) {
      WrappedImages.push_back(wrapImage(Images[i], Twine(i), OffloadKindTag));
      Images[i].Image.clear(); // This is just for economy of RAM.
    }

    return combineWrappedImages(WrappedImages, OffloadKindTag);
  }

  void createRegisterFatbinFunction(GlobalVariable *FatbinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  Twine("sycl") + ".descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get RegFuncName function declaration.
    auto *RegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee RegFuncC =
        M.getOrInsertFunction("__sycl_register_lib", RegFuncTy);

    // Construct function body
    IRBuilder Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to constructors.
    appendToGlobalCtors(M, Func, /*Priority*/ 1);
  }

  void createUnregisterFunction(GlobalVariable *FatbinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  "sycl.descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get UnregFuncName function declaration.
    auto *UnRegFuncTy =
        FunctionType::get(Type::getVoidTy(C), PointerType::getUnqual(C),
                          /*isVarArg=*/false);
    FunctionCallee UnRegFuncC =
        M.getOrInsertFunction("__sycl_unregister_lib", UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFuncC, FatbinDesc);
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    appendToGlobalDtors(M, Func, /*Priority*/ 1);
  }
}; // end of Wrapper

} // anonymous namespace

Error wrapSYCLBinaries(llvm::Module &M, SmallVector<SYCLImage> &Images,
                       SYCLWrappingOptions Options) {
  Wrapper W(M, Options);
  GlobalVariable *Desc = W.createFatbinDesc(Images);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No binary descriptors created.");

  W.createRegisterFatbinFunction(Desc);
  W.createUnregisterFunction(Desc);
  return Error::success();
}
