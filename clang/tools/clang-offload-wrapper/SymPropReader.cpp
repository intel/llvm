#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/WithColor.h"
#include <SymPropReader.h>

namespace {

// Field in the file listing all BC files supplying Symbols and Properties
constexpr char COL_SYM_AND_PROPS[] = "SymAndProps";

// Return a StringRef that refers to the initializer value of the variable with
// name `Name`.
auto getValueAsStringRef(const Module *M, Constant *Name) {
  auto Var = M->getGlobalVariable(Name->getName(), true);
  auto Initializer = Var->getInitializer();
  if (Initializer->isNullValue()) {
    // Return data even if zeroinitializer is in IR.
    // Length of SizeTy is necessary because PropertyValue with Data
    // puts a bitsize variable of type SizeTy at the start of Data.
    static const char zi[sizeof(llvm::util::PropertyValue::SizeTy)] = {0};
    static const StringRef ZeroInitializer{
        zi, sizeof(llvm::util::PropertyValue::SizeTy)};

    return ZeroInitializer;
  }
  auto StringRef = cast<ConstantDataArray>(Initializer)->getRawDataValues();
  return StringRef;
}

auto getInitializerNumElements(const Constant *Initializer) {
  Type *Ty = Initializer->getType();
  auto *ArrTy = dyn_cast<ArrayType>(Ty);
  assert(ArrTy && "Initializer must be of ArrayType");
  return ArrTy->getNumElements();
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

constexpr unsigned int EntriesBeginIndexInTDI{10};
constexpr unsigned int PropertySetBeginIndexInTDI{12};

// struct __tgt_offload_entry {
//   void *addr;
//   char *name;
//   size_t size;
//   int32_t flags;
//   int32_t reserved;
// };

constexpr int nameIndexInTOE{1};

// struct _pi_device_binary_property_set_struct {
//   char *Name;
//   _pi_device_binary_property_struct* PropertiesBegin;
//   _pi_device_binary_property_struct* PropertiesEnd;
// };

constexpr unsigned int NameIndexInPIDBPSS{0};
constexpr unsigned int PropertiesBeginIndexInPIDBPSS{1};

// struct _pi_device_binary_property_struct {
//   char *Name;
//   void *ValAddr;
//   pi_property_type Type;
//   uint64_t ValSize;
// };

constexpr unsigned int NameIndexInPIDBPS{0};
constexpr unsigned int ValAddrIndexInPIDBPS{1};
constexpr unsigned int TypeIndexInPIDBPS{2};
constexpr unsigned int ValSizeIndexInPIDBPS{3};

} // namespace

void SymPropReader::getNextDeviceImageInitializer() {
  if (ImageIndex == ImageCnt) {
    const auto &Row = SymPropTable->rows()[BCFileIndex++];

    // Read a BC File that will provide SYM and Props information
    std::unique_ptr<MemoryBuffer> MB = SymPropsExitOnErr(errorOrToExpected(
        MemoryBuffer::getFileOrSTDIN(Row.getCell(COL_SYM_AND_PROPS))));
    CurrentSymPropsM = SymPropsExitOnErr(
        getOwningLazyBitcodeModule(std::move(MB), SymPropsC,
                                   /*ShouldLazyLoadMetadata=*/true));
    SymPropsExitOnErr(CurrentSymPropsM->materializeAll());
    auto DeviceImagesVar = CurrentSymPropsM->getGlobalVariable(
        ".sycl_offloading.device_images", true);
    DeviceImagesInitializer = DeviceImagesVar->getInitializer();
    ImageIndex = 0;
    ImageCnt = getInitializerNumElements(DeviceImagesInitializer);
  }
  CurrentDeviceImageInitializer =
      DeviceImagesInitializer->getAggregateElement(ImageIndex++);

  auto EntriesName = CurrentDeviceImageInitializer->getAggregateElement(
      EntriesBeginIndexInTDI);
  auto EntriesVar =
      CurrentSymPropsM->getGlobalVariable(EntriesName->getName(), true);
  EntriesInitializer = EntriesVar->getInitializer();
}

uint64_t SymPropReader::getNumEntries() {
  return getInitializerNumElements(EntriesInitializer);
}
StringRef SymPropReader::getEntryName(uint64_t i) {
  Constant *Entry_Initializer = EntriesInitializer->getAggregateElement(i);
  Constant *Entry_Name = Entry_Initializer->getAggregateElement(nameIndexInTOE);
  auto Entry_AsStringRef =
      getValueAsStringRef(CurrentSymPropsM.get(), Entry_Name);
  return (Entry_AsStringRef.rtrim('\0'));
}

std::unique_ptr<llvm::util::PropertySetRegistry>
SymPropReader::getPropRegistry() {

  std::unique_ptr<llvm::util::PropertySetRegistry> PropRegistry;

  Constant *PropertySets_Name =
      CurrentDeviceImageInitializer->getAggregateElement(
          PropertySetBeginIndexInTDI);
  auto PropertySets_Var =
      CurrentSymPropsM->getGlobalVariable(PropertySets_Name->getName(), true);
  auto PropertySets_Initializer = PropertySets_Var->getInitializer();

  PropRegistry = std::make_unique<llvm::util::PropertySetRegistry>();
  for (uint64_t i = 0; i < getInitializerNumElements(PropertySets_Initializer);
       i++) {
    Constant *PropertySet_Initializer =
        PropertySets_Initializer->getAggregateElement(i);
    Constant *PropertySet_Name =
        PropertySet_Initializer->getAggregateElement(NameIndexInPIDBPSS);
    auto PropertySet_Name_AsStringRef =
        getValueAsStringRef(CurrentSymPropsM.get(), PropertySet_Name);
    Constant *Properties_Name = PropertySet_Initializer->getAggregateElement(
        PropertiesBeginIndexInPIDBPSS);

    llvm::util::PropertySet PropSet;

    if (!Properties_Name->isNullValue()) {
      auto Properties_Var =
          CurrentSymPropsM->getGlobalVariable(Properties_Name->getName(), true);
      auto Properties_Initializer = Properties_Var->getInitializer();

      for (uint64_t j = 0;
           j < getInitializerNumElements(Properties_Initializer); j++) {
        Constant *Property_Initializer =
            Properties_Initializer->getAggregateElement(j);
        Constant *Property_Name =
            Property_Initializer->getAggregateElement(NameIndexInPIDBPS);
        Constant *Property_ValAddr =
            Property_Initializer->getAggregateElement(ValAddrIndexInPIDBPS);
        Constant *Property_Type =
            Property_Initializer->getAggregateElement(TypeIndexInPIDBPS);
        Constant *Property_ValSize =
            Property_Initializer->getAggregateElement(ValSizeIndexInPIDBPS);

        auto Property_Name_AsStringRef =
            getValueAsStringRef(CurrentSymPropsM.get(), Property_Name);
        auto Property_Type_AsUInt64 =
            static_cast<ConstantInt *>(Property_Type)->getZExtValue();
        auto Property_ValSize_AsUInt64 =
            static_cast<ConstantInt *>(Property_ValSize)->getZExtValue();

        if (Property_Type_AsUInt64 == llvm::util::PropertyValue::UINT32) {
          PropSet.insert(std::pair(Property_Name_AsStringRef.rtrim('\0'),
                                   Property_ValSize_AsUInt64));
        } else if (Property_Type_AsUInt64 ==
                   llvm::util::PropertyValue::BYTE_ARRAY) {
          auto Data_AsStringRef =
              getValueAsStringRef(CurrentSymPropsM.get(), Property_ValAddr);

          llvm::util::PropertyValue::SizeTy DataBitSize = 0;
          for (size_t I = 0; I < sizeof(llvm::util::PropertyValue::SizeTy); ++I)
            DataBitSize |=
                (llvm::util::PropertyValue::SizeTy)Data_AsStringRef[I]
                << (8 * I);
          llvm::util::PropertyValue PV(
              reinterpret_cast<const unsigned char *>(Data_AsStringRef.data()) +
                  sizeof(llvm::util::PropertyValue::SizeTy),
              DataBitSize);
          PropSet.insert(std::pair(Property_Name_AsStringRef.rtrim('\0'), PV));
        } else {
          llvm_unreachable_internal("unsupported property");
        }
      }
    }
    PropRegistry->add(PropertySet_Name_AsStringRef.rtrim('\0'), PropSet);
  }

  return (PropRegistry);
}
