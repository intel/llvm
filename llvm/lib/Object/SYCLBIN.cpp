//===- SYCLBIN.cpp - SYCLBIN binary format support --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/SYCLBIN.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

namespace {

template <typename T> void BinaryWriteInteger(raw_ostream &OS, T Val) {
  static_assert(std::is_integral_v<T>);
  OS << StringRef(reinterpret_cast<const char *>(&Val), sizeof(T));
}

template <typename SizeType, typename BlockFunc>
void SizedBlockWrite(raw_ostream &OS, const BlockFunc &F) {
  SmallString<0> BlockData = F();
  BinaryWriteInteger(OS, static_cast<SizeType>(BlockData.size()));
  OS << BlockData;
}

class ConsumerParser {
public:
  ConsumerParser() = default;

  ConsumerParser(const char *Data, size_t Size)
      : Data{Data}, RemainingSize{Size} {}

  // Creates a consumer that "steals" bytes from this based on the read-size
  // promise at the top.
  template <typename ReadSizePromiseT>
  Expected<ConsumerParser> CreateSubConsumer() {
    auto SizeOrError = ConsumeReadSizePromise<uint64_t>();
    if (!SizeOrError)
      return SizeOrError.takeError();
    uint64_t Size = *SizeOrError;

    if (Error EC = ErrorIfSizeUnavailable(Size))
      return EC;
    ConsumerParser NewConsumer{GetCurrentPointer(), Size};
    Move(Size);
    return NewConsumer;
  }

  Error ConsumeCopy(void *Dest, size_t Size) {
    if (Error EC = ErrorIfSizeUnavailable(Size))
      return EC;
    std::memcpy(Dest, Data, Size);
    Move(Size);
    return Error::success();
  }

  template <typename T> Expected<T> ConsumeScalar() {
    T ReadVal{};
    if (Error EC = ConsumeCopy(&ReadVal, sizeof(T)))
      return EC;
    return ReadVal;
  }

  // A common case is where we need to read a size and make sure that size is
  // available after that piece of memory. We call this a "size promise".
  template <typename SizeT> Expected<SizeT> ConsumeReadSizePromise() {
    static_assert(std::is_integral_v<SizeT>);
    Expected<SizeT> ReadSizeOrError = ConsumeScalar<SizeT>();
    if (!ReadSizeOrError)
      return ReadSizeOrError.takeError();
    if (Error EC = ErrorIfSizeUnavailable(*ReadSizeOrError))
      return EC;
    return *ReadSizeOrError;
  }

  template <typename ReadSizePromiseT>
  Expected<llvm::StringRef> ConsumeStringRef() {
    Expected<ReadSizePromiseT> StringSizeOrError =
        ConsumeReadSizePromise<ReadSizePromiseT>();
    if (!StringSizeOrError)
      return StringSizeOrError.takeError();
    llvm::StringRef Result{GetCurrentPointer(), size_t{*StringSizeOrError}};
    Move(*StringSizeOrError);
    return Result;
  }

  template <typename ReadSizePromiseT>
  Expected<SmallString<0>> ConsumeString() {
    Expected<StringRef> StringRefOrError = ConsumeStringRef<ReadSizePromiseT>();
    if (!StringRefOrError)
      return StringRefOrError.takeError();
    return static_cast<SmallString<0>>(*StringRefOrError);
  }

  Expected<SmallVector<SmallString<0>>> ConsumeStringList() {
    ConsumerParser ListConsumer;
    if (Error EC = CreateSubConsumer<uint64_t>().moveInto(ListConsumer))
      return EC;

    Expected<uint32_t> NumStringsOrError =
        ListConsumer.ConsumeScalar<uint32_t>();
    if (!NumStringsOrError)
      return NumStringsOrError.takeError();

    SmallVector<SmallString<0>> Result;
    Result.reserve(*NumStringsOrError);
    for (size_t I = 0; I < *NumStringsOrError; ++I) {
      Expected<SmallString<0>> StringOrError =
          ListConsumer.ConsumeString<uint32_t>();
      if (!StringOrError)
        return StringOrError.takeError();
      Result.emplace_back(*StringOrError);
    }

    return Result;
  }

  size_t GetRemainingSize() const noexcept { return RemainingSize; }

  bool Empty() const noexcept { return GetRemainingSize() == 0; }

private:
  Error ErrorIfSizeUnavailable(size_t Size) const {
    if (RemainingSize < Size)
      return createStringError(inconvertibleErrorCode(),
                               "Incorrect SYCLBIN magic number.");
    return Error::success();
  }

  const char *GetCurrentPointer() const noexcept { return Data; }

  void Move(size_t Size) noexcept {
    Data += Size;
    RemainingSize -= Size;
  }

  const char *Data = nullptr;
  size_t RemainingSize = 0;
};

} // namespace

Expected<SmallString<0>>
SYCLBIN::write(const SmallVector<SYCLBIN::ModuleDesc> &ModuleDescs) {
  // TODO: Merge by properties and kernel names, so overlap can live in the same
  //       abstract module.

  SmallString<0> Data;
  raw_svector_ostream OS(Data);
  OS << StringRef(reinterpret_cast<const char *>(&MagicNumber),
                  sizeof(MagicNumber));
  BinaryWriteInteger<uint32_t>(OS, Version);
  BinaryWriteInteger<uint8_t>(OS, 0); // TODO: Deduce this from arguments.

  {
    SmallString<0> BodyData;
    raw_svector_ostream BodyOS(BodyData);
    for (const ModuleDesc &Desc : ModuleDescs) {
      for (const module_split::SplitModule &SM : Desc.SplitModules) {
        // Write the abstract module metadata block.
        SmallString<0> AbstractModuleData;
        raw_svector_ostream AbstractModuleOS(AbstractModuleData);

        // Write kernel name string list.
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> KernelNamesData;
          uint32_t StringCount = 0;
          {
            raw_svector_ostream KernelNamesOS(KernelNamesData);

            size_t CurrentSymbolPos = 0;
            size_t NextSeperator = 0;
            do {
              NextSeperator = SM.Symbols.find('\n', CurrentSymbolPos);
              size_t CurrentSymbolEnd =
                  (NextSeperator != std::string::npos ? NextSeperator
                                                      : SM.Symbols.size());
              size_t CurrentSymbolSize = CurrentSymbolEnd - CurrentSymbolPos;
              if (CurrentSymbolSize) {
                BinaryWriteInteger(KernelNamesOS,
                                   static_cast<uint32_t>(CurrentSymbolSize));
                KernelNamesOS << StringRef(
                    SM.Symbols.c_str() + CurrentSymbolPos, CurrentSymbolSize);
                ++StringCount;
              }
              CurrentSymbolPos = CurrentSymbolEnd + 1;
            } while (NextSeperator != std::string::npos &&
                     CurrentSymbolPos < SM.Symbols.size());
          }

          SmallString<0> FullKernelNamesData;
          FullKernelNamesData.reserve(sizeof(uint32_t) +
                                      KernelNamesData.size());
          raw_svector_ostream FullKernelNamesOS(FullKernelNamesData);
          BinaryWriteInteger<uint32_t>(FullKernelNamesOS, StringCount);
          FullKernelNamesOS << KernelNamesData;
          return FullKernelNamesData;
        });

        // Write imported symbols string list.
        // TODO: Currently empty, so the list byte size is the size of the
        //       string count.
        BinaryWriteInteger<uint64_t>(AbstractModuleOS, 4);
        BinaryWriteInteger<uint32_t>(AbstractModuleOS, 0);

        // Write exported symbols string list.
        // TODO: Currently empty, so the list byte size is the size of the
        //       string count.
        BinaryWriteInteger<uint64_t>(AbstractModuleOS, 4);
        BinaryWriteInteger<uint32_t>(AbstractModuleOS, 0);

        SizedBlockWrite<uint32_t>(AbstractModuleOS, [&]() {
          SmallString<0> PropertiesData;
          raw_svector_ostream PropsOS(PropertiesData);
          SM.Properties.write(PropsOS);
          return PropertiesData;
        });

        // Read the module data. This is needed no matter what kind of module it
        // is.
        auto BinaryDataOrError =
            llvm::MemoryBuffer::getFileOrSTDIN(SM.ModuleFilePath);
        if (std::error_code EC = BinaryDataOrError.getError())
          return createFileError(SM.ModuleFilePath, EC);
        SmallString<0> RawModuleData =
            StringRef((*BinaryDataOrError)->getBufferStart(),
                      (*BinaryDataOrError)->getBufferSize());

        // IR Modules
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> IRModuleData;
          // If no arch string is present, the module must be IR.
          if (!Desc.ArchString.empty())
            return IRModuleData;
          IRModuleData.reserve(sizeof(IRType) + sizeof(uint64_t) +
                               RawModuleData.size());
          raw_svector_ostream IRModuleOS(IRModuleData);
          BinaryWriteInteger<uint8_t>(IRModuleOS, 0); // TODO: Determine.
          BinaryWriteInteger<uint64_t>(IRModuleOS, RawModuleData.size());
          IRModuleOS << RawModuleData;
          return IRModuleData;
        });

        // Native device code images
        SizedBlockWrite<uint64_t>(AbstractModuleOS, [&]() {
          SmallString<0> NDCIData;
          if (Desc.ArchString.empty())
            return NDCIData;
          NDCIData.reserve(sizeof(uint32_t) + Desc.ArchString.size() +
                           sizeof(uint64_t) + RawModuleData.size());
          raw_svector_ostream NDCIOS(NDCIData);
          BinaryWriteInteger<uint32_t>(NDCIOS, Desc.ArchString.size());
          NDCIOS << Desc.ArchString;
          BinaryWriteInteger<uint64_t>(NDCIOS, RawModuleData.size());
          NDCIOS << RawModuleData;
          return NDCIData;
        });

        // Write abstract modules to body.
        BinaryWriteInteger<uint64_t>(BodyOS, AbstractModuleData.size());
        BodyOS << AbstractModuleData;
      }
    }
    BinaryWriteInteger<uint64_t>(OS, BodyData.size());
    OS << BodyData;
  }

  // Add final padding to required alignment.
  size_t AlignedSize = alignTo(OS.tell(), getAlignment());
  OS.write_zeros(AlignedSize - OS.tell());
  assert(AlignedSize == OS.tell() && "Size mismatch");

  return Data;
}

Expected<std::unique_ptr<SYCLBIN>> SYCLBIN::read(MemoryBufferRef Source) {
  auto Result = std::make_unique<SYCLBIN>(Source);
  ConsumerParser DataConsumer{Source.getBufferStart(), Source.getBufferSize()};

  // Read header.
  if (Error EC =
          DataConsumer.ConsumeCopy(Result->Header.Magic, 4 * sizeof(uint8_t)))
    return EC;
  if (std::memcmp(Result->Header.Magic, MagicNumber, 4) != 0)
    return createStringError(inconvertibleErrorCode(),
                             "Incorrect SYCLBIN magic number.");

  auto VersionOrError = DataConsumer.ConsumeScalar<uint32_t>();
  if (!VersionOrError)
    return VersionOrError.takeError();
  Result->Header.Version = *VersionOrError;

  if (Result->Header.Version > Version)
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported SYCLBIN version " +
                                 std::to_string(Result->Header.Version) + ".");

  auto StateOrError = DataConsumer.ConsumeScalar<BundleState>();
  if (!StateOrError)
    return StateOrError.takeError();
  Result->Header.State = *StateOrError;

  ConsumerParser BodyConsumer;
  if (Error EC =
          DataConsumer.CreateSubConsumer<uint64_t>().moveInto(BodyConsumer))
    return EC;

  while (!BodyConsumer.Empty()) {
    SYCLBIN::AbstractModule &AbstractModule =
        Result->AbstractModules.emplace_back();

    // Abstract module metadata.
    auto MDSizeOrError = BodyConsumer.ConsumeReadSizePromise<uint64_t>();
    if (!MDSizeOrError)
      return MDSizeOrError.takeError();

    if (Error EC = BodyConsumer.ConsumeStringList().moveInto(
            AbstractModule.KernelNames))
      return EC;
    if (Error EC = BodyConsumer.ConsumeStringList().moveInto(
            AbstractModule.ImportedSymbols))
      return EC;
    if (Error EC = BodyConsumer.ConsumeStringList().moveInto(
            AbstractModule.ExportedSymbols))
      return EC;

    {
      // Convert properties to a string to ensure null-terminator.
      SmallString<0> PropsString;
      if (Error EC =
              BodyConsumer.ConsumeString<uint32_t>().moveInto(PropsString))
        return EC;
      auto PropMemBuff =
          llvm::MemoryBuffer::getMemBuffer(llvm::StringRef{PropsString});
      auto ErrorOrProperties =
          llvm::util::PropertySetRegistry::read(PropMemBuff.get());
      if (!ErrorOrProperties)
        return ErrorOrProperties.takeError();
      AbstractModule.Properties = std::move(*ErrorOrProperties);
    }

    // IR modules.
    ConsumerParser IRModuleListConsumer;
    if (Error EC = BodyConsumer.CreateSubConsumer<uint64_t>().moveInto(
            IRModuleListConsumer))
      return EC;

    while (!IRModuleListConsumer.Empty()) {
      SYCLBIN::IRModule &IRModule = AbstractModule.IRModules.emplace_back();

      auto IRTypeOrError = IRModuleListConsumer.ConsumeScalar<IRType>();
      if (!IRTypeOrError)
        return IRTypeOrError.takeError();
      IRModule.Type = *IRTypeOrError;

      auto BinarySizeOrError =
          IRModuleListConsumer.ConsumeReadSizePromise<uint64_t>();
      if (!BinarySizeOrError)
        return BinarySizeOrError.takeError();
      IRModule.RawIRBytes.resize(*BinarySizeOrError);
      if (Error EC = IRModuleListConsumer.ConsumeCopy(
              IRModule.RawIRBytes.data(), *BinarySizeOrError))
        return EC;
    }

    // Native device code images.
    ConsumerParser NDCIListConsumer;
    if (Error EC = BodyConsumer.CreateSubConsumer<uint64_t>().moveInto(
            NDCIListConsumer))
      return EC;

    while (!NDCIListConsumer.Empty()) {
      SYCLBIN::NativeDeviceCodeImage &NDCI =
          AbstractModule.NativeDeviceCodeImages.emplace_back();

      auto ArchStringOrError = NDCIListConsumer.ConsumeString<uint32_t>();
      if (!ArchStringOrError)
        return ArchStringOrError.takeError();
      NDCI.ArchString = *ArchStringOrError;

      auto BinarySizeOrError =
          IRModuleListConsumer.ConsumeReadSizePromise<uint64_t>();
      if (!BinarySizeOrError)
        return BinarySizeOrError.takeError();
      NDCI.RawDeviceCodeImageBytes.resize(*BinarySizeOrError);
      if (Error EC = NDCIListConsumer.ConsumeCopy(
              NDCI.RawDeviceCodeImageBytes.data(), *BinarySizeOrError))
        return EC;
    }
  }

  return std::move(Result);
}

SYCLBIN::SYCLBIN(MemoryBufferRef Source) : Binary(Binary::ID_SYCLBIN, Source) {}
