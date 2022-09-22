#pragma once

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
  class Module;
  class OpaqueType;
  class StructType;
  class Type;
} // namespace llvm

namespace SPIRV {

class SPIRVOpaqueType {
  std::string Name;
  SPIRVOpaqueType(StringRef Name) : Name(Name) {};

public:
  SPIRVOpaqueType(const SPIRVOpaqueType &) = default;

  llvm::OpaqueType *getAsOpaqueType(llvm::Module &M) const;
  llvm::StructType *getAsSPIRVType(llvm::Module &M) const;
  StringRef getSPIRVName() const { return Name; }

  static llvm::Optional<SPIRVOpaqueType> createFromType(llvm::Type *Ty);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
      const SPIRVOpaqueType &Ty) {
    return OS << "opaque(" << Ty.Name << ")";
  };
};
} // namespace SPIRV
