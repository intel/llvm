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

#include <compiler/utils/mangling.h>
#include <compiler/utils/target_extension_types.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <cstring>
#include <optional>

namespace compiler {
namespace utils {
using namespace llvm;

NameMangler::NameMangler(LLVMContext *context) : Context(context) {}

std::string NameMangler::mangleName(StringRef Name, ArrayRef<Type *> Tys,
                                    ArrayRef<TypeQualifiers> Quals) {
  std::string MangledName;
  raw_string_ostream O(MangledName);
  O << "_Z" << Name.size() << Name;
  for (unsigned i = 0; i < Tys.size(); i++) {
    const ArrayRef<Type *> PrevTys = Tys.slice(0, i);
    const ArrayRef<TypeQualifiers> PrevQuals = Quals.slice(0, i);
    if (!mangleType(O, Tys[i], Quals[i], PrevTys, PrevQuals)) {
      return std::string();
    }
  }
  O.flush();
  return MangledName;
}

StringRef NameMangler::demangleName(
    StringRef Name, SmallVectorImpl<llvm::Type *> &Types,
    SmallVectorImpl<llvm::Type *> &PointerElementTypes,
    SmallVectorImpl<TypeQualifiers> &Quals) {
  // Parse the name part.
  Lexer L(Name);
  Name = demangleName(L);
  if (Name.empty()) {
    return StringRef{};
  }

  // Parse the argument part.
  while (L.Left() > 0) {
    Type *ArgTy = nullptr;
    Type *ArgEltTy = nullptr;
    TypeQualifiers ArgQuals;
    if (!demangleType(L, ArgTy, &ArgEltTy, ArgQuals, Types, Quals)) {
      return StringRef{};
    }
    Types.push_back(ArgTy);
    PointerElementTypes.push_back(ArgEltTy);
    Quals.push_back(ArgQuals);
  }
  return Name;
}

StringRef NameMangler::demangleName(StringRef Name,
                                    SmallVectorImpl<llvm::Type *> &Types,
                                    SmallVectorImpl<TypeQualifiers> &Quals) {
  SmallVector<llvm::Type *, 4> EltTys;
  return demangleName(Name, Types, EltTys, Quals);
}

StringRef NameMangler::demangleName(StringRef Name) {
  Lexer L(Name);
  StringRef DemangledName = demangleName(L);
  if (!DemangledName.empty()) {
    return DemangledName;
  }
  return Name;
}

int NameMangler::resolveSubstitution(unsigned SubID,
                                     SmallVectorImpl<Type *> &Tys,
                                     SmallVectorImpl<TypeQualifiers> &Quals) {
  unsigned CurrentSubID = 0;
  int ResolvedID = -1;
  for (unsigned i = 0; i < Tys.size(); i++) {
    // Determine whether the type is a builtin or not.
    // Builtin types cannot be substituted.
    Type *Ty = Tys[i];
    TypeQualifiers &TyQuals = Quals[i];
    if (isTypeBuiltin(Ty, TyQuals)) {
      continue;
    }
    if (CurrentSubID == SubID) {
      ResolvedID = (int)i;
      break;
    }
    CurrentSubID++;
  }
  return ResolvedID;
}

bool NameMangler::emitSubstitution(raw_ostream &O, Type *Ty,
                                   TypeQualifiers Quals,
                                   ArrayRef<Type *> PrevTys,
                                   ArrayRef<TypeQualifiers> PrevQuals) {
  if (isTypeBuiltin(Ty, Quals)) {
    return false;
  }

  // Look for a previously-mangled non-builtin type we could use as a
  // substitution.
  int SubstitutionID = -1;
  bool FoundMatch = false;
  for (unsigned j = 0; j < PrevTys.size(); j++) {
    Type *PrevTy = PrevTys[j];
    TypeQualifiers PrevQual = PrevQuals[j];
    if (!isTypeBuiltin(PrevTy, PrevQual)) {
      SubstitutionID++;
      if ((PrevTy == Ty) && (PrevQual == Quals)) {
        FoundMatch = true;
        break;
      }
    }
  }
  if (!FoundMatch) {
    return false;
  }

  // Found a match, emit the substitution.
  O << "S";
  if (SubstitutionID > 0) {
    O << SubstitutionID;
  }
  O << "_";
  return true;
}

bool NameMangler::isTypeBuiltin(Type *Ty, TypeQualifiers &Quals) {
  (void)Quals;
  switch (Ty->getTypeID()) {
    default:
    case Type::StructTyID:
    case Type::ArrayTyID:
    case Type::PointerTyID:
    case Type::FixedVectorTyID:
      return false;
    case Type::VoidTyID:
    case Type::HalfTyID:
    case Type::FloatTyID:
    case Type::DoubleTyID:
    case Type::IntegerTyID:
      return true;
  }
}

const char *NameMangler::mangleSimpleType(Type *Ty, TypeQualifier Qual) {
  const bool IsSigned = (Qual & eTypeQualSignedInt);
  switch (Ty->getTypeID()) {
    default:
      break;
    case Type::VoidTyID:
      return "v";
    case Type::HalfTyID:
      return "Dh";
    case Type::FloatTyID:
      return "f";
    case Type::DoubleTyID:
      return "d";
    case Type::IntegerTyID:
      switch (cast<IntegerType>(Ty)->getBitWidth()) {
        default:
          break;
        case 1:
          return "b";  // bool
        case 8:
          return IsSigned ? "c" : "h";
        case 16:
          return IsSigned ? "s" : "t";
        case 32:
          return IsSigned ? "i" : "j";
        case 64:
          return IsSigned ? "l" : "m";
      }
  }
  return nullptr;
}

bool NameMangler::mangleType(raw_ostream &O, Type *Ty, TypeQualifiers Qual) {
  return mangleType(O, Ty, Qual, ArrayRef<Type *>(),
                    ArrayRef<TypeQualifiers>());
}

static void manglePointerQuals(raw_ostream &O, TypeQualifier Qual,
                               unsigned AddressSpace) {
  if (Qual & eTypeQualPointerRestrict) {
    O << 'r';
  }
  if (Qual & eTypeQualPointerVolatile) {
    O << 'V';
  }
  if (Qual & eTypeQualPointerConst) {
    O << 'K';
  }
  if (AddressSpace > 0) {
    O << "U3AS" << AddressSpace;
  }
}

bool NameMangler::mangleType(raw_ostream &O, Type *Ty, TypeQualifiers Quals,
                             ArrayRef<Type *> PrevTys,
                             ArrayRef<TypeQualifiers> PrevQuals) {
  if (emitSubstitution(O, Ty, Quals, PrevTys, PrevQuals)) {
    return true;
  }

  const TypeQualifier Qual = Quals.pop_front();
  if (const char *SimpleName = mangleSimpleType(Ty, Qual)) {
    O << SimpleName;
    return true;
  } else if (isa<ScalableVectorType>(Ty)) {
    std::string tmp;
    raw_string_ostream Otmp(tmp);
    auto *VecTy = cast<llvm::VectorType>(Ty);
    Otmp << "nxv"
         << multi_llvm::getVectorElementCount(VecTy).getKnownMinValue();
    if (!mangleType(Otmp, VecTy->getElementType(), Quals, PrevTys, PrevQuals)) {
      return false;
    }
    O << "u" << tmp.size() << tmp;
    return true;
  } else if (Ty->isVectorTy()) {
    auto *VecTy = cast<FixedVectorType>(Ty);
    O << "Dv" << VecTy->getNumElements() << "_";
    return mangleType(O, VecTy->getElementType(), Quals, PrevTys, PrevQuals);
  } else if (Ty->isPointerTy()) {
    PointerType *PtrTy = cast<PointerType>(Ty);
    const unsigned AddressSpace = PtrTy->getAddressSpace();
#if LLVM_VERSION_LESS(17, 0)
    assert(PtrTy->isOpaque() && "No support for typed pointers past LLVM 15");
#endif
    O << "u3ptr";
    manglePointerQuals(O, Qual, AddressSpace);
    return true;
#if LLVM_VERSION_GREATER_EQUAL(17, 0)
  } else if (Ty->isTargetExtTy()) {
    if (auto Name = mangleBuiltinType(Ty)) {
      O << *Name;
      return true;
    }
    return false;
#endif
  } else {
    return false;
  }
}

bool NameMangler::demangleSimpleType(Lexer &L, Type *&Ty, TypeQualifier &Qual) {
  const int c = L.Current();
  Ty = nullptr;
  Qual = eTypeQualNone;
  if ((c < 0) || !Context) {
    return false;
  }

  switch (c) {
    default:
      return false;
    case 'v':
      Ty = llvm::Type::getVoidTy(*Context);
      break;
    case 'D':
      if (!L.Consume("Dh")) {
        return false;
      }
      Ty = llvm::Type::getHalfTy(*Context);
      return true;
    case 'f':
      Ty = llvm::Type::getFloatTy(*Context);
      break;
    case 'd':
      Ty = llvm::Type::getDoubleTy(*Context);
      break;
    case 'b':
      Ty = llvm::Type::getInt1Ty(*Context);
      break;
    case 'c':
    case 'h':
      Ty = llvm::Type::getInt8Ty(*Context);
      if (c == 'c') {
        Qual = eTypeQualSignedInt;
      }
      break;
    case 's':
    case 't':
      Ty = llvm::Type::getInt16Ty(*Context);
      if (c == 's') {
        Qual = eTypeQualSignedInt;
      }
      break;
    case 'i':
    case 'j':
      Ty = llvm::Type::getInt32Ty(*Context);
      if (c == 'i') {
        Qual = eTypeQualSignedInt;
      }
      break;
    case 'l':
    case 'm':
      Ty = llvm::Type::getInt64Ty(*Context);
      if (c == 'l') {
        Qual = eTypeQualSignedInt;
      }
      break;
  }
  L.Consume();
  return true;
}

std::optional<std::string> NameMangler::mangleBuiltinType(Type *Ty) {
  // With opaque pointers, before LLVM 17 we can't actually mangle OpenCL
  // builtin types because our APIs don't expose the ability to mangle a pointer
  // based on its element type.
  // This is never a problem in the compiler as we don't generate such functions
  // on the fly, but it is a weakness in the API. We could fix this, or wait it
  // out until LLVM 17 becomes the minimum version, at which point target
  // extension types save the day.
#if LLVM_VERSION_LESS(17, 0)
  (void)Ty;
  return nullptr;
#else
  auto *const TgtTy = cast<TargetExtType>(Ty);
  const StringRef Name = TgtTy->getName();

  if (Name == "spirv.Event") {
    return "9ocl_event";
  }

  if (Name == "spirv.Sampler") {
    return "11ocl_sampler";
  }

  if (Name != "spirv.Image") {
    // FIXME: Some types don't have official target extension types.
    // "opencl.clk_event_t" -> "12ocl_clkevent"
    // "opencl.queue_t" -> "9ocl_queue"
    // "opencl.ndrange_t" -> "11ocl_ndrange"
    // "opencl.reserve_id_t" -> "13ocl_reserveid"
    return std::nullopt;
  }

  auto Dim = TgtTy->getIntParameter(tgtext::ImageTyDimensionalityIdx);
  auto Depth = TgtTy->getIntParameter(tgtext::ImageTyDepthIdx);
  auto Arrayed = TgtTy->getIntParameter(tgtext::ImageTyArrayedIdx);
  auto MS = TgtTy->getIntParameter(tgtext::ImageTyMSIdx);

  std::string MangledName = "ocl_image";

  switch (Dim) {
    default:
      return std::nullopt;
    case tgtext::ImageDim1D:
      MangledName += "1d";
      break;
    case tgtext::ImageDim2D:
      MangledName += "2d";
      break;
    case tgtext::ImageDim3D:
      MangledName += "3d";
      break;
    case tgtext::ImageDimBuffer:
      MangledName += "1dbuffer";
      break;
  }

  if (Arrayed == tgtext::ImageArrayed) {
    MangledName += "array";
  }

  if (MS == tgtext::ImageMSMultiSampled) {
    MangledName += "msaa";
  }

  if (Depth == tgtext::ImageDepth) {
    MangledName += "depth";
  }

  return std::to_string(MangledName.size()) + MangledName;
#endif
}

bool NameMangler::demangleOpenCLBuiltinType(Lexer &L, Type *&Ty) {
  if (L.Consume("12memory_scope") || L.Consume("12memory_order")) {
    Ty = IntegerType::getInt32Ty(*Context);
    return true;
  }

#if LLVM_VERSION_GREATER_EQUAL(17, 0)
  if (auto *TargetExtTy = [this, &L]() -> Type * {
        if (L.Consume("11ocl_image1d")) {
          return compiler::utils::tgtext::getImage1DTy(*Context);
        } else if (L.Consume("16ocl_image1darray")) {
          return compiler::utils::tgtext::getImage1DArrayTy(*Context);
        } else if (L.Consume("17ocl_image1dbuffer")) {
          return compiler::utils::tgtext::getImage1DBufferTy(*Context);
        } else if (L.Consume("11ocl_image2d")) {
          return compiler::utils::tgtext::getImage2DTy(*Context);
        } else if (L.Consume("16ocl_image2darray")) {
          return compiler::utils::tgtext::getImage2DArrayTy(*Context);
        } else if (L.Consume("16ocl_image2ddepth")) {
          return compiler::utils::tgtext::getImage2DTy(*Context, /*Depth*/ true,
                                                       /*MS*/ false);
        } else if (L.Consume("21ocl_image2darraydepth")) {
          return compiler::utils::tgtext::getImage2DArrayTy(*Context);
        } else if (L.Consume("15ocl_image2dmsaa")) {
          return compiler::utils::tgtext::getImage2DTy(
              *Context, /*Depth*/ false, /*MS*/ true);
        } else if (L.Consume("20ocl_image2darraymsaa")) {
          return compiler::utils::tgtext::getImage2DArrayTy(
              *Context, /*Depth*/ false, /*MS*/ true);
        } else if (L.Consume("20ocl_image2dmsaadepth")) {
          return compiler::utils::tgtext::getImage2DTy(*Context, /*Depth*/ true,
                                                       /*MS*/ true);
        } else if (L.Consume("35ocl_image2darraymsaadepth")) {
          return compiler::utils::tgtext::getImage2DArrayTy(
              *Context, /*Depth*/ true, /*MS*/ true);
        } else if (L.Consume("11ocl_image3d")) {
          return compiler::utils::tgtext::getImage3DTy(*Context);
        } else if (L.Consume("11ocl_sampler")) {
          return compiler::utils::tgtext::getSamplerTy(*Context);
        } else if (L.Consume("9ocl_event")) {
          return compiler::utils::tgtext::getEventTy(*Context);
        }
        return nullptr;
      }()) {
    Ty = TargetExtTy;
    return true;
  }
#endif

  StringRef Name;
  //
  // TODO: Avoid hard coded name. See redmine issue #8656 please.
  //
  if (L.Consume("11ocl_image1d")) {
    Name = "opencl.image1d_t";
  } else if (L.Consume("16ocl_image1darray")) {
    Name = "opencl.image1d_array_t";
  } else if (L.Consume("17ocl_image1dbuffer")) {
    Name = "opencl.image1d_buffer_t";
  } else if (L.Consume("11ocl_image2d")) {
    Name = "opencl.image2d_t";
  } else if (L.Consume("16ocl_image2darray")) {
    Name = "opencl.image2d_array_t";
  } else if (L.Consume("16ocl_image2ddepth")) {
    Name = "opencl.image2d_depth_t";
  } else if (L.Consume("21ocl_image2darraydepth")) {
    Name = "opencl.image2d_array_depth_t";
  } else if (L.Consume("15ocl_image2dmsaa")) {
    Name = "opencl.image2d_msaa_t";
  } else if (L.Consume("20ocl_image2darraymsaa")) {
    Name = "opencl.image2d_array_msaa_t";
  } else if (L.Consume("20ocl_image2dmsaadepth")) {
    Name = "opencl.image2d_msaa_depth_t";
  } else if (L.Consume("35ocl_image2darraymsaadepth")) {
    Name = "opencl.image2d_array_msaa_depth_t";
  } else if (L.Consume("11ocl_image3d")) {
    Name = "opencl.image3d_t";
  } else if (L.Consume("11ocl_sampler")) {
    Name = "opencl_sampler_t";
  } else if (L.Consume("9ocl_event")) {
    Name = "opencl.event_t";
  } else if (L.Consume("12ocl_clkevent")) {
    Name = "opencl.clk_event_t";
  } else if (L.Consume("9ocl_queue")) {
    Name = "opencl.queue_t";
  } else if (L.Consume("11ocl_ndrange")) {
    Name = "opencl.ndrange_t";
  } else if (L.Consume("13ocl_reserveid")) {
    Name = "opencl.reserve_id_t";
  } else {
    return false;
  }

  if (auto *const OpenCLType =
          llvm::StructType::getTypeByName(*Context, Name)) {
    Ty = OpenCLType;
  } else {
    Ty = llvm::StructType::create(*Context, Name);
  }

  return true;
}

struct PointerASQuals {
  unsigned AS;
  TypeQualifier Qual;
};

static std::optional<PointerASQuals> demanglePointerQuals(Lexer &L) {
  TypeQualifier PointerQual = eTypeQualNone;

  // Parse the optional pointer qualifier.
  if (L.Current() < 0) {
    return std::nullopt;
  }

  // Parse the optional address space qualifier.
  bool DemangledAS = false;
  unsigned AddressSpace = 0;

  if (L.Consume("U3AS")) {
    if (!L.ConsumeInteger(AddressSpace)) {
      return std::nullopt;
    }
    DemangledAS = true;
  }

  switch (L.Current()) {
    default:
      break;
    case 'K':
      PointerQual = eTypeQualPointerConst;
      L.Consume();
      break;
    case 'r':
      PointerQual = eTypeQualPointerRestrict;
      L.Consume();
      break;
    case 'V':
      PointerQual = eTypeQualPointerVolatile;
      L.Consume();
      break;
  }

  if (!DemangledAS && L.Consume("U3AS") && !L.ConsumeInteger(AddressSpace)) {
    return std::nullopt;
  }

  return PointerASQuals{AddressSpace, PointerQual};
}

bool NameMangler::demangleType(Lexer &L, Type *&Ty, Type **PointerEltTy,
                               TypeQualifiers &Quals,
                               SmallVectorImpl<llvm::Type *> &CtxTypes,
                               SmallVectorImpl<TypeQualifiers> &CtxQuals) {
  Ty = nullptr;
  if (L.Left() < 1) {
    return false;
  }

  // Assume the element type is null, and set it if we find a pointer.
  if (PointerEltTy) {
    *PointerEltTy = nullptr;
  }

  // Match vector types.
  if (L.Consume("Dv")) {
    const TypeQualifier VectorQual = eTypeQualNone;
    unsigned NumElements = 0;
    Quals.push_back(VectorQual);
    if (!L.ConsumeInteger(NumElements) || !L.Consume("_")) {
      return false;
    }

    // Parse the vector element type.
    Type *ElementType = nullptr;
    if (!demangleType(L, ElementType, nullptr, Quals, CtxTypes, CtxQuals)) {
      return false;
    }
    Ty = FixedVectorType::get(ElementType, NumElements);
    return true;
  }

  // Match opaque pointer types
  if (L.Consume("u3ptr")) {
    const auto QualsAS = demanglePointerQuals(L);
    if (!QualsAS) {
      return false;
    }
    Quals.push_back(QualsAS->Qual);
    return PointerType::get(nullptr, QualsAS->AS);
  }

  // Match scalable vector types.
  if (L.Consume("u")) {
    unsigned TypeNameLength = 0;
    if (!L.ConsumeInteger(TypeNameLength) || !L.Consume("nxv")) {
      return false;
    }
    if (TypeNameLength > L.Left()) {
      return false;
    }
    const TypeQualifier VectorQual = eTypeQualNone;
    unsigned NumElements = 0;
    Quals.push_back(VectorQual);
    if (!L.ConsumeInteger(NumElements)) {
      return false;
    }

    // Parse the vector element type.
    Type *ElementType = nullptr;
    if (!demangleType(L, ElementType, nullptr, Quals, CtxTypes, CtxQuals)) {
      return false;
    }
    Ty = llvm::VectorType::get(ElementType,
                               ElementCount::getScalable(NumElements));
    return true;
  }

  // Match pointer types.
  if (L.Consume("P")) {
    const auto QualsAS = demanglePointerQuals(L);
    if (!QualsAS) {
      return false;
    }

    Quals.push_back(QualsAS->Qual);

    // Parse the element type.
    Type *ElementType = nullptr;
    if (!demangleType(L, ElementType, nullptr, Quals, CtxTypes, CtxQuals)) {
      return false;
    }
    assert(ElementType);
    if (PointerEltTy) {
      *PointerEltTy = ElementType;
    }
    if (ElementType->isVoidTy()) {
      Ty = llvm::PointerType::get(Type::getInt8Ty(*Context), QualsAS->AS);
    } else {
      Ty = llvm::PointerType::get(ElementType, QualsAS->AS);
    }
    return true;
  }

  // Match simple types.
  TypeQualifier SimpleQual = eTypeQualNone;
  if (demangleSimpleType(L, Ty, SimpleQual)) {
    Quals.push_back(SimpleQual);
    return true;
  }

  // Handle substitutions.
  if (L.Consume("S")) {
    unsigned SubID = 0;
    if (L.ConsumeInteger(SubID)) {
      SubID++;
    }
    if (!L.Consume("_")) {
      return false;
    }

    // Resolve it, using a previous type and qualifier.
    const int entryIndex = resolveSubstitution(SubID, CtxTypes, CtxQuals);
    if ((entryIndex < 0) || ((unsigned)entryIndex >= CtxTypes.size())) {
      return false;
    }
    Ty = CtxTypes[entryIndex];
    Quals.push_back(CtxQuals[entryIndex]);
    return true;
  }

  if (demangleOpenCLBuiltinType(L, Ty)) {
    return true;
  }

  return false;
}

StringRef NameMangler::demangleName(Lexer &L) {
  unsigned NameLength = 0;
  if (!L.Consume("_Z")) {
    return StringRef();
  } else if (!L.ConsumeInteger(NameLength)) {
    return StringRef();
  } else if (NameLength > L.Left()) {
    return StringRef();
  }
  StringRef Name = L.TextLeft().substr(0, NameLength);
  L.Consume(NameLength);
  return Name;
}

////////////////////////////////////////////////////////////////////////////////

TypeQualifiers::TypeQualifiers() : storage_(0) {}

TypeQualifiers::TypeQualifiers(TypeQualifier Qual) : storage_(0) {
  push_back(Qual);
}

TypeQualifiers::TypeQualifiers(TypeQualifier Qual1, TypeQualifier Qual2)
    : storage_(0) {
  push_back(Qual1);
  push_back(Qual2);
}

TypeQualifiers::TypeQualifiers(unsigned Qual) : storage_(0) { push_back(Qual); }

TypeQualifiers::TypeQualifiers(unsigned Qual1, unsigned Qual2) : storage_(0) {
  push_back(Qual1);
  push_back(Qual2);
}

TypeQualifiers::StorageT TypeQualifiers::getCount() const {
  const StorageT Mask = ((1 << NumCountBits) - 1);
  return storage_ & Mask;
}

void TypeQualifiers::setCount(StorageT NewCount) {
  const StorageT Mask = ((1 << NumCountBits) - 1);
  // Clear the old count.
  storage_ &= ~Mask;
  // Set the new count.
  storage_ |= ((NewCount << 0) & Mask);
}

TypeQualifier TypeQualifiers::front() const {
  const StorageT Size = getCount();
  if (Size == 0) {
    return eTypeQualNone;
  }
  const unsigned Mask = ((1 << NumQualBits) - 1);
  const unsigned Field = (storage_ >> NumCountBits) & Mask;
  return (TypeQualifier)Field;
}

TypeQualifier TypeQualifiers::pop_front() {
  const TypeQualifier Qual = front();
  const StorageT Size = getCount();
  if (Size > 0) {
    // Pop the field bits.
    storage_ >>= NumQualBits;
    // Set the new count, since the old one was overwritten.
    setCount(Size - 1);
  }
  return Qual;
}

TypeQualifier TypeQualifiers::at(unsigned Idx) const {
  const StorageT Size = getCount();
  if (Idx >= Size) {
    return eTypeQualNone;
  }
  const unsigned ShAmt = NumCountBits + (Idx * NumQualBits);
  const unsigned Field = (storage_ >> ShAmt) & ((1 << NumQualBits) - 1);
  return TypeQualifier(Field);
}

bool TypeQualifiers::push_back(TypeQualifier Qual) {
  const StorageT Size = getCount();
  if (Size == MaxSize) {
    return false;
  }
  const unsigned Offset = NumCountBits + (Size * NumQualBits);
  const unsigned Field = Qual & ((1 << NumQualBits) - 1);
  storage_ |= (static_cast<StorageT>(Field) << Offset);
  setCount(Size + 1);
  return true;
}

bool TypeQualifiers::push_back(unsigned Qual) {
  return push_back((TypeQualifier)Qual);
}

bool TypeQualifiers::push_back(TypeQualifiers Quals) {
  while (Quals.getCount() > 0) {
    if (!push_back(Quals.pop_front())) {
      return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////

Lexer::Lexer(StringRef text) : Text(text), Pos(0) {}

unsigned Lexer::Left() const { return Text.size() - Pos; }

unsigned Lexer::CurrentPos() const { return Pos; }

StringRef Lexer::TextLeft() const { return Text.substr(Pos); }

int Lexer::Current() const { return Left() ? Text[Pos] : -1; }

bool Lexer::Consume() { return Consume(1); }

bool Lexer::Consume(unsigned Size) {
  if (Left() < Size) {
    return false;
  }
  Pos += Size;
  return true;
}

bool Lexer::Consume(StringRef Pattern) {
  if (Left() < Pattern.size()) {
    return false;
  } else if (!TextLeft().starts_with(Pattern)) {
    return false;
  }
  Pos += Pattern.size();
  return true;
}

bool Lexer::ConsumeInteger(unsigned &Result) {
  size_t NumDigits = 0;
  size_t i = Pos;
  while ((i < Text.size()) && isdigit(Text[i])) {
    i++;
    NumDigits++;
  }
  const StringRef NumText = Text.substr(Pos, NumDigits);
  if (NumText.size() == 0) {
    return false;
  }
  if (NumText.getAsInteger(10, Result)) {
    return false;
  }
  Pos += NumDigits;
  return true;
}

bool Lexer::ConsumeSignedInteger(int &Result) {
  size_t NumChars = 0;
  size_t i = Pos;
  if (Text[i] == '-') {
    i++;
    NumChars++;
  }
  while ((i < Text.size()) && isdigit(Text[i])) {
    i++;
    NumChars++;
  }
  const StringRef NumText = Text.substr(Pos, NumChars);
  if (NumText.size() == 0) {
    return false;
  }
  if (NumText.getAsInteger(10, Result)) {
    return false;
  }
  Pos += NumChars;
  return true;
}

bool Lexer::ConsumeAlpha(StringRef &Result) {
  size_t NumChars = 0;
  size_t i = Pos;
  while ((i < Text.size()) && isalpha(Text[i])) {
    i++;
    NumChars++;
  }
  if (NumChars == 0) {
    return false;
  }
  Result = Text.substr(Pos, NumChars);
  Pos += NumChars;
  return true;
}

bool Lexer::ConsumeAlphanumeric(StringRef &Result) {
  size_t NumChars = 0;
  size_t i = Pos;
  while ((i < Text.size()) && isalnum(Text[i])) {
    i++;
    NumChars++;
  }
  if (NumChars == 0) {
    return false;
  }
  Result = Text.substr(Pos, NumChars);
  Pos += NumChars;
  return true;
}

bool Lexer::ConsumeUntil(char C, StringRef &Result) {
  const size_t CPos = Text.find_first_of(C, Pos);
  if (CPos == std::string::npos) {
    Result = StringRef();
    return false;
  }
  Result = Text.substr(Pos, CPos - Pos);
  Pos = CPos;
  return true;
}

bool Lexer::ConsumeWhitespace() {
  bool consumed = false;
  while (Pos < Text.size() && isspace(Text[Pos])) {
    consumed = true;
    ++Pos;
  }

  return consumed;
}
}  // namespace utils
}  // namespace compiler
