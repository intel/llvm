//===- SYCLAccessorAnalysis.cpp - Analysis for sycl::accessor -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/ConstructorBaseAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"

#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "sycl-accessor-analysis"

namespace mlir {
namespace sycl {

//===----------------------------------------------------------------------===//
// AccessorInformation
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const AccessorInformation &info) {
  bool isLocal = info.isLocalAccessor();

  if (isLocal)
    os.indent(4) << "(local_accessor)\n";

  if (info.isTop())
    return os.indent(4) << "<TOP>\n";
  os.indent(4) << "Needs range: " << ((info.needsSubRange()) ? "Yes" : "No")
               << "\n";
  os.indent(4) << "Range: ";
  if (info.hasConstantSubRange()) {
    os << "range{";
    llvm::interleaveComma(info.getConstantRange(), os);
    os << "}\n";
  } else {
    os << "<unknown>\n";
  }

  if (isLocal)
    return os;

  os.indent(4) << "Needs offset: " << ((info.needsOffset()) ? "Yes" : "No")
               << "\n";
  os.indent(4) << "Offset: ";
  if (info.hasConstantOffset()) {
    os << "id{";
    llvm::interleaveComma(info.getConstantOffset(), os);
    os << "}\n";
  } else {
    os << "<unknown>\n";
  }

  os.indent(4) << "Buffer: ";
  if (info.hasKnownBuffer())
    os << info.buffer;
  else
    os << "<unknown>";
  os << "\n";

  os.indent(4) << "Buffer information: ";
  if (info.hasBufferInformation())
    os << info.getBufferInfo();
  else
    os << "<unknown>";
  os << "\n";

  return os;
}

const AccessorInformation
AccessorInformation::join(const AccessorInformation &other,
                          mlir::AliasAnalysis &aliasAnalysis) const {
  // Return top
  if (other.isLocal != isLocal)
    return AccessorInformation();

  // The range can only be omitted if this and the incoming info agree it is
  // not needed, otherwise the conservative default is to assume the range is
  // needed.
  bool jointNeedsRange =
      (needsSubRange() == other.needsSubRange()) ? needsSubRange() : true;
  SmallVector<size_t, 3> jointRange = (constantRange == other.constantRange)
                                          ? constantRange
                                          : SmallVector<size_t, 3>{};

  if (isLocalAccessor())
    return AccessorInformation(jointNeedsRange, jointRange);

  Value jointBuf = nullptr;
  if (hasKnownBuffer() && other.hasKnownBuffer() &&
      (getBuffer() == other.getBuffer() ||
       aliasAnalysis.alias(getBuffer(), other.getBuffer()) ==
           AliasResult::MustAlias))
    jointBuf = getBuffer();

  std::optional<BufferInformation> jointBufInfo = std::nullopt;
  if (hasBufferInformation() && other.hasBufferInformation())
    jointBufInfo = getBufferInfo().join(other.getBufferInfo(), aliasAnalysis);

  // The offset can only be omitted if this and the incoming info agree it is
  // not needed, otherwise the conservative default is to assume the offset is
  // needed.
  bool jointNeedsOffset =
      (needsOffset() == other.needsOffset()) ? needsOffset() : true;
  SmallVector<size_t, 3> jointOffset = (constantOffset == other.constantOffset)
                                           ? constantOffset
                                           : SmallVector<size_t, 3>{};

  return AccessorInformation(jointBuf, jointBufInfo, jointNeedsRange,
                             jointRange, jointNeedsOffset, jointOffset);
}

AliasResult
AccessorInformation::alias(const AccessorInformation &other,
                           mlir::AliasAnalysis &aliasAnalysis) const {
  if (isTop() || other.isTop())
    return AliasResult::MayAlias;

  // A local accessor cannot alias with another accessor unless they are aliased
  // (caught by other analysis).
  if (other.isLocal || isLocal)
    return AliasResult::NoAlias;

  // If we can't analyze the underlying buffer, can't determine aliasing, so
  // must assume they may alias.
  if (!hasKnownBuffer() || !other.hasKnownBuffer())
    return AliasResult::MayAlias;

  auto isSameValue = [&](Value one, Value two) -> bool {
    return one == two ||
           aliasAnalysis.alias(one, two) == AliasResult::MustAlias;
  };

  if (isSameValue(getBuffer(), other.getBuffer())) {
    // Try to refine to must alias
    if (!needsSubRange() && !other.needsSubRange())
      // If neither uses a range (and therefore also no offset), they must
      // alias.
      return AliasResult::MustAlias;

    // Both accessors defined on the same buffer.
    if (!hasBufferInformation() ||
        (needsSubRange() && !hasConstantSubRange()) ||
        (needsOffset() && !hasConstantOffset()) ||
        !other.hasBufferInformation() ||
        (other.needsSubRange() && !other.hasConstantSubRange()) ||
        (other.needsOffset() && !other.hasConstantOffset()))
      // Without definitive information about the underlying buffer and the
      // range & offset of the two accessors, they might be set up in way
      // where they completely or partially overlap, so may alias.
      return AliasResult::MayAlias;

    auto checkFullOverlap =
        [](const AccessorInformation &thisInfo,
           const AccessorInformation &otherInfo) -> AliasResult {
      // Assuming the thisInfo accessor covers the entire buffer (no range and
      // no offset), check whether the otherInfo accessor also covers the entire
      // buffer or not.
      if (otherInfo.hasConstantSubRange() &&
          thisInfo.getBufferInfo().hasConstantSize() &&
          SmallVector<size_t, 3>(otherInfo.getConstantRange()) ==
              SmallVector<size_t, 3>(
                  thisInfo.getBufferInfo().getConstantSize()) &&
          !otherInfo.needsOffset())
        // The otherInfo accessor also covers the entire buffer, so they must
        // alias
        return AliasResult::MustAlias;

      // The otherInfo accessor only covers part of the buffer, so they alias
      // for some elements.
      return AliasResult::MayAlias;
    };

    if (!needsSubRange())
      // This accessor covers the entire range of the buffer.
      return checkFullOverlap(*this, other);

    if (!other.needsSubRange())
      // The other accessor covers the entire range of the buffer.
      return checkFullOverlap(other, *this);

    // If control flow reaches this point, this and the other accessor both
    // require a range.
    if (!hasConstantSubRange() || !other.hasConstantSubRange() ||
        (needsOffset() && !hasConstantOffset()) ||
        (other.needsOffset() && !other.hasConstantOffset()))
      // Not enough information to determine full or partial overlap, assume
      // they may alias.
      return AliasResult::MayAlias;

    auto thisRange = getConstantRange();
    auto otherRange = other.getConstantRange();
    auto thisOffset =
        (needsOffset()) ? constantOffset : SmallVector<size_t, 3>{};
    auto otherOffset =
        (other.needsOffset()) ? other.constantOffset : SmallVector<size_t, 3>{};

    if (thisRange == otherRange && thisOffset == otherOffset)
      // If both cover the same part of the buffer, they must alias.
      return AliasResult::MustAlias;

    if (thisRange.size() != otherRange.size() ||
        thisOffset.size() != otherOffset.size())
      // Insufficient information to further refine, assume they may alias.
      return AliasResult::MayAlias;

    auto noOverlap = [](ArrayRef<size_t> offset, ArrayRef<size_t> range,
                        ArrayRef<size_t> otherOffset) -> bool {
      return llvm::all_of(llvm::zip_equal(offset, range, otherOffset),
                          [](const std::tuple<size_t, size_t, size_t> &t) {
                            return (std::get<2>(t) >=
                                    std::get<0>(t) + std::get<1>(t));
                          });
    };

    if (noOverlap(thisOffset, thisRange, otherOffset) ||
        noOverlap(otherOffset, otherRange, thisOffset))
      // If the areas covered by the accessors do not overlap, they do not
      // overlap.
      return AliasResult::NoAlias;

    // Could not refine further, assume they overlap partially, so may alias.
    return AliasResult::MayAlias;
  }

  if (aliasAnalysis.alias(getBuffer(), other.getBuffer()) !=
      AliasResult::NoAlias)
    return AliasResult::MayAlias;

  // The two accessors are defined over two different buffers.
  if (hasBufferInformation() &&
      getBufferInfo().getSubBuffer() == SubBufferLattice::NO &&
      other.hasBufferInformation() &&
      other.getBufferInfo().getSubBuffer() == SubBufferLattice::NO)
    // If we definitely know that neither of the two buffers is a sub-buffer,
    // they won't alias.
    return AliasResult::NoAlias;

  // Otherwise, they might still alias.
  return AliasResult::MayAlias;
}

//===----------------------------------------------------------------------===//
// SYCLAccessorAnalysis
//===----------------------------------------------------------------------===//

SYCLAccessorAnalysis::SYCLAccessorAnalysis(Operation *op, AnalysisManager &mgr)
    : ConstructorBaseAnalysis<SYCLAccessorAnalysis, AccessorInformation>(op,
                                                                         mgr),
      idRangeAnalysis(op, mgr), bufferAnalysis(op, mgr) {}

void SYCLAccessorAnalysis::finalizeInitialization(bool useRelaxedAliasing) {
  idRangeAnalysis.initialize(useRelaxedAliasing);
  bufferAnalysis.initialize(useRelaxedAliasing);
}

std::optional<AccessorInformation>
SYCLAccessorAnalysis::getAccessorInformationFromConstruction(Operation *op,
                                                             Value operand) {
  return getInformationFromConstruction<sycl::AccessorType,
                                        sycl::LocalAccessorType>(op, operand);
}

template <>
AccessorInformation
SYCLAccessorAnalysis::getInformationImpl<sycl::AccessorType,
                                         sycl::LocalAccessorType>(
    const polygeist::Definition &def) {
  assert(def.isOperation() && "Expecting operation");
  auto constructor = cast<sycl::SYCLHostConstructorOp>(def.getOperation());
  return TypeSwitch<Type, AccessorInformation>(constructor.getType().getValue())
      .Case<sycl::AccessorType>([=](auto accessorTy) {
        Value buffer = nullptr;
        std::optional<BufferInformation> bufferInfo = std::nullopt;
        if (constructor->getNumOperands() > 2) {
          buffer = constructor->getOperand(1);
          bufferInfo = bufferAnalysis.getBufferInformationFromConstruction(
              constructor, constructor->getOperand(1));
        }

        // Whether the range and offset parameter are required by the accessor
        // constructed is encoded in the list of body types.
        bool needsRange = llvm::any_of(accessorTy.getBody(), [](Type ty) {
          return isa<sycl::RangeType>(ty);
        });
        bool needsOffset = llvm::any_of(accessorTy.getBody(), [](Type ty) {
          return isa<sycl::IDType>(ty);
        });
        SmallVector<size_t, 3> constRange;
        SmallVector<size_t, 3> constOffset;
        if (needsRange) {
          // In the SYCL API, the range can either be the second or third
          // parameter to the constructor.
          std::optional<IDRangeInformation> optRangeInfo =
              getOperandInfo<sycl::RangeType>(constructor, {2, 3});
          if (optRangeInfo.has_value() && optRangeInfo->isConstant())
            constRange = optRangeInfo->getConstantValues();
        }
        if (needsOffset) {
          // In the SYCL API, the offset can either be the third or fourth
          // parameter to the constructor.
          std::optional<IDRangeInformation> optIDInfo =
              getOperandInfo<sycl::IDType>(constructor, {3, 4});
          if (optIDInfo.has_value() && optIDInfo->isConstant())
            constOffset = optIDInfo->getConstantValues();
        }

        return AccessorInformation(buffer, bufferInfo, needsRange, constRange,
                                   needsOffset, constOffset);
      })
      .Case<sycl::LocalAccessorType>([=](auto accessorTy) {
        bool needsRange = llvm::any_of(accessorTy.getBody(), [](Type ty) {
          return isa<sycl::RangeType>(ty);
        });
        ArrayRef<size_t> constRange = std::nullopt;
        if (needsRange) {
          // In the SYCL API, the range must be the first parameter to the
          // constructor.
          std::optional<IDRangeInformation> optRangeInfo =
              getOperandInfo<sycl::RangeType>(constructor, 1);
          if (optRangeInfo.has_value() && optRangeInfo->isConstant())
            constRange = optRangeInfo->getConstantValues();
        }
        return AccessorInformation(needsRange, constRange);
      });
}

template <typename OperandType>
std::optional<IDRangeInformation>
SYCLAccessorAnalysis::getOperandInfo(sycl::SYCLHostConstructorOp constructor,
                                     ArrayRef<size_t> possibleIndices) {
  auto getInfo = [&](size_t index) -> std::optional<IDRangeInformation> {
    if (index < constructor->getNumOperands()) {
      auto *defOp = constructor->getOperand(index).getDefiningOp();
      if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
        // For one-dimensional id/range, the frontend might generate a scalar
        // value instead. This tries to detect the use of a constant scalar as
        // range or offset.
        SmallVector<OpFoldResult> folded;
        if (succeeded(defOp->fold({}, folded)) && folded.size() == 1 &&
            folded.front().is<Attribute>() &&
            isa<IntegerAttr>(folded.front().get<Attribute>()))
          return IDRangeInformation{ArrayRef<size_t>{static_cast<size_t>(
              cast<IntegerAttr>(folded.front().get<Attribute>()).getInt())}};
      }
      if (isa<LLVM::LLVMPointerType>(constructor->getOperand(index).getType()))
        return idRangeAnalysis
            .getIDRangeInformationFromConstruction<OperandType>(
                constructor, constructor->getOperand(index));
    }
    return std::nullopt;
  };

  for (size_t index : possibleIndices) {
    std::optional<IDRangeInformation> info = getInfo(index);
    if (info)
      return info;
  }
  return std::nullopt;
}

} // namespace sycl
} // namespace mlir
