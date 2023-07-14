//=== SYCLRaiseHost.cpp - Raise host constructs to SYCL dialect operations ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to detect instruction sequences of interest in the MLIR
// (mostly LLVM dialect) for the SYCL host side and raise them to types and
// operations from the SYCL dialect to facilitate analysis in other passes.
//
// Note all patterns defined in this pass must inherit either OpHostRaisePattern
// or OpInterfaceHostRaisePattern.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include <llvm/ADT/STLExtras.h>
#include <type_traits>

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SYCLRAISEHOSTCONSTRUCTS
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {

class SYCLRaiseHostConstructsPass
    : public polygeist::impl::SYCLRaiseHostConstructsBase<
          SYCLRaiseHostConstructsPass> {
public:
  using polygeist::impl::SYCLRaiseHostConstructsBase<
      SYCLRaiseHostConstructsPass>::SYCLRaiseHostConstructsBase;

  void runOnOperation() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

constexpr StringLiteral stdNamespace = "std";
constexpr StringLiteral syclNamespace = "sycl";

/// Class holding results of calling `llvm::ItaniumPartialDemangler` member
/// functions.
///
/// Result is owned by this type.
class DemangleResult {
public:
  template <typename FuncTy,
            typename = std::enable_if_t<std::is_same_v<
                char *, std::invoke_result_t<FuncTy, char *, std::size_t *>>>>
  static FailureOr<DemangleResult> get(FuncTy f) {
    char *buf = nullptr;
    std::size_t size = 0;
    buf = f(buf, &size);
    if (!buf)
      return failure();
    return DemangleResult(buf, size);
  }

  explicit operator StringRef() const { return {buf.get(), size - 1}; }

private:
  DemangleResult(char *buf, std::size_t size) : buf(buf), size(size) {}

  std::unique_ptr<char[]> buf;
  std::size_t size;
};

static Type getEmptyBodyType(MLIRContext *ctx) {
  return LLVM::LLVMVoidType::get(ctx);
}

/// Returns whether function \p demangled is a member of type \p targetType in
/// namespace \p targetNamespace.
static bool isMemberFunction(StringRef demangled, StringRef targetNamespace,
                             StringRef targetType) {
  // Check demangled == targetNamespace::...::targetType
  constexpr StringLiteral namespaceQualifier = "::";
  return demangled.consume_front(targetNamespace) &&
         demangled.consume_front(namespaceQualifier) &&
         demangled.consume_back(targetType) &&
         demangled.ends_with(namespaceQualifier);
}

/// Returns whether the function demangled by \p demangler is a member of type
/// \p targetType in namespace \p targetNamespace.
static bool isMemberFunction(const llvm::ItaniumPartialDemangler &demangler,
                             StringRef targetNamespace, StringRef targetType) {
  FailureOr<DemangleResult> demangled =
      DemangleResult::get([&](char *buf, std::size_t *size) {
        return demangler.getFunctionDeclContextName(buf, size);
      });
  return succeeded(demangled) &&
         isMemberFunction(static_cast<StringRef>(*demangled), targetNamespace,
                          targetType);
}

/// Returns whether the name of the function demangled by \p demangler matches
/// \p targetName.
static bool functionNameMatches(const llvm::ItaniumPartialDemangler &demangler,
                                StringRef targetName) {
  FailureOr<DemangleResult> demangled =
      DemangleResult::get([&](char *buf, std::size_t *size) {
        return demangler.getFunctionBaseName(buf, size);
      });
  return succeeded(demangled) &&
         static_cast<StringRef>(*demangled) == targetName;
}

/// Calls `partialDemangle` on \p demangler using the name of the function
/// referenced by \p ref.
static LogicalResult partialDemangle(llvm::ItaniumPartialDemangler &demangler,
                                     SymbolRefAttr ref) {
  StringRef mangledCalleeName = ref.getLeafReference();
  return failure(demangler.partialDemangle(mangledCalleeName.data()));
}

/// Calls `partialDemangle` on \p demangler using the name of the function
/// called by \p op.
static LogicalResult partialDemangle(llvm::ItaniumPartialDemangler &demangler,
                                     CallOpInterface op) {
  // Get the callable
  auto callable = dyn_cast<SymbolRefAttr>(op.getCallableForCallee());
  return callable ? partialDemangle(demangler, callable) : failure();
}

/// Calls `partialDemangle` on \p demangler using the name of the function \p
/// op.
static LogicalResult partialDemangle(llvm::ItaniumPartialDemangler &demangler,
                                     FunctionOpInterface op) {
  // Get the symbol
  auto symbol = dyn_cast<SymbolOpInterface>(*op);
  return failure(!symbol || demangler.partialDemangle(symbol.getName().data()));
}

template <typename T> static auto getUsersOfType(Value value) {
  constexpr auto filter = [](const OpOperand &operand) -> bool {
    return isa<T>(operand.getOwner());
  };
  constexpr auto map = [](const OpOperand &operand) -> T {
    return cast<T>(operand.getOwner());
  };
  return llvm::map_range(llvm::make_filter_range(value.getUsers(), filter),
                         map);
}

/// Returns whether \p type is an `llvm.struct` type with name \p className.
static bool isClassType(Type type, StringRef className) {
  auto st = dyn_cast<LLVM::LLVMStructType>(type);
  return st && st.isIdentified() && st.getName() == className;
}

/// Returns whether \p type is an `llvm.struct` type with a name matching
/// \p regex.
static bool isClassType(Type type, const llvm::Regex &regex) {
  auto st = dyn_cast<LLVM::LLVMStructType>(type);
  return st && st.isIdentified() && regex.match(st.getName());
}

/// Check \p op is a member of class \p className and return the first (this)
/// argument.
static Value getThisArgument(FunctionOpInterface op, StringRef className) {
  llvm::ItaniumPartialDemangler demangler;
  if (failed(partialDemangle(demangler, op)) || !demangler.isFunction() ||
      !isMemberFunction(demangler, syclNamespace, className) ||
      op.getNumArguments() == 0)
    return nullptr;

  Value firstArg = op.getArgument(0);
  assert(isa<LLVM::LLVMPointerType>(firstArg.getType()) &&
         "Expecting this argument to be a pointer");
  return firstArg;
}

static FailureOr<StringRef> getStringValue(LLVM::GlobalOp op) {
  // Check the operation has a value
  std::optional<Attribute> attr = op.getValue();
  if (!attr)
    return failure();

  // Check it is a string
  auto strAttr = dyn_cast<StringAttr>(*attr);
  if (!strAttr)
    return failure();

  // Drop the trailing `0` character
  return strAttr.getValue().drop_back();
}

static FailureOr<StringRef> getAnnotation(LLVM::VarAnnotation annotation) {
  auto addressofOp =
      annotation.getAnnotation().getDefiningOp<LLVM::AddressOfOp>();
  if (!addressofOp)
    return failure();

  SymbolTableCollection symbolTable;
  auto global = symbolTable.lookupNearestSymbolFrom<LLVM::GlobalOp>(
      addressofOp, addressofOp.getGlobalNameAttr());
  return global ? getStringValue(global) : FailureOr<StringRef>(failure());
}

/// Return the constructor responsible of the construction of \p value.
///
/// Returns an empty instance if conflicting users are found.
static sycl::SYCLHostConstructorOp getConstructor(Value value) {
  // TODO: Implement using ReachingDefinionAnalysis after testing with non
  // structured control flow.
  constexpr auto canOmitOperation = [](Operation *op) {
    return isa<sycl::SYCLDialect>(op->getDialect()) ||
           isa<LLVM::VarAnnotation, LLVM::LifetimeEndOp, LLVM::LifetimeStartOp>(
               op);
  };

  sycl::SYCLHostConstructorOp constructor;
  for (const OpOperand &operand : value.getUses()) {
    Operation *op = operand.getOwner();
    if (auto c = dyn_cast<sycl::SYCLHostConstructorOp>(op)) {
      if (constructor)
        return nullptr;
      constructor = c;
      continue;
    }
    if (!canOmitOperation(op))
      return nullptr;
  }
  return constructor;
}

static Value getHandler(FunctionOpInterface op) {
  Value handler;
  // First search for any operation known to receive a pointer to a handler
  if (op.walk([&](Operation *setKernel) {
          if (!setKernel->hasTrait<sycl::SYCLHostHandlerOp>())
            return WalkResult::advance();
          Value h = setKernel->getOperand(0);
          if (handler && handler != h)
            return WalkResult::interrupt();
          handler = h;
          return WalkResult::advance();
        }).wasInterrupted())
    return nullptr;
  if (!handler)
    handler = getThisArgument(op, "handler");
  return handler;
}

namespace {
template <typename SourceOp>
class OpOrInterfaceHostRaisePatternBase : public RewritePattern {
public:
  using RewritePattern::RewritePattern;

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  void rewrite(Operation *op, PatternRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), rewriter);
  }
  LogicalResult match(Operation *op) const final {
    // Do not run raising patterns in device code
    return isInDeviceModule(op) ? failure() : match(cast<SourceOp>(op));
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    // Do not run raising patterns in device code
    return isInDeviceModule(op) ? failure()
                                : matchAndRewrite(cast<SourceOp>(op), rewriter);
  }

  /// Rewrite and Match methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual void rewrite(SourceOp op, PatternRewriter &rewriter) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
  virtual LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }
  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        PatternRewriter &rewriter) const {
    if (succeeded(match(op))) {
      rewrite(op, rewriter);
      return success();
    }
    return failure();
  }

private:
  static bool isInDeviceModule(Operation *op) {
    return op->getParentOfType<gpu::GPUModuleOp>();
  }
};

/// OpHostRaisePattern is a wrapper around RewritePattern that allows for
/// matching and rewriting against an instance of a derived operation class as
/// opposed to a raw Operation.
///
/// This pattern can only be applied to operations in host code.
template <typename SourceOp>
struct OpHostRaisePattern : public OpOrInterfaceHostRaisePatternBase<SourceOp> {
  /// Patterns must specify the root operation name they match against, and can
  /// also specify the benefit of the pattern matching and a list of generated
  /// ops.
  OpHostRaisePattern(MLIRContext *context, PatternBenefit benefit = 1,
                     ArrayRef<StringRef> generatedNames = {})
      : OpOrInterfaceHostRaisePatternBase<SourceOp>(
            SourceOp::getOperationName(), benefit, context, generatedNames) {}
};

/// OpInterfaceHostRaisePattern is a wrapper around HostRaisePattern that allows
/// for matching and rewriting against an instance of an operation interface
/// instead of a raw Operation.
///
/// This pattern can only be applied to operations in host code.
template <typename SourceOp>
struct OpInterfaceHostRaisePattern
    : public OpOrInterfaceHostRaisePatternBase<SourceOp> {
  OpInterfaceHostRaisePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpOrInterfaceHostRaisePatternBase<SourceOp>(
            Pattern::MatchInterfaceOpTypeTag(), SourceOp::getInterfaceID(),
            benefit, context) {}
};

struct RaiseKernelName : public OpHostRaisePattern<LLVM::AddressOfOp> {
public:
  using OpHostRaisePattern<LLVM::AddressOfOp>::OpHostRaisePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp op,
                                PatternRewriter &rewriter) const final {
    // Get the global this operation uses
    SymbolTableCollection symbolTable;
    auto global = symbolTable.lookupNearestSymbolFrom<LLVM::GlobalOp>(
        op, op.getGlobalNameAttr());
    if (!global)
      return failure();

    // Get a reference to the kernel the global references
    std::optional<SymbolRefAttr> ref = getKernelRef(global, symbolTable);
    if (!ref)
      return failure();

    rewriter.replaceOpWithNewOp<sycl::SYCLHostGetKernelOp>(op, op.getType(),
                                                           *ref);
    return success();
  }

private:
  /// If the input global contains a constant string representing the name of a
  /// SYCL kernel, returns a reference to the `gpu.func` implementing this
  /// kernel.
  ///
  /// The string will contain a trailing character we need to get rid of before
  /// searching.
  static std::optional<SymbolRefAttr>
  getKernelRef(LLVM::GlobalOp op, SymbolTableCollection &symbolTable) {
    FailureOr<StringRef> name = getStringValue(op);
    if (failed(name))
      return std::nullopt;

    // Search the `gpu.func` in the device module
    auto ref =
        SymbolRefAttr::get(op->getContext(), DeviceModuleName,
                           FlatSymbolRefAttr::get(op->getContext(), *name));
    auto kernel = symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, ref);

    // If it was found and it is a kernel, return the reference
    return kernel && kernel.isKernel() ? std::optional<SymbolRefAttr>(ref)
                                       : std::nullopt;
  }
};

template <typename Derived, typename ConstructorOp, typename TypeTag,
          bool PostProcess = false>
class RaiseConstructorBasePattern : public OpHostRaisePattern<ConstructorOp> {
public:
  using OpHostRaisePattern<ConstructorOp>::OpHostRaisePattern;

  LogicalResult matchAndRewrite(ConstructorOp constructor,
                                PatternRewriter &rewriter) const final {

    if (constructor.getArgOperands().empty())
      return failure();

    // 'this*' is the first argument to the constructor call, if it is a
    // constructor.
    auto alloc = constructor.getArgOperands()
                     .front()
                     .template getDefiningOp<LLVM::AllocaOp>();

    if (!alloc)
      return failure();

    assert(alloc.getElemType().has_value() &&
           "Expecting element type attribute for opaque alloca");

    // Check whether the type allocated for the first argument ('this') matches
    // the expected type.
    if (!isClassType(*alloc.getElemType(), tag.getTypeName()))
      return failure();

    if (constructor.getNumResults())
      // Constructor should not return anything.
      return failure();

    if (!isConstructor(constructor))
      // Invoke is not a constructor call.
      return failure();

    auto constructedType = tag.getTypeFromConstructor(constructor);
    if (!constructedType)
      return failure();

    rewriter.create<sycl::SYCLHostConstructorOp>(
        constructor->getLoc(), constructor.getArgOperands().front(),
        constructor.getArgOperands().drop_front(1),
        TypeAttr::get(constructedType));

    if constexpr (PostProcess)
      static_cast<const Derived *>(this)->postprocess(constructor, rewriter);

    rewriter.eraseOp(constructor);
    return success();
  }

private:
  bool isConstructor(ConstructorOp call) const {
    CallInterfaceCallable callableOp = call.getCallableForCallee();
    StringRef funcName = callableOp.get<SymbolRefAttr>().getLeafReference();

    llvm::ItaniumPartialDemangler Demangler;
    Demangler.partialDemangle(funcName.data());
    if (!Demangler.isCtorOrDtor())
      return false;

    FailureOr<DemangleResult> demangled =
        DemangleResult::get([&](char *buf, std::size_t *size) {
          return Demangler.finishDemangle(buf, size);
        });
    if (failed(demangled))
      // Demangling failed
      return false;

    bool isDestructor = static_cast<StringRef>(*demangled).contains('~');
    return !isDestructor;
  }

  TypeTag tag;
};

template <typename TypeTag>
class RaiseInvokeConstructorBasePattern
    : public RaiseConstructorBasePattern<
          RaiseInvokeConstructorBasePattern<TypeTag>, LLVM::InvokeOp, TypeTag,
          true> {
public:
  using RaiseConstructorBasePattern<RaiseInvokeConstructorBasePattern<TypeTag>,
                                    LLVM::InvokeOp, TypeTag,
                                    true>::RaiseConstructorBasePattern;

  void postprocess(LLVM::InvokeOp invoke, PatternRewriter &rewriter) const {
    rewriter.create<LLVM::BrOp>(invoke->getLoc(),
                                invoke.getNormalDestOperands(),
                                invoke.getNormalDest());
  }
};

class BufferTypeTag {
public:
  BufferTypeTag() : regex{"class.sycl::_V1::buffer(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

  mlir::Type getTypeFromConstructor(CallOpInterface constructor) const {
    CallInterfaceCallable callableOp = constructor.getCallableForCallee();
    StringRef constructorName =
        callableOp.get<SymbolRefAttr>().getLeafReference();

    auto demangledName = llvm::demangle(constructorName);

    // Try to determine the dimensions of the buffer by parsing the template
    // parameter from the demangled name of the constructor.
    llvm::Regex bufferTemplate("buffer<.*, ([0-9]+)");
    llvm::SmallVector<StringRef> matches;
    bool regexMatch = bufferTemplate.match(demangledName, &matches);
    if (!regexMatch)
      return nullptr;
    unsigned dimensions = std::stoul(matches[1].str());

    // Determine whether the buffer constructed is a sub-buffer by looking at
    // the type of the first parameter of the demangled constructor function
    // name.
    bool isSubBuffer = false;
    StringRef demangled{demangledName};
    if (auto paramStart = demangled.find('('); paramStart != StringRef::npos) {
      if (demangled.drop_front(paramStart + 1).starts_with("sycl::_V1::buffer"))
        isSubBuffer = true;
    }

    // FIXME: There's currently no good way to obtain the element type of the
    // buffer from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for 'dimensions' above, would require translation
    // from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

    return sycl::BufferType::get(constructor->getContext(), elemTy, dimensions,
                                 isSubBuffer);
  }

private:
  llvm::Regex regex;
};

struct BufferInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<BufferTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      BufferTypeTag>::RaiseInvokeConstructorBasePattern;
};

class AccessorTypeTag {
public:
  AccessorTypeTag() : regex{"class.sycl::_V1::accessor(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

  mlir::Type getTypeFromConstructor(CallOpInterface constructor) const {
    CallInterfaceCallable callableOp = constructor.getCallableForCallee();
    StringRef constructorName =
        callableOp.get<SymbolRefAttr>().getLeafReference();

    auto demangledName = llvm::demangle(constructorName);

    // Try to determine the parameters of the accessor by parsing the template
    // parameter from the demangled name of the constructor.
    llvm::Regex accessorTemplate(
        "accessor<.*, ([0-9]+), \\(sycl::_V1::access::mode\\)([0-9]+), "
        "\\(sycl::_V1::access::target\\)([0-9]+)");
    llvm::SmallVector<StringRef> matches;
    bool regexMatch = accessorTemplate.match(demangledName, &matches);

    if (!regexMatch)
      return nullptr;

    unsigned dimensions = std::stoul(matches[1].str());
    unsigned mode = std::stoul(matches[2].str());
    unsigned target = std::stoul(matches[3].str());

    auto accessModeOrNone = mlir::sycl::symbolizeAccessMode(mode);
    auto accessTargetOrNone = mlir::sycl::symbolizeTarget(target);
    if (!accessModeOrNone || !accessTargetOrNone)
      return nullptr;

    sycl::AccessMode accessMode = *accessModeOrNone;
    sycl::Target accessTarget = *accessTargetOrNone;

    // FIXME: There's currently no good way to obtain the element type of the
    // accessor from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for other parameters above, would require
    // translation from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

    return sycl::AccessorType::get(
        constructor.getContext(), elemTy, dimensions, accessMode, accessTarget,
        // On the host, the body type of the accessor currently has no relevance
        // for the analyses. However, leaving the body types empty leads to
        // problems when parsing an MLIR file containing such an accessor type
        // with empty body type list. Therefore, simply put !llvm.void for now.
        getEmptyBodyType(constructor->getContext()));
  }

private:
  llvm::Regex regex;
};

struct AccessorInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<AccessorTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      AccessorTypeTag>::RaiseInvokeConstructorBasePattern;
};

template <typename TypeTag>
class RaiseArrayConstructorBasePattern
    : public OpHostRaisePattern<LLVM::StoreOp> {
public:
  using OpHostRaisePattern<LLVM::StoreOp>::OpHostRaisePattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op,
                                PatternRewriter &rewriter) const final {
    // The 'this*' is the address to which the element stores are performed.
    auto alloc = op.getAddr().getDefiningOp<LLVM::AllocaOp>();

    if (!alloc || !alloc.getElemType().has_value())
      return failure();

    // Check whether the type allocated for address operand matches the expected
    // type.
    auto allocTy = *alloc.getElemType();
    if (!isClassType(allocTy, tag.getTypeName()))
      return failure();

    auto arrayTyOrNone =
        getNumAndTypeOfComponents(cast<LLVM::LLVMStructType>(allocTy));
    if (!arrayTyOrNone)
      // Failed to identify the number of dimensions/components
      return failure();

    auto numComponents = arrayTyOrNone->getNumElements();
    auto componentTy = arrayTyOrNone->getElementType();

    llvm::SmallVector<Component, 3> components;
    if (!identifyComponents(alloc, op->getBlock(), components))
      // Multiple stores to the same component.
      return failure();

    if (components.size() != numComponents)
      // Expected number of components not matched
      return failure();

    llvm::sort(components, [](const Component &LHS, const Component &RHS) {
      return LHS.byteOffset < RHS.byteOffset;
    });

    assert(componentTy.getIntOrFloatBitWidth());
    auto componentWidth = componentTy.getIntOrFloatBitWidth() / 8;
    llvm::SmallVector<Value, 3> values;
    for (auto &c : components) {
      if ((c.byteOffset / componentWidth) > numComponents)
        // Store outside the expected number of components.
        return failure();

      values.push_back(c.store.getValue());
    }

    // Find the last of the stores in the block and insert the constructor after
    // it.
    auto *lastStore = findLastComponentInBlock(op->getBlock(), components);
    rewriter.setInsertionPointAfter(lastStore);
    rewriter.create<sycl::SYCLHostConstructorOp>(
        op->getLoc(), alloc, values,
        TypeAttr::get(
            TypeTag::SYCLType::get(getContext(), numComponents, componentTy)));

    llvm::for_each(components,
                   [&](Component &c) { rewriter.eraseOp(c.store); });
    return success();
  }

private:
  std::optional<LLVM::LLVMArrayType>
  getNumAndTypeOfComponents(LLVM::LLVMStructType structTy) const {
    if (structTy.getBody().empty())
      return std::nullopt;

    auto detailType =
        dyn_cast<LLVM::LLVMStructType>(structTy.getBody().front());
    static llvm::Regex arrayRegex{"class.sycl::_V1::detail::array(\\.[0-9]+)?"};
    if (!detailType || !arrayRegex.match(detailType.getName()) ||
        detailType.getBody().empty())
      return std::nullopt;

    auto arrayType =
        dyn_cast<LLVM::LLVMArrayType>(detailType.getBody().front());

    return arrayType;
  }

  struct Component {

    Component(size_t offset, LLVM::StoreOp storeOp)
        : byteOffset{offset}, store{storeOp} {}
    size_t byteOffset;
    LLVM::StoreOp store;
  };

  bool identifyComponents(LLVM::AllocaOp alloc, Block *block,
                          llvm::SmallVectorImpl<Component> &components) const {
    for (auto *user : alloc->getUsers()) {
      if (!user->getBlock() || user->getBlock() != block)
        continue;

      if (auto gep = dyn_cast<LLVM::GEPOp>(user)) {
        if (!gep.getDynamicIndices().empty())
          continue;

        auto constantIndices = gep.getRawConstantIndices();
        if (constantIndices.size() != 1)
          continue;

        assert(gep.getElemType().has_value() &&
               "Expecting element type to be set");
        auto byteWidth = gep.getElemType()->getIntOrFloatBitWidth() / 8;

        if (byteWidth == 0)
          continue;

        for (auto *gepUser : gep->getUsers()) {
          if (auto store = dyn_cast<LLVM::StoreOp>(gepUser);
              store && store.getAddr() == gep) {
            if (!insertComponent(components,
                                 byteWidth * constantIndices.front(), store))
              // Another component with the same offset already exists.
              return false;
          }
        }
      }

      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        if (store.getAddr() == alloc)
          if (!insertComponent(components, 0, store))
            // Another component with the same offset already exists.
            return false;
      }
    }
    return true;
  }

  bool insertComponent(llvm::SmallVectorImpl<Component> &components,
                       size_t offset, LLVM::StoreOp store) const {
    if (llvm::any_of(components,
                     [&](Component &c) { return c.byteOffset == offset; }))
      // Another component with the same offset exists.
      return false;

    components.emplace_back(offset, store);
    return true;
  }

  Operation *
  findLastComponentInBlock(mlir::Block *block,
                           llvm::ArrayRef<Component> components) const {
    llvm::SmallPtrSet<Operation *, 3> stores;
    for (const auto &c : components)
      stores.insert(c.store);

    auto lastComponent =
        llvm::find_if(llvm::reverse(*block),
                      [&](Operation &op) { return stores.contains(&op); });
    assert(lastComponent != block->rend() &&
           "Expected to find at least one store in the block");
    return &*lastComponent;
  }

  TypeTag tag;
};

class IDTypeTag {
public:
  using SYCLType = mlir::sycl::IDType;

  IDTypeTag() : regex{"class.sycl::_V1::id(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

private:
  llvm::Regex regex;
};

struct RaiseIDConstructor : public RaiseArrayConstructorBasePattern<IDTypeTag> {
  using RaiseArrayConstructorBasePattern<
      IDTypeTag>::RaiseArrayConstructorBasePattern;
};

class RangeTypeTag {
public:
  using SYCLType = mlir::sycl::RangeType;

  RangeTypeTag() : regex{"class.sycl::_V1::range(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

private:
  llvm::Regex regex;
};

class NDRangeTypeTag {
public:
  using SYCLType = mlir::sycl::NdRangeType;

  NDRangeTypeTag() : regex{"class.sycl::_V1::nd_range(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

  /// Translate an LLVM type to a SYCL type. Do not care about the body.
  static sycl::NdRangeType translateLLVMType(Type type) {
    auto st = cast<LLVM::LLVMStructType>(type);
    auto rt = cast<LLVM::LLVMStructType>(st.getBody()[0]);
    auto at = cast<LLVM::LLVMStructType>(rt.getBody()[0]);
    unsigned dimensions =
        cast<LLVM::LLVMArrayType>(at.getBody()[0]).getNumElements();
    return sycl::NdRangeType::get(type.getContext(), dimensions,
                                  getEmptyBodyType(type.getContext()));
  }

private:
  llvm::Regex regex;
};

struct RaiseRangeConstructor
    : public RaiseArrayConstructorBasePattern<RangeTypeTag> {
  using RaiseArrayConstructorBasePattern<
      RangeTypeTag>::RaiseArrayConstructorBasePattern;
};

class RaiseNDRangeConstructor
    : public OpInterfaceHostRaisePattern<FunctionOpInterface> {
public:
  using OpInterfaceHostRaisePattern<
      FunctionOpInterface>::OpInterfaceHostRaisePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const final {
    // Gather every alloca
    SmallVector<LLVM::AllocaOp> allocas;
    op.walk([&](LLVM::AllocaOp alloca) { allocas.push_back(alloca); });

    // Try to raise constructors from allocas
    bool changed = false;
    for (LLVM::AllocaOp alloca : allocas) {
      if (succeeded(raiseAlloca(op, alloca, rewriter)))
        changed = true;
    }

    return success(changed);
  }

private:
  /// Struct representing the initialzers of an nd_range
  class Initializers {
  public:
    /// Create a new instance from alloca \p alloc. The nd_range to initialize
    /// has \p dimensions dimensions.
    static FailureOr<Initializers>
    get(LLVM::AllocaOp alloc, const NDRangeTypeTag &tag, unsigned dimensions);

    Initializers() = delete;

    /// Return an ArrayRef containing the operations used to initialize the
    /// nd_range or an error if it is bad formed.
    FailureOr<ArrayRef<Operation *>> getGlobalSizeInitializers() const {
      return shrunk(globalSizeInitializers);
    }
    FailureOr<ArrayRef<Operation *>> getLocalSizeInitializers() const {
      return shrunk(localSizeInitializers);
    }
    FailureOr<ArrayRef<Operation *>> getOffsetInitializers() const {
      return shrunk(offsetInitializers);
    }

  private:
    static FailureOr<ArrayRef<Operation *>> shrunk(ArrayRef<Operation *> ops) {
      // Find first null operation
      auto end = llvm::find(ops, nullptr);
      // Check no operation right of end is set
      if (std::any_of(end, ops.end(), llvm::identity<Operation *>()))
        return failure();
      return ArrayRef<Operation *>(ops.begin(), end);
    }

    explicit Initializers(unsigned dimensions) : dimensions(dimensions) {
      assert(0 < dimensions && dimensions <= 3 &&
             "Invalid number of dimensions");
      globalSizeInitializers.fill(nullptr);
      localSizeInitializers.fill(nullptr);
      offsetInitializers.fill(nullptr);
    }

    LogicalResult handle(LLVM::StoreOp store, LLVM::AllocaOp alloc,
                         const NDRangeTypeTag &tag);
    LogicalResult handle(LLVM::MemcpyOp memcpy, LLVM::AllocaOp alloc,
                         const NDRangeTypeTag &tag);
    LogicalResult handle(LLVM::GEPOp gep, LLVM::AllocaOp alloc,
                         const NDRangeTypeTag &tag);

    /// Returns whether this is a valid sequence of initializers
    LogicalResult checkValidity() const;

    /// Safely stores an operation. Will return failure if the setting cannot be
    /// done safely.
    LogicalResult set(std::size_t i, std::size_t j, Operation *op) {
      if (i > 2 || j >= dimensions)
        return failure();
      auto &initializers = [&]() -> std::array<Operation *, 3> & {
        switch (i) {
        case 0:
          return globalSizeInitializers;
        case 1:
          return localSizeInitializers;
        case 2:
          return offsetInitializers;
        default:
          llvm_unreachable("Invalid index");
        }
      }();
      Operation *&ref = initializers[j];
      if (ref)
        return failure();
      ref = op;
      return success();
    }

    unsigned dimensions;
    std::array<Operation *, 3> globalSizeInitializers;
    std::array<Operation *, 3> localSizeInitializers;
    std::array<Operation *, 3> offsetInitializers;
  };

  LogicalResult raiseAlloca(FunctionOpInterface func, LLVM::AllocaOp alloc,
                            PatternRewriter &rewriter) const {
    assert(alloc.getElemType().has_value() &&
           "Expecting element type attribute for opaque alloca");

    // Check whether the type allocated matches the expected type.
    Type elemType = *alloc.getElemType();
    if (!isClassType(elemType, tag.getTypeName()))
      return failure();

    // Get the constructed type early to check number of dimensions
    OpBuilder::InsertionGuard IG(rewriter);
    sycl::NdRangeType constructedType = tag.translateLLVMType(elemType);
    FailureOr<std::array<Value, 3>> args =
        getArgs(func, alloc, constructedType.getDimension(), rewriter);
    if (failed(args))
      return failure();

    // Drop default offset initializer
    ArrayRef<Value> argsRef(*args);
    if (!argsRef[1])
      argsRef = argsRef.drop_back(2);
    else if (!argsRef.back())
      argsRef = argsRef.drop_back();

    rewriter.updateRootInPlace(func, [&] {
      rewriter.create<sycl::SYCLHostConstructorOp>(
          UnknownLoc::get(alloc.getContext()), alloc, argsRef,
          TypeAttr::get(constructedType));
    });

    return success();
  }

  /// Returns the arguments to be passed to the sycl.host.constructor operation.
  ///
  /// Also sets rewriter to the correct insertion point.
  FailureOr<std::array<Value, 3>> getArgs(FunctionOpInterface func,
                                          LLVM::AllocaOp alloc,
                                          unsigned dimensions,
                                          PatternRewriter &rewriter) const;

  NDRangeTypeTag tag;
};

LogicalResult RaiseNDRangeConstructor::Initializers::handle(
    LLVM::StoreOp store, LLVM::AllocaOp alloc, const NDRangeTypeTag &) {
  if (store.getAddr().getDefiningOp() == alloc)
    return set(0, 0, store);
  return success();
}

LogicalResult RaiseNDRangeConstructor::Initializers::handle(
    LLVM::MemcpyOp memcpy, LLVM::AllocaOp alloc, const NDRangeTypeTag &) {
  if (memcpy.getDst().getDefiningOp() == alloc)
    return set(0, 0, memcpy);
  return success();
}

LogicalResult RaiseNDRangeConstructor::Initializers::handle(
    LLVM::GEPOp gep, LLVM::AllocaOp alloc, const NDRangeTypeTag &tag) {
  assert(gep.getElemType().has_value() &&
         "Expecting element type attribute for opaque alloca");

  // In some cases we will find elemType = i8
  if (!isClassType(*gep.getElemType(), tag.getTypeName()))
    return failure();

  auto memcpyUsers = llvm::make_filter_range(
      getUsersOfType<LLVM::MemcpyOp>(gep),
      [=](auto op) { return op.getDst().getDefiningOp() == gep; });
  auto storeUsers =
      llvm::make_filter_range(getUsersOfType<LLVM::StoreOp>(gep), [=](auto op) {
        return op.getAddr().getDefiningOp() == gep;
      });
  constexpr auto isSingleton = [](auto range) {
    return std::next(range.begin()) == range.end();
  };
  Operation *op;
  if (memcpyUsers.empty()) {
    if (storeUsers.empty())
      return success();
    if (!isSingleton(storeUsers))
      return failure();
    // Handle store
    op = *storeUsers.begin();
  } else if (storeUsers.empty()) {
    if (!isSingleton(memcpyUsers))
      return failure();
    // Handle memcpy
    op = *memcpyUsers.begin();
  } else {
    return failure();
  }
  assert(op && "Operation not set");
  ArrayRef<int32_t> indices = gep.getRawConstantIndices();
  std::size_t numIndices = indices.size();
  if (numIndices != 2 && numIndices != 5)
    return failure();
  std::size_t componentIndex = indices[1];
  std::size_t dimensionIndex = numIndices == 5 ? indices[4] : 0;
  return set(componentIndex, dimensionIndex, op);
}

FailureOr<std::array<Value, 3>>
RaiseNDRangeConstructor::getArgs(FunctionOpInterface func, LLVM::AllocaOp alloc,
                                 unsigned dimensions,
                                 PatternRewriter &rewriter) const {
  FailureOr<Initializers> initializers =
      Initializers::get(alloc, tag, dimensions);
  if (failed(initializers))
    return failure();

  // If the id/range is not explicit, we need to reconstruct it
  std::array<Value, 3> args;
  std::array<ArrayRef<Operation *>, 3> ops{
      *initializers->getGlobalSizeInitializers(),
      *initializers->getLocalSizeInitializers(),
      *initializers->getOffsetInitializers()};
  llvm::transform(llvm::enumerate(ops), args.begin(), [&](auto iter) -> Value {
    ArrayRef<Operation *> ops = iter.value();
    // No offset or copy/move constructor cases
    if (ops.empty())
      return nullptr;
    rewriter.setInsertionPointAfter(ops.back());
    // llvm.intr.memcpy
    if (auto memcpy = dyn_cast<LLVM::MemcpyOp>(ops[0])) {
      assert(ops.size() == 1 && "Expecting a single argument when copying");
      rewriter.eraseOp(memcpy);
      return memcpy.getSrc();
    }
    // gep+store
    assert(ops.size() == dimensions && "Expecting a value per dimension");
    OpBuilder::InsertionGuard IG(rewriter);
    rewriter.setInsertionPoint(alloc);
    Type resultType = rewriter.getType<LLVM::LLVMPointerType>();
    Type elementType = cast<LLVM::LLVMStructType>(*alloc.getElemType())
                           .getBody()[iter.index()];
    Location loc = ops[0]->getLoc();
    SmallVector<Value> args;
    llvm::transform(ops, std::back_inserter(args), [](Operation *op) {
      return cast<LLVM::StoreOp>(op).getValue();
    });
    // Build SYCL type. We do not care about the body.
    auto type = [&]() -> Type {
      Type body = getEmptyBodyType(func->getContext());
      switch (iter.index()) {
      case 0:
      case 1:
        // Global or local size
        return rewriter.getType<sycl::RangeType>(dimensions, body);
      case 2:
        // Offset
        return rewriter.getType<sycl::IDType>(dimensions, body);
      default:
        llvm_unreachable("Invalid index");
      }
    }();
    Value result;
    rewriter.updateRootInPlace(func, [&] {
      // Erase all stores
      for (Operation *op : ops)
        rewriter.eraseOp(op);
      // Create new alloca
      Value arraySize =
          rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
      result = rewriter.create<LLVM::AllocaOp>(loc, resultType, elementType,
                                               arraySize);
      // Construct it
      rewriter.create<sycl::SYCLHostConstructorOp>(loc, result, args,
                                                   TypeAttr::get(type));
    });
    return result;
  });

  return args;
}

FailureOr<RaiseNDRangeConstructor::Initializers>
RaiseNDRangeConstructor::Initializers::get(LLVM::AllocaOp alloc,
                                           const NDRangeTypeTag &tag,
                                           unsigned dimensions) {
  Initializers initializers(dimensions);
  for (OpOperand user : alloc->getUsers()) {
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(user.getOwner())
            .Case<LLVM::StoreOp, LLVM::MemcpyOp, LLVM::GEPOp>(
                [&](auto op) { return initializers.handle(op, alloc, tag); })
            .Default([](auto) { return success(); });
    if (failed(result))
      return result;
  }
  if (failed(initializers.checkValidity()))
    return failure();
  return initializers;
}

LogicalResult RaiseNDRangeConstructor::Initializers::checkValidity() const {
  FailureOr<ArrayRef<Operation *>> failureOrGlobalSizeInit =
      getGlobalSizeInitializers();
  FailureOr<ArrayRef<Operation *>> failureOrLocalSizeInit =
      getLocalSizeInitializers();
  FailureOr<ArrayRef<Operation *>> failureOrOffsetInit =
      getOffsetInitializers();
  if (failed(failureOrGlobalSizeInit) || failed(failureOrLocalSizeInit) ||
      failed(failureOrOffsetInit))
    return failure();

  ArrayRef<Operation *> globalSizeInit = *failureOrGlobalSizeInit;
  ArrayRef<Operation *> localSizeInit = *failureOrLocalSizeInit;
  ArrayRef<Operation *> offsetInit = *failureOrOffsetInit;

  if (globalSizeInit.size() == 1 && isa<LLVM::MemcpyOp>(globalSizeInit[0]) &&
      localSizeInit.empty() && offsetInit.empty())
    return success();

  // Check global and local size initialization
  const auto checkInit = [=](ArrayRef<Operation *> ops) {
    if (ops.empty())
      return failure();
    if (ops.size() == dimensions)
      // When initializing each component, we must be using store operations
      return success(dimensions == 1 || llvm::all_of(ops, [](Operation *op) {
                       return isa<LLVM::StoreOp>(op);
                     }));
    // Should be a memcpy
    return success(ops.size() == 1 && isa<LLVM::MemcpyOp>(ops[0]));
  };

  return success(succeeded(checkInit(globalSizeInit)) &&
                 succeeded(checkInit(localSizeInit)) &&
                 (offsetInit.empty() || succeeded(checkInit(offsetInit))));
}

/// Raise constructs assigning a kernel name to a handler to
/// `sycl.host.handler.set_kernel`.
///
/// This pattern acts on `FunctionOpInterface` instances as it will not be
/// removing the operations assigning the kernel name, but creating additional
/// ones to mark the construct.
class RaiseSetKernel : public OpInterfaceHostRaisePattern<FunctionOpInterface> {
public:
  using OpInterfaceHostRaisePattern<
      FunctionOpInterface>::OpInterfaceHostRaisePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const final {
    llvm::ItaniumPartialDemangler demangler;
    bool changed = false;
    // Go through each `sycl.host.get_kernel` introduced
    op.walk([&](sycl::SYCLHostGetKernelOp getKernel) {
      SymbolRefAttr symbolRef = getKernel.getKernelNameAttr();
      // `llvm.invoke` operations using that are candidates for this pattern
      for (auto invoke : getUsersOfType<LLVM::InvokeOp>(getKernel)) {
        // We mark already visited `llvm.invoke` operations to avoid infinite
        // recursion
        constexpr StringLiteral alreadyVisitedAttrName =
            "RaiseSetKernelVisited";
        if (invoke->hasAttr(alreadyVisitedAttrName))
          continue;
        invoke->setAttr(alreadyVisitedAttrName, rewriter.getAttr<UnitAttr>());

        // Arity check
        constexpr unsigned expectedArity = 5;
        if (invoke.getNumOperands() != expectedArity)
          continue;

        // Check the 4th argument (input `str`) is defined by the
        // `sycl.host.get_kernel` operation
        constexpr unsigned inputStringArgumentNumber = 3;
        if (invoke.getOperand(inputStringArgumentNumber) != getKernel)
          continue;

        // Check the first (this) argument is a pointer to a member of a
        // `sycl::handler`
        Value handler = getHandlerFromThisArg(invoke);
        if (!handler)
          continue;

        // Search for `std::basic_string<...>::replace` calls
        constexpr StringLiteral targetType =
            "basic_string<char, std::char_traits<char>, std::allocator<char>>";
        constexpr StringLiteral targetFunction = "_M_replace";
        if (!(succeeded(partialDemangle(demangler, invoke)) &&
              demangler.isFunction() &&
              functionNameMatches(demangler, targetFunction) &&
              isMemberFunction(demangler, stdNamespace, targetType)))
          continue;

        // Introduce operation marking this construct
        rewriter.updateRootInPlace(op, [&] {
          OpBuilder::InsertionGuard IG(rewriter);
          rewriter.setInsertionPoint(invoke);
          rewriter.create<sycl::SYCLHostHandlerSetKernel>(invoke.getLoc(),
                                                          handler, symbolRef);
        });
        changed = true;
      }
    });

    return success(changed);
  }

private:
  /// Return the handler being accessed through the `this` argument of the input
  /// `llvm.invoke` \p op.
  static Value getHandlerFromThisArg(LLVM::InvokeOp op) {
    auto gep = op.getOperand(0).getDefiningOp<LLVM::GEPOp>();
    return gep && isClassType(gep.getSourceElementType(),
                              "class.sycl::_V1::handler")
               ? gep.getBase()
               : Value();
  }
};

class RaiseSetNDRange
    : public OpInterfaceHostRaisePattern<FunctionOpInterface> {
public:
  using OpInterfaceHostRaisePattern<
      FunctionOpInterface>::OpInterfaceHostRaisePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const final {
    // Only handle functions with body
    if (op.isExternal())
      return failure();

    // The function must be a sycl::handler member
    Value handler = getHandler(op);
    if (!handler)
      return failure();

    // Annotated values constructors
    SmallVector<sycl::SYCLHostConstructorOp> constructors;
    // Annotations
    SmallVector<LLVM::VarAnnotation> annotations;

    // We will fill the constructors and annotations vectors
    const auto getConstructors = [&](LLVM::VarAnnotation annotation) {
      // Check the annotation is correct
      FailureOr<StringRef> failureOrAnnotationStr = getAnnotation(annotation);
      if (failed(failureOrAnnotationStr))
        return WalkResult::advance();

      StringRef annotationStr = *failureOrAnnotationStr;
      if (!(annotationStr == "nd_range" || annotationStr == "offset" ||
            annotationStr == "range"))
        return WalkResult::advance();

      // Get value's constructor
      Value value = annotation.getVal();
      sycl::SYCLHostConstructorOp constructor = getConstructor(value);
      if (!constructor)
        return WalkResult::interrupt();

      assert((TypeSwitch<Type, bool>(constructor.getType().getValue())
                  .Case<sycl::NdRangeType>(
                      [=](auto) { return annotationStr == "nd_range"; })
                  .Case<sycl::IDType>(
                      [=](auto) { return annotationStr == "offset"; })
                  .Case<sycl::RangeType>(
                      [=](auto) { return annotationStr == "range"; })
                  .Default(false)) &&
             "Invalid constructor type");

      constructors.push_back(constructor);
      annotations.push_back(annotation);

      return WalkResult::advance();
    };
    if (op.walk(getConstructors).wasInterrupted() || constructors.empty())
      return failure();

    // We will introduce the operation after the last annotation
    OpBuilder::InsertionGuard ig(rewriter);
    rewriter.setInsertionPointAfter(annotations.back());
    Location loc = annotations.back().getLoc();

    // Remove annotations, no longer needed
    // Also prevents infinite recursion
    rewriter.updateRootInPlace(op, [&] {
      for (LLVM::VarAnnotation annotation : annotations)
        rewriter.eraseOp(annotation);
    });

    assert((constructors.size() < 3 &&
            isa<sycl::NdRangeType, sycl::RangeType>(
                constructors.front().getType().getValue()) &&
            (constructors.size() == 1 ||
             (isa<sycl::RangeType>(constructors.front().getType().getValue()) &&
              isa<sycl::IDType>(constructors.back().getType().getValue())))) &&
           "Expecting nd-range or global size+[offset] combination");

    // TODO: nd_range constructor receives pointers to structs, so we are not
    // able to handle this yet.
    if (isa<sycl::NdRangeType>(constructors.front().getType().getValue()))
      return failure();

    SmallVector<Value> arguments = getArguments(op, constructors, rewriter);

    // Finally insert sycl.host.handler.set_nd_range after the last annotation
    rewriter.updateRootInPlace(op, [&] {
      rewriter.create<sycl::SYCLHostHandlerSetNDRange>(loc, handler, arguments);
    });

    return success();
  }

private:
  static SmallVector<Value>
  getArguments(FunctionOpInterface op,
               ArrayRef<sycl::SYCLHostConstructorOp> constructors,
               PatternRewriter &rewriter) {
    SmallVector<Value> arguments;
    OpBuilder::InsertionGuard ig(rewriter);
    llvm::transform(
        constructors, std::back_inserter(arguments),
        [&](sycl::SYCLHostConstructorOp constructor) -> Value {
          // Insert sycl.X.constructor right after sycl.host.constructor
          rewriter.setInsertionPoint(constructor);
          Location loc = constructor.getLoc();
          Type type = constructor.getType().getValue();
          auto mt = MemRefType::get(1, type);
          return TypeSwitch<Type, Value>(type)
              .Case<sycl::RangeType, sycl::IDType>([&](auto t) {
                Value v;
                rewriter.updateRootInPlace(op, [&] {
                  SmallVector<Value> args;
                  auto indexType = rewriter.getIndexType();
                  // Cast each argument from iX to index
                  llvm::transform(constructor.getArgs(),
                                  std::back_inserter(args), [&](Value arg) {
                                    assert(isa<IntegerType>(arg.getType()) &&
                                           "Expecting integer type");
                                    return rewriter.create<arith::IndexCastOp>(
                                        loc, indexType, arg);
                                  });
                  // Create the required constructor operation depending on
                  // the type
                  using OpTy = std::conditional_t<
                      std::is_same_v<decltype(t), sycl::RangeType>,
                      sycl::SYCLRangeConstructorOp, sycl::SYCLIDConstructorOp>;
                  v = rewriter.create<OpTy>(loc, mt, args);
                });
                return v;
              });
        });
    return arguments;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SYCLRaiseHostConstructsPass
//===----------------------------------------------------------------------===//

void SYCLRaiseHostConstructsPass::runOnOperation() {
  Operation *scopeOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet rewritePatterns{context};
  rewritePatterns
      .add<BufferInvokeConstructorPattern, AccessorInvokeConstructorPattern>(
          context);

  // RaiseKernelName should be prioritized, as RaiseSetKernel depends on that.
  rewritePatterns.add<RaiseKernelName>(context, /*benefit=*/2);
  rewritePatterns.add<RaiseSetKernel>(context, /*benefit=*/1);

  // Raising of some constructors (id, range and nd_range) should be
  // prioritized, as RaiseSetNDRange depends on those. Also, raising of id and
  // range constructors should be prioritized, as nd_range constructor uses
  // them.
  rewritePatterns.add<RaiseIDConstructor, RaiseRangeConstructor>(context,
                                                                 /*benefit=*/3);
  rewritePatterns.add<RaiseNDRangeConstructor>(context,
                                               /*benefit=*/2);
  rewritePatterns.add<RaiseSetNDRange>(context, /*benefit=*/1);

  FrozenRewritePatternSet frozen(std::move(rewritePatterns));

  if (failed(applyPatternsAndFoldGreedily(scopeOp, frozen)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass() {
  return std::make_unique<SYCLRaiseHostConstructsPass>();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass(
    const polygeist::SYCLRaiseHostConstructsOptions &options) {
  return std::make_unique<SYCLRaiseHostConstructsPass>(options);
}
