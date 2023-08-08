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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

#include <llvm/ADT/STLExtras.h>
#include <type_traits>

#define DEBUG_TYPE "sycl-raise-host"

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

/// Returns whether the name of the function demangled by \p demangler is a
/// constructor.
static bool isConstructor(const llvm::ItaniumPartialDemangler &demangler) {
  if (!demangler.isCtorOrDtor())
    return false;

  FailureOr<DemangleResult> demangled =
      DemangleResult::get([&](char *buf, std::size_t *size) {
        return demangler.finishDemangle(buf, size);
      });
  if (failed(demangled))
    // Demangling failed
    return false;

  bool isDestructor = static_cast<StringRef>(*demangled).contains('~');
  return !isDestructor;
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

/// Get the parameter list of the function with name \p functionName from the
/// demangler.
static FailureOr<DemangleResult>
getDemangledParameterList(StringRef functionName) {
  llvm::ItaniumPartialDemangler Demangler;
  if (Demangler.partialDemangle(functionName.data()))
    return failure();

  return DemangleResult::get([&](char *buf, size_t *size) {
    return Demangler.getFunctionParameters(buf, size);
  });
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

static std::optional<LLVM::LLVMArrayType>
getNumAndTypeOfComponents(LLVM::LLVMStructType structTy) {
  if (structTy.getBody().empty())
    return std::nullopt;

  auto detailType = dyn_cast<LLVM::LLVMStructType>(structTy.getBody().front());
  static llvm::Regex arrayRegex{"class.sycl::_V1::detail::array(\\.[0-9]+)?"};
  if (!detailType || !arrayRegex.match(detailType.getName()) ||
      detailType.getBody().empty())
    return std::nullopt;

  return dyn_cast<LLVM::LLVMArrayType>(detailType.getBody().front());
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

    if (!tag.isCandidate(constructor))
      // Not handled by the tag.
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

template <typename TypeTag>
class RaiseCallConstructorBasePattern
    : public RaiseConstructorBasePattern<
          RaiseCallConstructorBasePattern<TypeTag>, LLVM::CallOp, TypeTag,
          false> {
public:
  using RaiseConstructorBasePattern<RaiseCallConstructorBasePattern<TypeTag>,
                                    LLVM::CallOp, TypeTag,
                                    false>::RaiseConstructorBasePattern;
};

class BufferTypeTag {
public:
  BufferTypeTag() : regex{"class.sycl::_V1::buffer(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

  bool isCandidate(CallOpInterface constructor) const {
    llvm::ItaniumPartialDemangler demangler;
    return succeeded(partialDemangle(demangler, constructor)) &&
           isConstructor(demangler);
  }

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
    auto failureOrParams = getDemangledParameterList(constructorName);
    if (succeeded(failureOrParams)) {
      auto paramList = static_cast<StringRef>(*failureOrParams);
      isSubBuffer = paramList.starts_with("(sycl::_V1::buffer");
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

  bool isCandidate(CallOpInterface constructor) const {
    llvm::ItaniumPartialDemangler demangler;
    return succeeded(partialDemangle(demangler, constructor)) &&
           isConstructor(demangler);
  }

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

    bool needsRange = true;
    bool needsID = true;
    auto failureOrParams = getDemangledParameterList(constructorName);
    if (succeeded(failureOrParams)) {
      auto paramList = static_cast<StringRef>(*failureOrParams);
      needsRange = paramList.find("sycl::_V1::range") != StringRef::npos;
      needsID = paramList.find("sycl::_V1::id") != StringRef::npos;
    }

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

    auto *context = constructor->getContext();
    // FIXME: There's currently no good way to obtain the element type of the
    // accessor from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for other parameters above, would require
    // translation from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(context);

    // On the host, the body type of the accessor currently has no relevance
    // for the analyses. However, to encode whether the constructed accessor
    // requires the range and id (offset) parameter, the corresponding type is
    // added to the list of body types, if they are required. Otherwise, the
    // body remains empty, encoded by a single `!llvm.void` type, as leaving the
    // body types empty leads to problems when parsing an MLIR file containing
    // such an accessor type with empty body type list.
    SmallVector<Type, 2> bodyTypes;
    if (needsRange)
      bodyTypes.push_back(
          sycl::RangeType::get(context, dimensions, getEmptyBodyType(context)));
    if (needsID)
      bodyTypes.push_back(
          sycl::IDType::get(context, dimensions, getEmptyBodyType(context)));
    if (!needsRange && !needsID)
      bodyTypes.push_back(getEmptyBodyType(context));

    return sycl::AccessorType::get(context, elemTy, dimensions, accessMode,
                                   accessTarget, bodyTypes);
  }

private:
  llvm::Regex regex;
};

struct AccessorInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<AccessorTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      AccessorTypeTag>::RaiseInvokeConstructorBasePattern;
};

// TODO: This tag is limited to match the (deprecated) construction of a local
//       accessor via `sycl::accessor` with `target::local`.
class LocalAccessorTypeTag : public AccessorTypeTag {
public:
  mlir::Type getTypeFromConstructor(CallOpInterface constructor) const {
    CallInterfaceCallable callableOp = constructor.getCallableForCallee();
    StringRef constructorName =
        callableOp.get<SymbolRefAttr>().getLeafReference();

    auto demangledName = llvm::demangle(constructorName);

    // Try to determine the parameters of the local accessor by parsing the
    // template parameter from the demangled name of the constructor.
    llvm::Regex accessorTemplate("local_accessor_base<.*, ([0-9]+)");
    llvm::SmallVector<StringRef> matches;
    bool regexMatch = accessorTemplate.match(demangledName, &matches);

    bool needsRange = false;
    auto failureOrParams = getDemangledParameterList(constructorName);
    if (succeeded(failureOrParams)) {
      auto paramList = static_cast<StringRef>(*failureOrParams);
      needsRange = paramList.find("sycl::_V1::range") != StringRef::npos;
    }

    if (!regexMatch)
      return nullptr;

    unsigned dimensions = std::stoul(matches[1].str());

    auto *context = constructor->getContext();
    // FIXME: There's currently no good way to obtain the element type of the
    // accessor from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for other parameters above, would require
    // translation from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(context);

    // On the host, the body type of the accessor currently has no relevance
    // for the analyses. However, to encode whether the constructed accessor
    // requires the range parameter, the corresponding type is added to the list
    // of body types, if it is required. Otherwise, the body remains empty,
    // encoded by a single `!llvm.void` type, as leaving the body types empty
    // leads to problems when parsing an MLIR file containing such an accessor
    // type with empty body type list.
    Type bodyTy = needsRange ? sycl::RangeType::get(context, dimensions,
                                                    getEmptyBodyType(context))
                             : getEmptyBodyType(context);
    return sycl::LocalAccessorType::get(context, elemTy, dimensions, {bodyTy});
  }
};

struct LocalAccessorInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<LocalAccessorTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      LocalAccessorTypeTag>::RaiseInvokeConstructorBasePattern;
};

// `sycl::buffer::get_access` thinly wraps the construction of an accessor
// object, hence the same patterns as for the actual constructor invocation
// apply here as well.
class GetAccessTypeTag : public AccessorTypeTag {
public:
  bool isCandidate(CallOpInterface constructor) const {
    llvm::ItaniumPartialDemangler demangler;
    if (failed(partialDemangle(demangler, constructor)))
      return false;

    FailureOr<DemangleResult> demangled =
        DemangleResult::get([&](char *buf, std::size_t *size) {
          return demangler.getFunctionDeclContextName(buf, size);
        });
    if (failed(demangled))
      return false;

    StringRef declContext = static_cast<StringRef>(*demangled);
    // Can't use `isMemberFunction` because `sycl::buffer` is a template.
    return declContext.startswith("sycl::_V1::buffer") &&
           functionNameMatches(demangler, "get_access");
  }
};

// Read-only `get_access` requests result in an `llvm.call`.
struct GetAccessCallPattern
    : public RaiseCallConstructorBasePattern<GetAccessTypeTag> {
  using RaiseCallConstructorBasePattern<
      GetAccessTypeTag>::RaiseCallConstructorBasePattern;
};

struct GetAccessInvokePattern
    : public RaiseInvokeConstructorBasePattern<GetAccessTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      GetAccessTypeTag>::RaiseInvokeConstructorBasePattern;
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

    // Check we are not dealing with a default constructor
    if constexpr (TypeTag::hasDefaultConstructor()) {
      auto matcher = m_Zero();
      if (llvm::all_of(values, [&](Value value) {
            return matchPattern(value, matcher);
          }))
        values.clear();
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

  constexpr static bool hasDefaultConstructor() { return true; }

private:
  llvm::Regex regex;
};

/// Pattern raising operations of type \tparam OpTy encoding constructors of
/// the type given by \tparam TypeTag.
template <typename OpTy, typename TypeTag>
class RaiseArrayIntrinsicConstructor : public OpHostRaisePattern<OpTy> {
public:
  virtual ~RaiseArrayIntrinsicConstructor() = default;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    // Check the destination is an alloca of the given type
    auto alloc = op.getDst().template getDefiningOp<LLVM::AllocaOp>();
    if (!alloc || !alloc.getElemType().has_value())
      return failure();

    Type allocTy = *alloc.getElemType();
    if (!isClassType(allocTy, tag.getTypeName()))
      return failure();

    std::optional<LLVM::LLVMArrayType> arrayTyOrNone =
        getNumAndTypeOfComponents(cast<LLVM::LLVMStructType>(allocTy));
    if (!arrayTyOrNone)
      // Failed to identify the number of dimensions/components
      return failure();

    if (failed(finishMatch(op)))
      return failure();

    unsigned numComponents = arrayTyOrNone->getNumElements();
    Type componentTy = arrayTyOrNone->getElementType();

    Type constructedType = rewriter.getType<typename TypeTag::SYCLType>(
        numComponents, componentTy);
    rewriter.replaceOpWithNewOp<sycl::SYCLHostConstructorOp>(
        op, alloc, getArgs(op), TypeAttr::get(constructedType));
    return success();
  }

protected:
  using OpHostRaisePattern<OpTy>::OpHostRaisePattern;

  /// Further match \p op.
  virtual LogicalResult finishMatch(OpTy op) const { return success(); }

  /// Get arguments to be passed to the constructor from the given operation \p
  /// op.
  virtual SmallVector<Value> getArgs(OpTy op) const = 0;

  TypeTag tag;
};

template <typename TypeTag>
class RaiseArrayCopyConstructorBasePattern
    : public RaiseArrayIntrinsicConstructor<LLVM::MemcpyOp, TypeTag> {
public:
  using RaiseArrayIntrinsicConstructor<LLVM::MemcpyOp,
                                       TypeTag>::RaiseArrayIntrinsicConstructor;

protected:
  SmallVector<Value> getArgs(LLVM::MemcpyOp op) const final {
    return {op.getSrc()};
  }
};

struct RaiseIDConstructor : public RaiseArrayConstructorBasePattern<IDTypeTag> {
  using RaiseArrayConstructorBasePattern<
      IDTypeTag>::RaiseArrayConstructorBasePattern;
};

struct RaiseIDCopyConstructor
    : public RaiseArrayCopyConstructorBasePattern<IDTypeTag> {
  using RaiseArrayCopyConstructorBasePattern<
      IDTypeTag>::RaiseArrayCopyConstructorBasePattern;
};

class RaiseIDDefaultConstructor
    : public RaiseArrayIntrinsicConstructor<LLVM::MemsetOp, IDTypeTag> {
public:
  using RaiseArrayIntrinsicConstructor<
      LLVM::MemsetOp, IDTypeTag>::RaiseArrayIntrinsicConstructor;

protected:
  LogicalResult finishMatch(LLVM::MemsetOp op) const final {
    // The value being used as a filler is 0
    return success(matchPattern(op.getVal(), m_Zero()));
  }

  SmallVector<Value> getArgs(LLVM::MemsetOp) const final { return {}; }
};

class RangeTypeTag {
public:
  using SYCLType = mlir::sycl::RangeType;

  RangeTypeTag() : regex{"class.sycl::_V1::range(\\.[0-9]+])?"} {}

  const llvm::Regex &getTypeName() const { return regex; }

  constexpr static bool hasDefaultConstructor() { return false; }

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

struct RaiseRangeCopyConstructor
    : public RaiseArrayCopyConstructorBasePattern<RangeTypeTag> {
  using RaiseArrayCopyConstructorBasePattern<
      RangeTypeTag>::RaiseArrayCopyConstructorBasePattern;
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

    FailureOr<std::pair<std::size_t, std::size_t>>
    getIndices(LLVM::GEPOp gep, const NDRangeTypeTag &tag) const;

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

FailureOr<std::pair<std::size_t, std::size_t>>
RaiseNDRangeConstructor::Initializers::getIndices(
    LLVM::GEPOp gep, const NDRangeTypeTag &tag) const {
  if (!gep.getDynamicIndices().empty())
    // Do not allow dynamic indices
    return failure();

  Type type = *gep.getElemType();
  if (isClassType(type, tag.getTypeName())) {
    // nd_range case
    ArrayRef<int32_t> indices = gep.getRawConstantIndices();
    std::size_t numIndices = indices.size();
    switch (numIndices) {
    case 2:
      return std::pair<std::size_t, std::size_t>(indices[1], 0);
    case 5:
      return std::pair<std::size_t, std::size_t>(indices[1], indices[4]);
    default:
      return failure();
    }
  }
  // i8 case
  if (!type.isInteger(8))
    return failure();
  ArrayRef<int32_t> indices = gep.getRawConstantIndices();
  assert(indices.size() == 1 && "Expecting a single index");

  auto ndType = cast<LLVM::LLVMStructType>(
      *gep.getBase().getDefiningOp<LLVM::AllocaOp>().getElemType());
  assert(isClassType(ndType, tag.getTypeName()) && "Using wrong alloca");
  auto rt = cast<LLVM::LLVMStructType>(ndType.getBody()[0]);
  auto at = cast<LLVM::LLVMStructType>(rt.getBody()[0]);
  Type indexType = cast<LLVM::LLVMArrayType>(at.getBody()[0]).getElementType();
  unsigned indexWidth = indexType.getIntOrFloatBitWidth() / 8;

  std::size_t offset = indices.front();
  assert(offset % indexWidth == 0 && "Unaligned GEP");
  offset /= indexWidth;
  assert(offset < dimensions * 3 && "Out of bounds GEP");
  return std::pair<std::size_t, std::size_t>(offset / dimensions,
                                             offset % dimensions);
} // namespace

LogicalResult RaiseNDRangeConstructor::Initializers::handle(
    LLVM::GEPOp gep, LLVM::AllocaOp alloc, const NDRangeTypeTag &tag) {
  assert(gep.getElemType().has_value() &&
         "Expecting element type attribute for opaque alloca");

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
  auto failureOrIndices = getIndices(gep, tag);
  if (failed(failureOrIndices))
    return failure();
  const auto &[componentIndex, dimensionIndex] = *failureOrIndices;
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
      {
        OpBuilder::InsertionGuard IG(rewriter);
        Value arraySize =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
        rewriter.setInsertionPoint(alloc);
        result = rewriter.create<LLVM::AllocaOp>(loc, resultType, elementType,
                                                 arraySize);
      }
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
  RaiseSetKernel(Pass::Statistic &NumRaisedSetKernelOps, MLIRContext *context,
                 PatternBenefit benefit)
      : OpInterfaceHostRaisePattern<FunctionOpInterface>(context, benefit),
        NumRaisedSetKernelOps(NumRaisedSetKernelOps) {}

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
          ++NumRaisedSetKernelOps;
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

  Pass::Statistic &NumRaisedSetKernelOps;
};

class RaiseSetNDRange
    : public OpInterfaceHostRaisePattern<FunctionOpInterface> {
public:
  RaiseSetNDRange(Pass::Statistic &NumRaisedSetNDRangeOps, MLIRContext *context,
                  PatternBenefit benefit)
      : OpInterfaceHostRaisePattern<FunctionOpInterface>(context, benefit),
        NumRaisedSetNDRangeOps(NumRaisedSetNDRangeOps) {}

  LogicalResult matchAndRewrite(FunctionOpInterface op,
                                PatternRewriter &rewriter) const final {
    // Only handle functions with body
    if (op.isExternal())
      return failure();

    // The function must be a sycl::handler member
    Value handler = getHandler(op);
    if (!handler)
      return failure();

    // Gather arguments from annotations
    bool ndRangePassed = false;
    SmallVector<Argument, 2> arguments;
    const auto getArguments = [&](LLVM::VarAnnotation annotation) {
      // Check the annotation is correct
      FailureOr<StringRef> failureOrAnnotationStr = getAnnotation(annotation);
      if (failed(failureOrAnnotationStr))
        return WalkResult::advance();

      StringRef annotationStr = *failureOrAnnotationStr;
      if (annotationStr == "nd_range" || annotationStr == "range") {
        if (!arguments.empty()) {
          bool ndRangeMismatch = (annotationStr == "nd_range") != ndRangePassed;
          // We can handle the case in which the same [nd-]range pointer is used
          // in different branches. [nd-]range analysis will spot mismatches.
          if (ndRangeMismatch ||
              arguments.front().getValue() != annotation.getVal())
            WalkResult::interrupt();
          arguments.front().addAnnotation(annotation);
          return WalkResult::advance();
        }
        ndRangePassed = annotationStr == "nd_range";
      } else if (annotationStr == "offset") {
        if (arguments.size() != 1) {
          // We can handle the case in which the same offset pointer is used in
          // different branches. id analysis will spot mismatches.
          if (arguments[1].getValue() != annotation.getVal())
            WalkResult::interrupt();
          arguments[1].addAnnotation(annotation);
          return WalkResult::advance();
        }
      } else {
        return WalkResult::advance();
      }

      arguments.emplace_back(annotation, annotation.getVal());

      return WalkResult::advance();
    };
    if (op.walk(getArguments).wasInterrupted() || arguments.empty())
      return failure();

    assert(
        (arguments.size() == 1 || (arguments.size() == 2 && !ndRangePassed)) &&
        "Expecting nd-range or global size+[offset] combination");

    // We will introduce the operation after the last annotation
    OpBuilder::InsertionGuard ig(rewriter);
    LLVM::VarAnnotation lastAnnotation =
        arguments.back().getAnnotations().back();
    rewriter.setInsertionPointAfter(lastAnnotation);
    Location loc = lastAnnotation.getLoc();

    // Remove annotations, no longer needed
    // Also prevents infinite recursion
    rewriter.updateRootInPlace(op, [&] {
      for (const Argument &arg : arguments) {
        for (LLVM::VarAnnotation annot : arg.getAnnotations())
          rewriter.eraseOp(annot);
      }
    });

    // Finally insert sycl.host.handler.set_nd_range after the last annotation
    rewriter.updateRootInPlace(op, [&] {
      switch (arguments.size()) {
      case 1:
        rewriter.create<sycl::SYCLHostHandlerSetNDRange>(
            loc, handler, arguments[0].getValue(), ndRangePassed);
        ++NumRaisedSetNDRangeOps;
        break;
      case 2:
        rewriter.create<sycl::SYCLHostHandlerSetNDRange>(
            loc, handler, arguments[0].getValue(), arguments[1].getValue());
        ++NumRaisedSetNDRangeOps;
        break;
      default:
        llvm_unreachable("Invalid number of arguments");
      }
    });

    return success();
  }

private:
  class Argument {
  public:
    Argument(LLVM::VarAnnotation annotation, Value value)
        : annotations({annotation}), value(value) {
      assert(annotation && value && "All members must be set");
    }

    ArrayRef<LLVM::VarAnnotation> getAnnotations() const { return annotations; }
    Value getValue() const { return value; }
    void addAnnotation(LLVM::VarAnnotation annot) {
      annotations.push_back(annot);
    }

  private:
    SmallVector<LLVM::VarAnnotation> annotations;
    Value value;
  };

  Pass::Statistic &NumRaisedSetNDRangeOps;
};

/// This pattern detects stores (or memcpy ops) to specific elements in a kernel
/// lambda object, and raises them to individual `sycl.host.set_captured` ops.
/// The lambda object is found through an `llvm.intr.var.annotation` with the
/// tag `kernel`.
///
/// Note: This pattern is intended to operate only on the compiler-generated
/// lambda capturing code, which is idiomatic and can be reasonably expected to
/// be alias-free. Hence, instead of employing DFA and alias analysis, the
/// approach for finding assignments-of-interest is detecting GEPs of the form
/// `<lambda obj>[0, <capture idx>]`.
class RaiseSetCaptured : public OpHostRaisePattern<LLVM::VarAnnotation> {
public:
  using OpHostRaisePattern<LLVM::VarAnnotation>::OpHostRaisePattern;
  RaiseSetCaptured(Pass::Statistic &NumRaisedSetCapturedOps,
                   MLIRContext *context, PatternBenefit benefit)
      : OpHostRaisePattern<LLVM::VarAnnotation>(context, benefit),
        NumRaisedSetCapturedOps(NumRaisedSetCapturedOps) {}

  LogicalResult matchAndRewrite(LLVM::VarAnnotation op,
                                PatternRewriter &rewriter) const final {
    // The annotation must be inside a handler function.
    auto enclosingFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!enclosingFunc || !getHandler(enclosingFunc))
      return failure();

    // It must be a "kernel" annotation.
    FailureOr<StringRef> failureOrAnnotationStr = getAnnotation(op);
    if (failed(failureOrAnnotationStr) || *failureOrAnnotationStr != "kernel")
      return failure();

    // There should be exactly one store to the marked pointer.
    Value annotatedPtr = op.getVal();
    auto [store, lambdaObj] = getUniqueAssignment(annotatedPtr);
    if (!store || !isa<LLVM::StoreOp>(store))
      return failure();

    auto alloca = dyn_cast_or_null<LLVM::AllocaOp>(lambdaObj.getDefiningOp());
    if (!alloca || !alloca.getElemType().has_value())
      return failure();

    auto lambdaClassTy =
        dyn_cast<LLVM::LLVMStructType>(alloca.getElemType().value());
    if (!lambdaClassTy)
      return failure();

    auto captureTypes = lambdaClassTy.getBody();

    // Map from index to a collection of array components.
    llvm::IndexedMap<ArrayCollector> idxToArrayComponents;
    idxToArrayComponents.grow(captureTypes.size());

    auto createOps = [&, &lo = lambdaObj](Operation *captureOp,
                                          Value capturedVal, unsigned index) {
      ImplicitLocOpBuilder builder(captureOp->getLoc(), rewriter);
      builder.setInsertionPointAfter(captureOp);

      auto adaptedValuesAndTypes =
          tryToAdapt(capturedVal, captureTypes[index],
                     idxToArrayComponents[index], builder);
      for (auto it : llvm::enumerate(adaptedValuesAndTypes)) {
        auto [adaptedVal, typeAttr] = it.value();
        if (!adaptedVal)
          continue;

        builder.create<sycl::SYCLHostSetCaptured>(
            lo, rewriter.getI64IntegerAttr(index + it.index()), adaptedVal,
            typeAttr);
        ++NumRaisedSetCapturedOps;
      }
    };

    // Look for unique assignments that represent the capturing of kernel args.
    // Assuming the no-op `getelementptr %lambdaObj[0, 0]` was folded, the
    // first capture is actually a store to the pointer.
    // Note that `lambdaObj` is stored into `annotatedPtr` (this is how we found
    // it in the first place), hence we have to specifically allow that here.
    if (auto [captureOp, capturedVal] =
            getUniqueAssignment(lambdaObj, annotatedPtr);
        captureOp) {
      auto completeArray = detectPartialArrayComponent(
          0, captureOp, capturedVal, captureTypes[0], idxToArrayComponents[0]);
      if (completeArray != ArrayResult::PARTIAL_ARRAY) {
        // Only create operations if the captured value was not a part of a
        // bigger, yet incomplete array.
        createOps(captureOp, capturedVal, 0);
      }
    }

    // All other captures are unique assignments to GEPs with two or three
    // constant indices `[0, <capture #>, [<offset>]?]` to the lambda object.
    // The offset case can only happen for array members of the lambda.
    for (auto *user : lambdaObj.getUsers()) {
      auto gep = dyn_cast<LLVM::GEPOp>(user);
      if (!gep || !gep.getElemType().has_value() ||
          gep.getElemType().value() != lambdaClassTy ||
          !gep.getDynamicIndices().empty())
        continue;

      auto indices = gep.getRawConstantIndices();
      if (indices.size() > 3 || indices[0] != 0)
        continue;
      unsigned captureIdx = indices[1];

      auto [captureOp, capturedVal] = getUniqueAssignment(gep);
      if (!captureOp)
        continue;

      auto arrayOffset = (indices.size() == 3) ? indices[2] : 0;
      auto completeArray = detectPartialArrayComponent(
          arrayOffset, captureOp, capturedVal, captureTypes[captureIdx],
          idxToArrayComponents[captureIdx]);

      if (completeArray == ArrayResult::PARTIAL_ARRAY)
        // Detected a partial component of a bigger array. The set_captured will
        // ony be inserted later once all components have been found.
        continue;

      if (completeArray == ArrayResult::NO_ARRAY && indices.size() > 2)
        // Not an array, so only two indices allowed in that case.
        continue;

      createOps(captureOp, capturedVal, captureIdx);
    }

    // Finally erase the annotation op.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// If there is exactly one store or memcpy to \p ptr, return it and the
  /// stored value/copied-from pointer; otherwise return `nullptr` and an emtpy
  /// value.
  static std::tuple<Operation *, Value>
  getUniqueAssignment(Value ptr, Value allowStoreTo = Value()) {
    bool ptrIsAlloca = isa_and_nonnull<LLVM::AllocaOp>(ptr.getDefiningOp());
    Operation *op = nullptr;
    Value value;
    for (auto *user : ptr.getUsers()) {
      if (auto v = TypeSwitch<Operation *, Value>(user)
                       .Case<LLVM::StoreOp>([&](auto st) {
                         return st.getAddr() == ptr ? st.getValue() : Value();
                       })
                       .Case<LLVM::MemcpyOp>([&](auto mc) {
                         return mc.getDst() == ptr ? mc.getSrc() : Value();
                       })
                       .Default(Value())) {
        if (op)
          // Assignment is not unique.
          return {nullptr, Value()};

        op = user;
        value = v;
      } else {
        // Marker intrinsics, and reading from the pointer are unproblematic.
        // (We already know that `ptr` is not the memcpy's `dst` operand.)
        if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp, LLVM::VarAnnotation,
                LLVM::LoadOp, LLVM::MemcpyOp>(user))
          continue;

        // Special case for the first capture: if `ptr` is an alloca (= the
        // lambda object), there may be calls to a destructor or certain API
        // methods with it.
        if (auto call = dyn_cast<CallOpInterface>(user); ptrIsAlloca && call) {
          llvm::ItaniumPartialDemangler demangler;
          auto res = partialDemangle(demangler, call);
          if (succeeded(res)) {
            if (demangler.isCtorOrDtor() && !isConstructor(demangler))
              continue;
            if (isMemberFunction(demangler, syclNamespace, "handler") &&
                functionNameMatches(demangler, "unpack"))
              continue;
          }
        }

        // GEPs with only constant indices have a non-zero offset to `ptr`
        // (otherwise they would've been folded away).
        if (auto gep = dyn_cast<LLVM::GEPOp>(user);
            gep && cast<LLVM::GEPOp>(user).getDynamicIndices().empty()) {
          assert(llvm::any_of(gep.getRawConstantIndices(),
                              [](auto idx) { return idx > 0; }));
          continue;
        }

        // Storing `ptr` somewhere else is only allowed if specifically
        // requested by the caller.
        if (auto st = dyn_cast<LLVM::StoreOp>(user);
            st && st.getAddr() == allowStoreTo)
          continue;

        // Unexpected user; conservatively assume that assignment is not unique.
        LLVM_DEBUG(llvm::dbgs() << "getUniqueAssignment: unexpected user\n";
                   llvm::dbgs().indent(2) << "- " << ptr << '\n';
                   llvm::dbgs().indent(2) << "- " << *user << '\n';);
        return {nullptr, Value()};
      }
    }

    return {op, value};
  }

  /// Represent a part of a bigger array by its offset, the value and the
  /// operation that captured this part of the array.
  struct ArrayComponent {
    ArrayComponent(size_t o, Value v, Operation *op)
        : offset{o}, value{v}, captureOp{op} {}

    size_t offset;
    Value value;
    Operation *captureOp;
  };

  /// Collection of parts of a bigger array. The first element of the pair
  /// indicates how many elements have been collected so far, the second element
  /// is the list of components.
  using ArrayCollector = std::pair<size_t, SmallVector<ArrayComponent>>;

  // Use domain knowledge to try to adapt an \p assigned value to match the
  // \p expected type.
  SmallVector<std::tuple<Value, TypeAttr>>
  tryToAdapt(Value assigned, Type expected, ArrayCollector &collector,
             ImplicitLocOpBuilder &builder) const {
    if (isClassType(expected, accessorTypeTag.getTypeName())) {
      // The getelementpointer[0, <capture #>] we have matched earlier might not
      // address the entire accessor, but rather the first member in the
      // accessor class (think: getelementpointer[0, <capture #>, 0...]).
      // Check whether the assigned value is a load from an alloca with the
      // expected type (i.e. representing an accessor), and if so, return that
      // address instead.
      if (auto load = dyn_cast_or_null<LLVM::LoadOp>(assigned.getDefiningOp()))
        if (auto alloca = dyn_cast_or_null<LLVM::AllocaOp>(
                load.getAddr().getDefiningOp()))
          if (auto elemTy = alloca.getElemType();
              elemTy.has_value() && elemTy.value() == expected)
            if (auto accessorAlloca =
                    dyn_cast<LLVM::AllocaOp>(load.getAddr().getDefiningOp()))
              for (auto *user : accessorAlloca->getUsers())
                if (auto hostCons = dyn_cast<sycl::SYCLHostConstructorOp>(user);
                    hostCons &&
                    isa<sycl::AccessorType, sycl::LocalAccessorType>(
                        hostCons.getType().getValue()))
                  return {{accessorAlloca, hostCons.getType()}};

      LLVM_DEBUG(llvm::dbgs()
                 << "tryToAdapt: Could not infer captured accessor\n");
    }
    if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(expected);
        arrayTy && collector.first != 0) {
      size_t expectedNumElements = arrayTy.getNumElements();
      Type expectedElemType = arrayTy.getElementType();
      assert(expectedNumElements == collector.first && "Incomplete collector");
      SmallVector<ArrayComponent> &offsetAndValues = collector.second;
      // We need to sort them by index to ensure dominance.
      llvm::sort(offsetAndValues,
                 [](const ArrayComponent &p1, const ArrayComponent &p2) {
                   return p1.offset < p2.offset;
                 });
      auto allConstant = llvm::all_of(offsetAndValues, [](const auto &p) {
        return matchPattern(p.value, m_Constant());
      });
      // If all elements of the array are constant, we will insert a global
      // array and capture the address of that global array containing the
      // constant values of the array.
      if (allConstant) {
        builder.setInsertionPoint(
            assigned.getDefiningOp()->getParentOfType<FunctionOpInterface>());
        auto failureOrConstArray = getConstantArray(offsetAndValues, arrayTy);
        if (failed(failureOrConstArray)) {
          return {{nullptr, TypeAttr()}};
        }
        static size_t globalArrayCounter = 0;
        std::string globalName =
            ("constant_array_" + Twine(globalArrayCounter++)).str();
        LLVM::GlobalOp constGlobal = builder.create<LLVM::GlobalOp>(
            arrayTy,
            /* isConstant*/ true, LLVM::Linkage::Private, globalName,
            *failureOrConstArray);

        builder.setInsertionPointAfter(offsetAndValues.back().captureOp);
        auto addrOfGlobal = builder.create<LLVM::AddressOfOp>(constGlobal);

        return {{addrOfGlobal, TypeAttr()}};
      }
      // If not all elements of the array are constant, we will construct a
      // vector<N x Ty> using insert operations and capture that vector value.
      Value vecAccumulator = nullptr;
      for (auto &component : offsetAndValues) {
        builder.setInsertionPointAfter(component.captureOp);
        if (!vecAccumulator)
          vecAccumulator = builder.create<LLVM::UndefOp>(
              VectorType::get(expectedNumElements, expectedElemType));

        auto failureOrAcc =
            TypeSwitch<Type, FailureOr<Value>>(component.value.getType())
                .Case<VectorType>([&](VectorType ty) -> FailureOr<Value> {
                  if (ty.getElementType() != expectedElemType)
                    return failure();

                  return builder
                      .create<vector::InsertStridedSliceOp>(
                          component.value, vecAccumulator, component.offset, 1)
                      ->getResult(0);
                })
                .Default([&](auto ty) -> FailureOr<Value> {
                  if (ty != expectedElemType)
                    return failure();

                  return builder
                      .create<vector::InsertOp>(component.value, vecAccumulator,
                                                component.offset)
                      ->getResult(0);
                });

        if (failed(failureOrAcc))
          return {{nullptr, TypeAttr()}};

        vecAccumulator = *failureOrAcc;
      }

      return {{vecAccumulator, TypeAttr()}};
    }

    // The frontend sometimes groups multiple scalars in vectors - reverse that.
    if (auto vecTy = dyn_cast<VectorType>(assigned.getType());
        vecTy && vecTy.getElementType() == expected) {
      SmallVector<std::tuple<Value, TypeAttr>> scalars;
      for (unsigned i = 0; i < vecTy.getNumElements(); ++i)
        scalars.emplace_back(
            builder.create<vector::ExtractElementOp>(
                expected, assigned, builder.create<arith::ConstantIndexOp>(i)),
            TypeAttr());
      return scalars;
    }

    // No special handling, just return the argument as-is.
    return {{assigned, TypeAttr()}};
  }

  /// Construct a DenseElementAttr to represent a constant array made up from \p
  /// components.
  static FailureOr<DenseElementsAttr>
  getConstantArray(ArrayRef<ArrayComponent> components,
                   LLVM::LLVMArrayType &arrTy) {
    SmallVector<Attribute> constantValues(arrTy.getNumElements());
    for (const auto &p : components) {
      // Retrieve constant value.
      Attribute constantComponent;
      if (!matchPattern(p.value, m_Constant(&constantComponent)))
        return failure();
      size_t offset = p.offset;

      // Retrieve individual elements of the constant array and collect them.
      auto result =
          TypeSwitch<Attribute, LogicalResult>(constantComponent)
              .Case<DenseElementsAttr>([&](auto &denseAttr) {
                for (auto attr : denseAttr.template getValues<Attribute>()) {
                  constantValues[offset++] = attr;
                }
                return success();
              })
              .Case<IntegerAttr, FloatAttr>([&](auto &attr) {
                constantValues[offset] = attr;
                return success();
              })
              .Default(failure());

      if (failed(result))
        return failure();
    }
    return DenseElementsAttr::get(
        RankedTensorType::get(arrTy.getNumElements(), arrTy.getElementType()),
        constantValues);
  }

  enum class ArrayResult { NO_ARRAY, PARTIAL_ARRAY, COMPLETE_ARRAY };
  /// Detect if the \p assigned value is a partial component of a bigger array.
  /// Returns:
  ///  * NO_ARRAY if it is not part of an array
  ///  * PARTIAL_ARRAY if it is part of a bigger array, but not all array
  ///  components have been found so far
  ///  * COMPLETE_ARRAY if \p assigned was the last missing part of the bigger
  ///  array and set_captured should be inserted now.
  static ArrayResult detectPartialArrayComponent(size_t offset,
                                                 Operation *captureOp,
                                                 Value assigned, Type expected,
                                                 ArrayCollector &collector) {
    auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(expected);
    if (!arrayTy)
      return ArrayResult::NO_ARRAY;

    size_t expectedNumElements = arrayTy.getNumElements();
    Type expectedElemTy = arrayTy.getElementType();

    auto numElements =
        TypeSwitch<Type, size_t>(assigned.getType())
            .Case<LLVM::LLVMFixedVectorType, VectorType>([=](auto ty) {
              return (ty.getElementType() == expectedElemTy)
                         ? ty.getNumElements()
                         : 0;
            })
            .Default([=](Type ty) { return (ty == expectedElemTy) ? 1 : 0; });

    if (numElements == 0)
      return ArrayResult::NO_ARRAY;

    // Store the assigned value as a part of the array at the offset.
    collector.second.emplace_back(offset, assigned, captureOp);
    // Increment count of how many elements we have found so far.
    collector.first += numElements;
    LLVM_DEBUG(llvm::dbgs() << "Detected " << std::get<0>(collector)
                            << " components, latest: " << assigned
                            << " from: " << *captureOp << "\n";);
    // Return true in case we have reached the expected number of elements, to
    // trigger insertion of the set_captured operation.
    return (std::get<0>(collector) == expectedNumElements)
               ? ArrayResult::COMPLETE_ARRAY
               : ArrayResult::PARTIAL_ARRAY;
  }

  AccessorTypeTag accessorTypeTag;
  Pass::Statistic &NumRaisedSetCapturedOps;
};

/// Starting from a `sycl.host.handler.set_kernel` op, this pattern discovers
/// the corresponding `sycl.host.handler.set_nd_range` and
/// `sycl.host.set_captured` ops in the CGF. The pattern relies on variable
/// annotations in SYCL's `handler.hpp` to find the number and kinds of kernel
/// parameters. This info is used to verify that suitable values/entities have
/// been captured.
///
/// Known limitations:
/// - This pattern only matches `parallel_for` launches with the kernel passed
///   as a lambda function.
/// - It requires that the actual `parallel_for` call was inlined, so that the
///   precursor ops are all in the same function (= CGF).
/// - Launches without arguments are currently not raised due to their limited
///   practical relevance.
class RaiseScheduleKernel
    : public OpHostRaisePattern<sycl::SYCLHostHandlerSetKernel> {
public:
  using OpHostRaisePattern<sycl::SYCLHostHandlerSetKernel>::OpHostRaisePattern;
  RaiseScheduleKernel(Pass::Statistic &NumRaisedScheduleKernelOps,
                      MLIRContext *context, PatternBenefit benefit)
      : OpHostRaisePattern<sycl::SYCLHostHandlerSetKernel>(context, benefit),
        NumRaisedScheduleKernelOps(NumRaisedScheduleKernelOps) {}

  LogicalResult matchAndRewrite(sycl::SYCLHostHandlerSetKernel op,
                                PatternRewriter &rewriter) const final {
    // `set_kernel` op can only be raised in a CGF.
    auto enclosingFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(enclosingFunc);
    auto handler = op.getHandler();
    assert(handler == getHandler(enclosingFunc));

    // Discover `set_nd_range` ops in the CGF. We are currently only interested
    // in `parallel_for` launches, hence there should be exactly one
    // `set_nd_range` op in the function.
    auto setNDRangeOps =
        enclosingFunc.getOps<sycl::SYCLHostHandlerSetNDRange>();
    if (!llvm::hasNItems(setNDRangeOps, 1))
      return failure();
    auto range = *setNDRangeOps.begin();

    // Extract the number of parameters and the pointer to the
    // `kernel_signatures` data structure from annotations in the predecessor
    // block (the `llvm.invoke` to `extractArgsAndReqsFromLambda` separates the
    // annotations and the `set_kernel` op into different blocks).
    auto *predBlock = op->getBlock()->getSinglePredecessor();
    if (!predBlock)
      return failure();
    auto failureOrParamInfo = getKernelParameterInfo(predBlock);
    if (failed(failureOrParamInfo))
      return failure();
    auto [numParams, paramDesc] = *failureOrParamInfo;

    // Parse `kernel_signatures` to get the parameter kinds.
    auto failureOfParamKinds = getParameterKinds(numParams, paramDesc);
    if (failed(failureOfParamKinds))
      return failure();
    auto paramKinds = *failureOfParamKinds;

    // No point in trying to raise a `schedule_kernel` without arguments.
    if (numParams == 0)
      return failure();

    // Discover `set_captured` ops in the CGF. We check that we have captured a
    // value matching the expected kind for each parameter.
    auto setCapturedOps = enclosingFunc.getOps<sycl::SYCLHostSetCaptured>();
    unsigned numCaptured = llvm::range_size(setCapturedOps);

    // Wrong number of `set_captured` ops.
    if (numCaptured != numParams)
      return failure();

    SmallVector<Value> args(numCaptured);
    SmallVector<Type> syclTypes(numCaptured, rewriter.getNoneType());
    for (auto capture : setCapturedOps) {
      unsigned idx = capture.getIndex();
      // Index is out of bounds.
      if (idx >= numCaptured)
        return failure();

      args[idx] = capture.getValue();

      auto syclType = capture.getSyclType();
      auto kind = paramKinds[idx];
      switch (kind) {
      case kernel_param_kind_t::kind_accessor:
        // We expected an accessor, but the captured value isn't one.
        if (!syclType.has_value() ||
            !isa<sycl::AccessorType, sycl::LocalAccessorType>(syclType.value()))
          return failure();

        syclTypes[idx] = syclType.value();
        break;

      case kernel_param_kind_t::kind_std_layout:
        // We expected standard layout, but the captured value is an accessor.
        if (syclType.has_value() &&
            isa<sycl::AccessorType, sycl::LocalAccessorType>(syclType.value()))
          return failure();
        break;

      default:
        // TODO: support other argument kinds as well
        return failure();
      }
    }

    // `set_captured` ops didn't cover all args.
    if (!args.empty() &&
        !llvm::all_of(args, [](auto v) { return static_cast<bool>(v); }))
      return failure();

    // Finally construct the op with the raised information.
    rewriter.replaceOpWithNewOp<sycl::SYCLHostScheduleKernel>(
        op, handler, op.getKernelName(), range.getRange(), range.getOffset(),
        args, rewriter.getTypeArrayAttr(syclTypes), range.getNdRange());
    ++NumRaisedScheduleKernelOps;

    return success();
  }

private:
  // Copy of `sycl::_V1::detail::kernel_param_kind_t` from
  // `sycl/include/sycl/detail/kernel_desc.hpp`.
  enum class kernel_param_kind_t {
    kind_accessor = 0,
    kind_std_layout = 1, // standard layout object parameters
    kind_sampler = 2,
    kind_pointer = 3,
    kind_specialization_constants_buffer = 4,
    kind_stream = 5,
    kind_invalid = 0xf, // not a valid kernel kind
  };

  /// Returns the unique value stored into the variable annotated by
  /// \p annotation, or nullptr if there is no unique value, or the variable has
  /// unexpected users.
  static Value getAnnotatedValue(LLVM::VarAnnotation annotation) {
    Value ptr = annotation.getVal();
    if (!isa_and_nonnull<LLVM::AllocaOp>(ptr.getDefiningOp()))
      return {};

    Value value;
    for (auto *user : ptr.getUsers()) {
      if (user == annotation)
        continue;
      if (auto v = TypeSwitch<Operation *, Value>(user)
                       .Case<LLVM::StoreOp>([&](auto st) {
                         return st.getAddr() == ptr ? st.getValue() : Value();
                       })
                       .Default(Value())) {
        if (value)
          // Assignment is not unique.
          return {};

        value = v;
      } else {
        // Marker intrinsics are unproblematic.
        if (isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(user))
          continue;

        // Unexpected user; conservatively assume that assignment is not unique.
        return {};
      }
    }

    return value;
  }

  /// Scan \p block for `llvm.intr.var.annotation`s to try to extract the number
  /// of kernel parameters and a pointer to the first descriptor.
  static FailureOr<std::tuple<unsigned, Value>>
  getKernelParameterInfo(Block *block) {
    auto annotations = block->getOps<LLVM::VarAnnotation>();
    if (!llvm::hasNItems(annotations, 2))
      return failure();

    LLVM::VarAnnotation numParamsAnn = *annotations.begin();
    LLVM::VarAnnotation paramDescAnn = *std::next(annotations.begin());
    FailureOr<StringRef> numParamsAnnStr = getAnnotation(numParamsAnn);
    FailureOr<StringRef> paramDescAnnStr = getAnnotation(paramDescAnn);
    if (failed(numParamsAnnStr) || failed(paramDescAnnStr) ||
        *numParamsAnnStr != "kernel_num_params" ||
        *paramDescAnnStr != "kernel_param_desc")
      return failure();

    Value numParams = getAnnotatedValue(numParamsAnn);
    Value paramDesc = getAnnotatedValue(paramDescAnn);

    APInt numParamsConst;
    if (!matchPattern(numParams, m_ConstantInt(&numParamsConst)))
      return failure();

    return std::tuple<unsigned, Value>{numParamsConst.getZExtValue(),
                                       paramDesc};
  }

  /// Try to parse \p numParams kernel parameter descriptors starting from the
  /// \p paramDesc pointer into the `kernel_signatures` constant from the
  /// integration header. If successful, return the descriptors' kinds.
  static FailureOr<SmallVector<kernel_param_kind_t>>
  getParameterKinds(unsigned numParams, Value paramDesc) {
    FlatSymbolRefAttr kernelSignaturesRef;
    unsigned offset = 0;
    if (auto addressOf = paramDesc.getDefiningOp<LLVM::AddressOfOp>()) {
      kernelSignaturesRef = addressOf.getGlobalNameAttr();
    } else {
      auto gep = paramDesc.getDefiningOp<LLVM::GEPOp>();
      if (!gep || !gep.getDynamicIndices().empty())
        return failure();

      auto indices = gep.getRawConstantIndices();
      if (indices.size() != 2 || indices[0] != 0)
        return failure();
      offset = indices[1];

      addressOf = gep.getBase().getDefiningOp<LLVM::AddressOfOp>();
      if (!addressOf)
        return failure();
      kernelSignaturesRef = addressOf.getGlobalNameAttr();
    }

    if (kernelSignaturesRef.getValue() !=
        "_ZN4sycl3_V16detailL17kernel_signaturesE")
      return failure();

    auto kernelSignatures =
        SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
            paramDesc.getDefiningOp(), kernelSignaturesRef);
    if (!kernelSignatures)
      return failure();

    // We have found the compiler-generated `kernel_signatures` constant from
    // the integration header. From now on, assert on assumptions rather than
    // returning `failure()`.

    Value descArray =
        kernelSignatures.getBody()->getTerminator()->getOperand(0);

    auto extractKindFromDescriptor = [](Value desc) -> kernel_param_kind_t {
      APInt kind;
      bool match = matchPattern(
          desc, m_Op<LLVM::InsertValueOp>(
                    m_Op<LLVM::InsertValueOp>(
                        m_Op<LLVM::InsertValueOp>(m_Op<LLVM::UndefOp>(),
                                                  m_ConstantInt(&kind)),
                        m_Constant()),
                    m_Constant()));
      assert(match);
      return static_cast<kernel_param_kind_t>(kind.getZExtValue());
    };

    SmallVector<kernel_param_kind_t> paramKinds(
        numParams, kernel_param_kind_t::kind_invalid);

    Operation *currentOp = descArray.getDefiningOp();
    while (isa_and_nonnull<LLVM::InsertValueOp>(currentOp)) {
      auto insertValue = cast<LLVM::InsertValueOp>(currentOp);
      auto positions = insertValue.getPosition();
      assert(positions.size() == 1);
      unsigned position = positions.front();
      if (position >= offset && position < offset + numParams)
        paramKinds[position - offset] =
            extractKindFromDescriptor(insertValue.getValue());
      currentOp = insertValue.getContainer().getDefiningOp();
    }

    return paramKinds;
  }

  Pass::Statistic &NumRaisedScheduleKernelOps;
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
      .add<BufferInvokeConstructorPattern, AccessorInvokeConstructorPattern,
           LocalAccessorInvokeConstructorPattern, GetAccessCallPattern,
           GetAccessInvokePattern>(context);

  // RaiseKernelName should be prioritized, as RaiseSetKernel depends on that.
  rewritePatterns.add<RaiseKernelName>(context, /*benefit=*/2);
  rewritePatterns.add<RaiseSetKernel>(NumRaisedSetKernelOps, context,
                                      /*benefit=*/1);

  // Raising of some constructors (id, range and nd_range) should be
  // prioritized, as RaiseSetNDRange depends on those. Also, raising of id and
  // range constructors should be prioritized, as nd_range constructor uses
  // them.
  rewritePatterns.add<RaiseIDDefaultConstructor, RaiseIDCopyConstructor,
                      RaiseRangeCopyConstructor, RaiseRangeCopyConstructor,
                      RaiseIDConstructor, RaiseRangeConstructor>(context,
                                                                 /*benefit=*/4);
  rewritePatterns.add<RaiseNDRangeConstructor>(context,
                                               /*benefit=*/3);
  rewritePatterns.add<RaiseSetNDRange>(NumRaisedSetNDRangeOps, context,
                                       /*benefit=*/2);
  rewritePatterns.add<RaiseSetCaptured>(NumRaisedSetCapturedOps, context,
                                        /*benefit=*/2);
  rewritePatterns.add<RaiseScheduleKernel>(NumRaisedScheduleKernelOps, context,
                                           /*benefit=*/1);
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
