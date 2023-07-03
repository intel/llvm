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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"

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

namespace {
struct RaiseKernelName : public OpRewritePattern<LLVM::AddressOfOp> {
public:
  using OpRewritePattern<LLVM::AddressOfOp>::OpRewritePattern;

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
    // Check the operation has a value
    std::optional<Attribute> attr = op.getValue();
    if (!attr)
      return std::nullopt;

    // Check it is a string
    auto strAttr = dyn_cast<StringAttr>(*attr);
    if (!strAttr)
      return std::nullopt;

    // Drop the trailing `0` character
    StringRef name = strAttr.getValue().drop_back();

    // Search the `gpu.func` in the device module
    auto ref =
        SymbolRefAttr::get(op->getContext(), DeviceModuleName,
                           FlatSymbolRefAttr::get(op->getContext(), name));
    auto kernel = symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, ref);

    // If it was found and it is a kernel, return the reference
    return kernel && kernel.isKernel() ? std::optional<SymbolRefAttr>(ref)
                                       : std::nullopt;
  }
};

template <typename Derived, typename ConstructorOp, typename TypeTag,
          bool PostProcess = false>
class RaiseConstructorBasePattern : public OpRewritePattern<ConstructorOp> {
public:
  using OpRewritePattern<ConstructorOp>::OpRewritePattern;

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
    auto allocTy = *alloc.getElemType();
    auto structAllocTy = dyn_cast<LLVM::LLVMStructType>(allocTy);
    if (!structAllocTy || structAllocTy.getName() != TypeTag::getTypeName())
      return failure();

    if (constructor.getNumResults())
      // Constructor should not return anything.
      return failure();

    if (!isConstructor(constructor))
      // Invoke is not a constructor call.
      return failure();

    auto constructedType = TypeTag::getTypeFromConstructor(constructor);
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

    char *demangled = Demangler.finishDemangle(nullptr, 0);
    if (!demangled)
      // Demangling failed
      return false;

    llvm::StringRef demangledName{demangled};
    bool isDestructor = demangledName.contains('~');
    free(demangled);
    return !isDestructor;
  }
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

struct BufferTypeTag {

  static llvm::StringRef getTypeName() { return "class.sycl::_V1::buffer"; }

  static mlir::Type getTypeFromConstructor(CallOpInterface constructor) {
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

    // FIXME: There's currently no good way to obtain the element type of the
    // buffer from the constructor call (or allocation). Parsing it from the
    // demangled name, as done for 'dimensions' above, would require translation
    // from C++ types to MLIR types, which is not available here.
    Type elemTy = LLVM::LLVMVoidType::get(constructor->getContext());

    return sycl::BufferType::get(constructor->getContext(), elemTy, dimensions);
  }
};

struct BufferInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<BufferTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      BufferTypeTag>::RaiseInvokeConstructorBasePattern;
};

struct AccessorTypeTag {

  static llvm::StringRef getTypeName() { return "class.sycl::_V1::accessor"; }

  static mlir::Type getTypeFromConstructor(CallOpInterface constructor) {
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

    return sycl::AccessorType::get(constructor.getContext(), elemTy, dimensions,
                                   accessMode, accessTarget, {});
  }
};

struct AccessorInvokeConstructorPattern
    : public RaiseInvokeConstructorBasePattern<AccessorTypeTag> {
  using RaiseInvokeConstructorBasePattern<
      AccessorTypeTag>::RaiseInvokeConstructorBasePattern;
};

template <typename TypeTag>
class RaiseArrayConstructorBasePattern
    : public OpRewritePattern<LLVM::StoreOp> {
public:
  using OpRewritePattern<LLVM::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op,
                                PatternRewriter &rewriter) const final {
    // The 'this*' is the address to which the element stores are performed.
    auto alloc = op.getAddr().getDefiningOp<LLVM::AllocaOp>();

    if (!alloc || !alloc.getElemType().has_value())
      return failure();

    // Check whether the type allocated for address operand matches the expected
    // type.
    auto allocTy = *alloc.getElemType();
    auto structAllocTy = dyn_cast<LLVM::LLVMStructType>(allocTy);
    if (!structAllocTy ||
        !TypeTag::getTypeName().match(structAllocTy.getName()))
      return failure();

    auto arrayTyOrNone = getNumAndTypeOfComponents(structAllocTy);
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
};

struct IDTypeTag {
  using SYCLType = mlir::sycl::IDType;

  static llvm::Regex &getTypeName() {
    static llvm::Regex regex{"class.sycl::_V1::id(\\.[0-9]+])?"};
    return regex;
  }
};

struct RaiseIDConstructor : public RaiseArrayConstructorBasePattern<IDTypeTag> {
  using RaiseArrayConstructorBasePattern<
      IDTypeTag>::RaiseArrayConstructorBasePattern;
};

struct RangeTypeTag {
  using SYCLType = mlir::sycl::RangeType;

  static llvm::Regex &getTypeName() {
    static llvm::Regex regex{"class.sycl::_V1::range(\\.[0-9]+])?"};
    return regex;
  }
};

struct RaiseRangeConstructor
    : public RaiseArrayConstructorBasePattern<RangeTypeTag> {
  using RaiseArrayConstructorBasePattern<
      RangeTypeTag>::RaiseArrayConstructorBasePattern;
};

/// Raise constructs assigning a kernel name to a handler to
/// `sycl.host.handler.set_kernel`.
///
/// This pattern acts on `FunctionOpInterface` instances as it will not be
/// removing the operations assigning the kernel name, but creating additional
/// ones to mark the construct.
class RaiseSetKernel : public OpInterfaceRewritePattern<FunctionOpInterface> {
public:
  using OpInterfaceRewritePattern<
      FunctionOpInterface>::OpInterfaceRewritePattern;

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
} // namespace

//===----------------------------------------------------------------------===//
// SYCLRaiseHostConstructsPass
//===----------------------------------------------------------------------===//

void SYCLRaiseHostConstructsPass::runOnOperation() {
  Operation *scopeOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet rewritePatterns{context};
  rewritePatterns.add<RaiseKernelName, BufferInvokeConstructorPattern,
                      AccessorInvokeConstructorPattern, RaiseIDConstructor,
                      RaiseRangeConstructor>(context);

  // RaiseKernelName should be prioritized, as RaiseSetKernel depends on that.
  rewritePatterns.add<RaiseKernelName>(context, /*benefit=*/2);
  rewritePatterns.add<RaiseSetKernel>(context, /*benefit=*/1);

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
