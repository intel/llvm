//===--- Stencil.cpp - Stencil implementation -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/Stencil.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "clang/Tooling/Transformer/SourceCodeBuilders.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include <atomic>
#include <memory>
#include <string>

using namespace clang;
using namespace transformer;

using ast_matchers::MatchFinder;
using ast_type_traits::DynTypedNode;
using llvm::errc;
using llvm::Error;
using llvm::Expected;
using llvm::StringError;

static llvm::Expected<DynTypedNode>
getNode(const ast_matchers::BoundNodes &Nodes, StringRef Id) {
  auto &NodesMap = Nodes.getMap();
  auto It = NodesMap.find(Id);
  if (It == NodesMap.end())
    return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                               "Id not bound: " + Id);
  return It->second;
}

namespace {
// An arbitrary fragment of code within a stencil.
struct RawTextData {
  explicit RawTextData(std::string T) : Text(std::move(T)) {}
  std::string Text;
};

// A debugging operation to dump the AST for a particular (bound) AST node.
struct DebugPrintNodeData {
  explicit DebugPrintNodeData(std::string S) : Id(std::move(S)) {}
  std::string Id;
};

// Operators that take a single node Id as an argument.
enum class UnaryNodeOperator {
  Parens,
  Deref,
  Address,
};

// Generic container for stencil operations with a (single) node-id argument.
struct UnaryOperationData {
  UnaryOperationData(UnaryNodeOperator Op, std::string Id)
      : Op(Op), Id(std::move(Id)) {}
  UnaryNodeOperator Op;
  std::string Id;
};

// The fragment of code corresponding to the selected range.
struct SelectorData {
  explicit SelectorData(RangeSelector S) : Selector(std::move(S)) {}
  RangeSelector Selector;
};

// A stencil operation to build a member access `e.m` or `e->m`, as appropriate.
struct AccessData {
  AccessData(StringRef BaseId, StencilPart Member)
      : BaseId(BaseId), Member(std::move(Member)) {}
  std::string BaseId;
  StencilPart Member;
};

struct IfBoundData {
  IfBoundData(StringRef Id, StencilPart TruePart, StencilPart FalsePart)
      : Id(Id), TruePart(std::move(TruePart)), FalsePart(std::move(FalsePart)) {
  }
  std::string Id;
  StencilPart TruePart;
  StencilPart FalsePart;
};

std::string toStringData(const RawTextData &Data) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "\"";
  OS.write_escaped(Data.Text);
  OS << "\"";
  OS.flush();
  return Result;
}

std::string toStringData(const DebugPrintNodeData &Data) {
  return (llvm::Twine("dPrint(\"") + Data.Id + "\")").str();
}

std::string toStringData(const UnaryOperationData &Data) {
  StringRef OpName;
  switch (Data.Op) {
  case UnaryNodeOperator::Parens:
    OpName = "expression";
    break;
  case UnaryNodeOperator::Deref:
    OpName = "deref";
    break;
  case UnaryNodeOperator::Address:
    OpName = "addressOf";
    break;
  }
  return (OpName + "(\"" + Data.Id + "\")").str();
}

std::string toStringData(const SelectorData &) { return "selection(...)"; }

std::string toStringData(const AccessData &Data) {
  return (llvm::Twine("access(\"") + Data.BaseId + "\", " +
          Data.Member.toString() + ")")
      .str();
}

std::string toStringData(const IfBoundData &Data) {
  return (llvm::Twine("ifBound(\"") + Data.Id + "\", " +
          Data.TruePart.toString() + ", " + Data.FalsePart.toString() + ")")
      .str();
}

std::string toStringData(const MatchConsumer<std::string> &) {
  return "run(...)";
}

// The `evalData()` overloads evaluate the given stencil data to a string, given
// the match result, and append it to `Result`. We define an overload for each
// type of stencil data.

Error evalData(const RawTextData &Data, const MatchFinder::MatchResult &,
               std::string *Result) {
  Result->append(Data.Text);
  return Error::success();
}

Error evalData(const DebugPrintNodeData &Data,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  std::string Output;
  llvm::raw_string_ostream Os(Output);
  auto NodeOrErr = getNode(Match.Nodes, Data.Id);
  if (auto Err = NodeOrErr.takeError())
    return Err;
  NodeOrErr->print(Os, PrintingPolicy(Match.Context->getLangOpts()));
  *Result += Os.str();
  return Error::success();
}

Error evalData(const UnaryOperationData &Data,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  const auto *E = Match.Nodes.getNodeAs<Expr>(Data.Id);
  if (E == nullptr)
    return llvm::make_error<StringError>(
        errc::invalid_argument, "Id not bound or not Expr: " + Data.Id);
  llvm::Optional<std::string> Source;
  switch (Data.Op) {
  case UnaryNodeOperator::Parens:
    Source = tooling::buildParens(*E, *Match.Context);
    break;
  case UnaryNodeOperator::Deref:
    Source = tooling::buildDereference(*E, *Match.Context);
    break;
  case UnaryNodeOperator::Address:
    Source = tooling::buildAddressOf(*E, *Match.Context);
    break;
  }
  if (!Source)
    return llvm::make_error<StringError>(
        errc::invalid_argument,
        "Could not construct expression source from ID: " + Data.Id);
  *Result += *Source;
  return Error::success();
}

Error evalData(const SelectorData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  auto Range = Data.Selector(Match);
  if (!Range)
    return Range.takeError();
  *Result += tooling::getText(*Range, *Match.Context);
  return Error::success();
}

Error evalData(const AccessData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  const auto *E = Match.Nodes.getNodeAs<Expr>(Data.BaseId);
  if (E == nullptr)
    return llvm::make_error<StringError>(errc::invalid_argument,
                                         "Id not bound: " + Data.BaseId);
  if (!E->isImplicitCXXThis()) {
    if (llvm::Optional<std::string> S =
            E->getType()->isAnyPointerType()
                ? tooling::buildArrow(*E, *Match.Context)
                : tooling::buildDot(*E, *Match.Context))
      *Result += *S;
    else
      return llvm::make_error<StringError>(
          errc::invalid_argument,
          "Could not construct object text from ID: " + Data.BaseId);
  }
  return Data.Member.eval(Match, Result);
}

Error evalData(const IfBoundData &Data, const MatchFinder::MatchResult &Match,
               std::string *Result) {
  auto &M = Match.Nodes.getMap();
  return (M.find(Data.Id) != M.end() ? Data.TruePart : Data.FalsePart)
      .eval(Match, Result);
}

Error evalData(const MatchConsumer<std::string> &Fn,
               const MatchFinder::MatchResult &Match, std::string *Result) {
  Expected<std::string> Value = Fn(Match);
  if (!Value)
    return Value.takeError();
  *Result += *Value;
  return Error::success();
}

template <typename T>
class StencilPartImpl : public StencilPartInterface {
  T Data;

public:
  template <typename... Ps>
  explicit StencilPartImpl(Ps &&... Args) : Data(std::forward<Ps>(Args)...) {}

  Error eval(const MatchFinder::MatchResult &Match,
             std::string *Result) const override {
    return evalData(Data, Match, Result);
  }

  std::string toString() const override { return toStringData(Data); }
};
} // namespace

StencilPart Stencil::wrap(StringRef Text) {
  return transformer::text(Text);
}

StencilPart Stencil::wrap(RangeSelector Selector) {
  return transformer::selection(std::move(Selector));
}

void Stencil::append(Stencil OtherStencil) {
  for (auto &Part : OtherStencil.Parts)
    Parts.push_back(std::move(Part));
}

llvm::Expected<std::string>
Stencil::eval(const MatchFinder::MatchResult &Match) const {
  std::string Result;
  for (const auto &Part : Parts)
    if (auto Err = Part.eval(Match, &Result))
      return std::move(Err);
  return Result;
}

StencilPart transformer::text(StringRef Text) {
  return StencilPart(std::make_shared<StencilPartImpl<RawTextData>>(Text));
}

StencilPart transformer::selection(RangeSelector Selector) {
  return StencilPart(
      std::make_shared<StencilPartImpl<SelectorData>>(std::move(Selector)));
}

StencilPart transformer::dPrint(StringRef Id) {
  return StencilPart(std::make_shared<StencilPartImpl<DebugPrintNodeData>>(Id));
}

StencilPart transformer::expression(llvm::StringRef Id) {
  return StencilPart(std::make_shared<StencilPartImpl<UnaryOperationData>>(
      UnaryNodeOperator::Parens, Id));
}

StencilPart transformer::deref(llvm::StringRef ExprId) {
  return StencilPart(std::make_shared<StencilPartImpl<UnaryOperationData>>(
      UnaryNodeOperator::Deref, ExprId));
}

StencilPart transformer::addressOf(llvm::StringRef ExprId) {
  return StencilPart(std::make_shared<StencilPartImpl<UnaryOperationData>>(
      UnaryNodeOperator::Address, ExprId));
}

StencilPart transformer::access(StringRef BaseId, StencilPart Member) {
  return StencilPart(
      std::make_shared<StencilPartImpl<AccessData>>(BaseId, std::move(Member)));
}

StencilPart transformer::ifBound(StringRef Id, StencilPart TruePart,
                             StencilPart FalsePart) {
  return StencilPart(std::make_shared<StencilPartImpl<IfBoundData>>(
      Id, std::move(TruePart), std::move(FalsePart)));
}

StencilPart transformer::run(MatchConsumer<std::string> Fn) {
  return StencilPart(
      std::make_shared<StencilPartImpl<MatchConsumer<std::string>>>(
          std::move(Fn)));
}
