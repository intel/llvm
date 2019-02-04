//===--- DiagnosticsTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdUnit.h"
#include "SourceCode.h"
#include "TestIndex.h"
#include "TestTU.h"
#include "index/MemIndex.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

using testing::ElementsAre;
using testing::Field;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

testing::Matcher<const Diag &> WithFix(testing::Matcher<Fix> FixMatcher) {
  return Field(&Diag::Fixes, ElementsAre(FixMatcher));
}

testing::Matcher<const Diag &> WithNote(testing::Matcher<Note> NoteMatcher) {
  return Field(&Diag::Notes, ElementsAre(NoteMatcher));
}

MATCHER_P2(Diag, Range, Message,
           "Diag at " + llvm::to_string(Range) + " = [" + Message + "]") {
  return arg.Range == Range && arg.Message == Message;
}

MATCHER_P3(Fix, Range, Replacement, Message,
           "Fix " + llvm::to_string(Range) + " => " +
               testing::PrintToString(Replacement) + " = [" + Message + "]") {
  return arg.Message == Message && arg.Edits.size() == 1 &&
         arg.Edits[0].range == Range && arg.Edits[0].newText == Replacement;
}

MATCHER_P(EqualToLSPDiag, LSPDiag,
          "LSP diagnostic " + llvm::to_string(LSPDiag)) {
  return std::tie(arg.range, arg.severity, arg.message) ==
         std::tie(LSPDiag.range, LSPDiag.severity, LSPDiag.message);
}

MATCHER_P(EqualToFix, Fix, "LSP fix " + llvm::to_string(Fix)) {
  if (arg.Message != Fix.Message)
    return false;
  if (arg.Edits.size() != Fix.Edits.size())
    return false;
  for (std::size_t I = 0; I < arg.Edits.size(); ++I) {
    if (arg.Edits[I].range != Fix.Edits[I].range ||
        arg.Edits[I].newText != Fix.Edits[I].newText)
      return false;
  }
  return true;
}


// Helper function to make tests shorter.
Position pos(int line, int character) {
  Position Res;
  Res.line = line;
  Res.character = character;
  return Res;
}

TEST(DiagnosticsTest, DiagnosticRanges) {
  // Check we report correct ranges, including various edge-cases.
  Annotations Test(R"cpp(
    namespace test{};
    void $decl[[foo]]();
    int main() {
      $typo[[go\
o]]();
      foo()$semicolon[[]]//with comments
      $unk[[unknown]]();
      double $type[[bar]] = "foo";
      struct Foo { int x; }; Foo a;
      a.$nomember[[y]];
      test::$nomembernamespace[[test]];
    }
  )cpp");
  EXPECT_THAT(
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(
          // This range spans lines.
          AllOf(Diag(Test.range("typo"),
                     "use of undeclared identifier 'goo'; did you mean 'foo'?"),
                WithFix(
                    Fix(Test.range("typo"), "foo", "change 'go\\ o' to 'foo'")),
                // This is a pretty normal range.
                WithNote(Diag(Test.range("decl"), "'foo' declared here"))),
          // This range is zero-width and insertion. Therefore make sure we are
          // not expanding it into other tokens. Since we are not going to
          // replace those.
          AllOf(Diag(Test.range("semicolon"), "expected ';' after expression"),
                WithFix(Fix(Test.range("semicolon"), ";", "insert ';'"))),
          // This range isn't provided by clang, we expand to the token.
          Diag(Test.range("unk"), "use of undeclared identifier 'unknown'"),
          Diag(Test.range("type"),
               "cannot initialize a variable of type 'double' with an lvalue "
               "of type 'const char [4]'"),
          Diag(Test.range("nomember"), "no member named 'y' in 'Foo'"),
          Diag(Test.range("nomembernamespace"),
               "no member named 'test' in namespace 'test'")));
}

TEST(DiagnosticsTest, FlagsMatter) {
  Annotations Test("[[void]] main() {}");
  auto TU = TestTU::withCode(Test.code());
  EXPECT_THAT(TU.build().getDiagnostics(),
              ElementsAre(AllOf(Diag(Test.range(), "'main' must return 'int'"),
                                WithFix(Fix(Test.range(), "int",
                                            "change 'void' to 'int'")))));
  // Same code built as C gets different diagnostics.
  TU.Filename = "Plain.c";
  EXPECT_THAT(
      TU.build().getDiagnostics(),
      ElementsAre(AllOf(
          Diag(Test.range(), "return type of 'main' is not 'int'"),
          WithFix(Fix(Test.range(), "int", "change return type to 'int'")))));
}

TEST(DiagnosticsTest, ClangTidy) {
  Annotations Test(R"cpp(
    #include $deprecated[["assert.h"]]

    #define $macrodef[[SQUARE]](X) (X)*(X)
    int main() {
      return $doubled[[sizeof]](sizeof(int));
      int y = 4;
      return SQUARE($macroarg[[++]]y);
    }
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  TU.HeaderFilename = "assert.h"; // Suppress "not found" error.
  TU.ClangTidyChecks =
      "-*, bugprone-sizeof-expression, bugprone-macro-repeated-side-effects, "
      "modernize-deprecated-headers";
  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("deprecated"),
                     "inclusion of deprecated C++ header 'assert.h'; consider "
                     "using 'cassert' instead [modernize-deprecated-headers]"),
                WithFix(Fix(Test.range("deprecated"), "<cassert>",
                            "change '\"assert.h\"' to '<cassert>'"))),
          Diag(Test.range("doubled"),
               "suspicious usage of 'sizeof(sizeof(...))' "
               "[bugprone-sizeof-expression]"),
          AllOf(
              Diag(Test.range("macroarg"),
                   "side effects in the 1st macro argument 'X' are repeated in "
                   "macro expansion [bugprone-macro-repeated-side-effects]"),
              WithNote(Diag(Test.range("macrodef"),
                            "macro 'SQUARE' defined here "
                            "[bugprone-macro-repeated-side-effects]"))),
          Diag(Test.range("macroarg"),
               "multiple unsequenced modifications to 'y'")));
}

TEST(DiagnosticsTest, Preprocessor) {
  // This looks like a preamble, but there's an #else in the middle!
  // Check that:
  //  - the #else doesn't generate diagnostics (we had this bug)
  //  - we get diagnostics from the taken branch
  //  - we get no diagnostics from the not taken branch
  Annotations Test(R"cpp(
    #ifndef FOO
    #define FOO
      int a = [[b]];
    #else
      int x = y;
    #endif
    )cpp");
  EXPECT_THAT(
      TestTU::withCode(Test.code()).build().getDiagnostics(),
      ElementsAre(Diag(Test.range(), "use of undeclared identifier 'b'")));
}

TEST(DiagnosticsTest, InsideMacros) {
  Annotations Test(R"cpp(
    #define TEN 10
    #define RET(x) return x + 10

    int* foo() {
      RET($foo[[0]]);
    }
    int* bar() {
      return $bar[[TEN]];
    }
    )cpp");
  EXPECT_THAT(TestTU::withCode(Test.code()).build().getDiagnostics(),
              ElementsAre(Diag(Test.range("foo"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'"),
                          Diag(Test.range("bar"),
                               "cannot initialize return object of type "
                               "'int *' with an rvalue of type 'int'")));
}

TEST(DiagnosticsTest, ToLSP) {
  clangd::Diag D;
  D.Message = "something terrible happened";
  D.Range = {pos(1, 2), pos(3, 4)};
  D.InsideMainFile = true;
  D.Severity = DiagnosticsEngine::Error;
  D.File = "foo/bar/main.cpp";

  clangd::Note NoteInMain;
  NoteInMain.Message = "declared somewhere in the main file";
  NoteInMain.Range = {pos(5, 6), pos(7, 8)};
  NoteInMain.Severity = DiagnosticsEngine::Remark;
  NoteInMain.File = "../foo/bar/main.cpp";
  NoteInMain.InsideMainFile = true;
  D.Notes.push_back(NoteInMain);

  clangd::Note NoteInHeader;
  NoteInHeader.Message = "declared somewhere in the header file";
  NoteInHeader.Range = {pos(9, 10), pos(11, 12)};
  NoteInHeader.Severity = DiagnosticsEngine::Note;
  NoteInHeader.File = "../foo/baz/header.h";
  NoteInHeader.InsideMainFile = false;
  D.Notes.push_back(NoteInHeader);

  clangd::Fix F;
  F.Message = "do something";
  D.Fixes.push_back(F);

  auto MatchingLSP = [](const DiagBase &D, StringRef Message) {
    clangd::Diagnostic Res;
    Res.range = D.Range;
    Res.severity = getSeverity(D.Severity);
    Res.message = Message;
    return Res;
  };

  // Diagnostics should turn into these:
  clangd::Diagnostic MainLSP =
      MatchingLSP(D, R"(Something terrible happened (fix available)

main.cpp:6:7: remark: declared somewhere in the main file

../foo/baz/header.h:10:11:
note: declared somewhere in the header file)");

  clangd::Diagnostic NoteInMainLSP =
      MatchingLSP(NoteInMain, R"(Declared somewhere in the main file

main.cpp:2:3: error: something terrible happened)");

  // Transform dianostics and check the results.
  std::vector<std::pair<clangd::Diagnostic, std::vector<clangd::Fix>>> LSPDiags;
  toLSPDiags(D,
#ifdef _WIN32
             URIForFile::canonicalize("c:\\path\\to\\foo\\bar\\main.cpp",
                                      /*TUPath=*/""),
#else
      URIForFile::canonicalize("/path/to/foo/bar/main.cpp", /*TUPath=*/""),
#endif
             ClangdDiagnosticOptions(),
             [&](clangd::Diagnostic LSPDiag, ArrayRef<clangd::Fix> Fixes) {
               LSPDiags.push_back(
                   {std::move(LSPDiag),
                    std::vector<clangd::Fix>(Fixes.begin(), Fixes.end())});
             });

  EXPECT_THAT(
      LSPDiags,
      ElementsAre(Pair(EqualToLSPDiag(MainLSP), ElementsAre(EqualToFix(F))),
                  Pair(EqualToLSPDiag(NoteInMainLSP), IsEmpty())));
}

TEST(IncludeFixerTest, IncompleteType) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
  class X;
}
class Y : $base[[public ns::X]] {};
int main() {
  ns::X *x;
  x$access[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  Symbol Sym = cls("ns::X");
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  Sym.CanonicalDeclaration.FileURI = "unittest:///x.h";
  Sym.Definition.FileURI = "unittest:///x.h";
  Sym.IncludeHeaders.emplace_back("\"x.h\"", 1);

  SymbolSlab::Builder Slab;
  Slab.insert(Sym);
  auto Index = MemIndex::build(std::move(Slab).build(), RefSlab());
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(
      TU.build().getDiagnostics(),
      UnorderedElementsAre(
          AllOf(Diag(Test.range("base"), "base class has incomplete type"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X"))),
          AllOf(Diag(Test.range("access"),
                     "member access into incomplete type 'ns::X'"),
                WithFix(Fix(Test.range("insert"), "#include \"x.h\"\n",
                            "Add include \"x.h\" for symbol ns::X")))));
}

TEST(IncludeFixerTest, NoSuggestIncludeWhenNoDefinitionInHeader) {
  Annotations Test(R"cpp(
$insert[[]]namespace ns {
  class X;
}
class Y : $base[[public ns::X]] {};
int main() {
  ns::X *x;
  x$access[[->]]f();
}
  )cpp");
  auto TU = TestTU::withCode(Test.code());
  Symbol Sym = cls("ns::X");
  Sym.Flags |= Symbol::IndexedForCodeCompletion;
  Sym.CanonicalDeclaration.FileURI = "unittest:///x.h";
  Sym.Definition.FileURI = "unittest:///x.cc";
  Sym.IncludeHeaders.emplace_back("\"x.h\"", 1);

  SymbolSlab::Builder Slab;
  Slab.insert(Sym);
  auto Index = MemIndex::build(std::move(Slab).build(), RefSlab());
  TU.ExternalIndex = Index.get();

  EXPECT_THAT(TU.build().getDiagnostics(),
              UnorderedElementsAre(
                  Diag(Test.range("base"), "base class has incomplete type"),
                  Diag(Test.range("access"),
                       "member access into incomplete type 'ns::X'")));
}

} // namespace
} // namespace clangd
} // namespace clang

