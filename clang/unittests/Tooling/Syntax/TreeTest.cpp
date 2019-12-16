//===- TreeTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Tree.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Syntax/BuildTree.h"
#include "clang/Tooling/Syntax/Nodes.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace clang;

namespace {
class SyntaxTreeTest : public ::testing::Test {
protected:
  // Build a syntax tree for the code.
  syntax::TranslationUnit *buildTree(llvm::StringRef Code) {
    // FIXME: this code is almost the identical to the one in TokensTest. Share
    //        it.
    class BuildSyntaxTree : public ASTConsumer {
    public:
      BuildSyntaxTree(syntax::TranslationUnit *&Root,
                      std::unique_ptr<syntax::Arena> &Arena,
                      std::unique_ptr<syntax::TokenCollector> Tokens)
          : Root(Root), Arena(Arena), Tokens(std::move(Tokens)) {
        assert(this->Tokens);
      }

      void HandleTranslationUnit(ASTContext &Ctx) override {
        Arena = std::make_unique<syntax::Arena>(Ctx.getSourceManager(),
                                                Ctx.getLangOpts(),
                                                std::move(*Tokens).consume());
        Tokens = nullptr; // make sure we fail if this gets called twice.
        Root = syntax::buildSyntaxTree(*Arena, *Ctx.getTranslationUnitDecl());
      }

    private:
      syntax::TranslationUnit *&Root;
      std::unique_ptr<syntax::Arena> &Arena;
      std::unique_ptr<syntax::TokenCollector> Tokens;
    };

    class BuildSyntaxTreeAction : public ASTFrontendAction {
    public:
      BuildSyntaxTreeAction(syntax::TranslationUnit *&Root,
                            std::unique_ptr<syntax::Arena> &Arena)
          : Root(Root), Arena(Arena) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        // We start recording the tokens, ast consumer will take on the result.
        auto Tokens =
            std::make_unique<syntax::TokenCollector>(CI.getPreprocessor());
        return std::make_unique<BuildSyntaxTree>(Root, Arena,
                                                 std::move(Tokens));
      }

    private:
      syntax::TranslationUnit *&Root;
      std::unique_ptr<syntax::Arena> &Arena;
    };

    constexpr const char *FileName = "./input.cpp";
    FS->addFile(FileName, time_t(), llvm::MemoryBuffer::getMemBufferCopy(""));
    if (!Diags->getClient())
      Diags->setClient(new IgnoringDiagConsumer);
    // Prepare to run a compiler.
    std::vector<const char *> Args = {"syntax-test", "-std=c++11",
                                      "-fsyntax-only", FileName};
    auto CI = createInvocationFromCommandLine(Args, Diags, FS);
    assert(CI);
    CI->getFrontendOpts().DisableFree = false;
    CI->getPreprocessorOpts().addRemappedFile(
        FileName, llvm::MemoryBuffer::getMemBufferCopy(Code).release());
    CompilerInstance Compiler;
    Compiler.setInvocation(std::move(CI));
    Compiler.setDiagnostics(Diags.get());
    Compiler.setFileManager(FileMgr.get());
    Compiler.setSourceManager(SourceMgr.get());

    syntax::TranslationUnit *Root = nullptr;
    BuildSyntaxTreeAction Recorder(Root, this->Arena);
    if (!Compiler.ExecuteAction(Recorder)) {
      ADD_FAILURE() << "failed to run the frontend";
      std::abort();
    }
    return Root;
  }

  // Adds a file to the test VFS.
  void addFile(llvm::StringRef Path, llvm::StringRef Contents) {
    if (!FS->addFile(Path, time_t(),
                     llvm::MemoryBuffer::getMemBufferCopy(Contents))) {
      ADD_FAILURE() << "could not add a file to VFS: " << Path;
    }
  }

  // Data fields.
  llvm::IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      new DiagnosticsEngine(new DiagnosticIDs, new DiagnosticOptions);
  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      new llvm::vfs::InMemoryFileSystem;
  llvm::IntrusiveRefCntPtr<FileManager> FileMgr =
      new FileManager(FileSystemOptions(), FS);
  llvm::IntrusiveRefCntPtr<SourceManager> SourceMgr =
      new SourceManager(*Diags, *FileMgr);
  // Set after calling buildTree().
  std::unique_ptr<syntax::Arena> Arena;
};

TEST_F(SyntaxTreeTest, Basic) {
  std::pair</*Input*/ std::string, /*Expected*/ std::string> Cases[] = {
      {
          R"cpp(
int main() {}
void foo() {}
    )cpp",
          R"txt(
*: TranslationUnit
|-SimpleDeclaration
| |-int
| |-main
| |-(
| |-)
| `-CompoundStatement
|   |-{
|   `-}
`-SimpleDeclaration
  |-void
  |-foo
  |-(
  |-)
  `-CompoundStatement
    |-{
    `-}
)txt"},
      // if.
      {
          R"cpp(
int main() {
  if (true) {}
  if (true) {} else if (false) {}
}
        )cpp",
          R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-main
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-true
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-true
    | |-)
    | |-CompoundStatement
    | | |-{
    | | `-}
    | |-else
    | `-IfStatement
    |   |-if
    |   |-(
    |   |-UnknownExpression
    |   | `-false
    |   |-)
    |   `-CompoundStatement
    |     |-{
    |     `-}
    `-}
        )txt"},
      // for.
      {R"cpp(
void test() {
  for (;;)  {}
}
)cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-ForStatement
    | |-for
    | |-(
    | |-;
    | |-;
    | |-)
    | `-CompoundStatement
    |   |-{
    |   `-}
    `-}
        )txt"},
      // declaration statement.
      {"void test() { int a = 10; }",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | |-a
    | | |-=
    | | `-UnknownExpression
    | |   `-10
    | `-;
    `-}
)txt"},
      {"void test() { ; }", R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-EmptyStatement
    | `-;
    `-}
)txt"},
      // switch, case and default.
      {R"cpp(
void test() {
  switch (true) {
    case 0:
    default:;
  }
}
)cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-SwitchStatement
    | |-switch
    | |-(
    | |-UnknownExpression
    | | `-true
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-CaseStatement
    |   | |-case
    |   | |-UnknownExpression
    |   | | `-0
    |   | |-:
    |   | `-DefaultStatement
    |   |   |-default
    |   |   |-:
    |   |   `-EmptyStatement
    |   |     `-;
    |   `-}
    `-}
)txt"},
      // while.
      {R"cpp(
void test() {
  while (true) { continue; break; }
}
)cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-WhileStatement
    | |-while
    | |-(
    | |-UnknownExpression
    | | `-true
    | |-)
    | `-CompoundStatement
    |   |-{
    |   |-ContinueStatement
    |   | |-continue
    |   | `-;
    |   |-BreakStatement
    |   | |-break
    |   | `-;
    |   `-}
    `-}
)txt"},
      // return.
      {R"cpp(
int test() { return 1; }
      )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-ReturnStatement
    | |-return
    | |-UnknownExpression
    | | `-1
    | `-;
    `-}
)txt"},
      // Range-based for.
      {R"cpp(
void test() {
  int a[3];
  for (int x : a) ;
}
      )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | |-a
    | | |-[
    | | |-UnknownExpression
    | | | `-3
    | | `-]
    | `-;
    |-RangeBasedForStatement
    | |-for
    | |-(
    | |-SimpleDeclaration
    | | |-int
    | | |-x
    | | `-:
    | |-UnknownExpression
    | | `-a
    | |-)
    | `-EmptyStatement
    |   `-;
    `-}
       )txt"},
      // Unhandled statements should end up as 'unknown statement'.
      // This example uses a 'label statement', which does not yet have a syntax
      // counterpart.
      {"void main() { foo: return 100; }", R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-main
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-UnknownStatement
    | |-foo
    | |-:
    | `-ReturnStatement
    |   |-return
    |   |-UnknownExpression
    |   | `-100
    |   `-;
    `-}
)txt"},
      // expressions should be wrapped in 'ExpressionStatement' when they appear
      // in a statement position.
      {R"cpp(
void test() {
  test();
  if (true) test(); else test();
}
    )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-test
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-ExpressionStatement
    | |-UnknownExpression
    | | |-test
    | | |-(
    | | `-)
    | `-;
    |-IfStatement
    | |-if
    | |-(
    | |-UnknownExpression
    | | `-true
    | |-)
    | |-ExpressionStatement
    | | |-UnknownExpression
    | | | |-test
    | | | |-(
    | | | `-)
    | | `-;
    | |-else
    | `-ExpressionStatement
    |   |-UnknownExpression
    |   | |-test
    |   | |-(
    |   | `-)
    |   `-;
    `-}
)txt"},
      // Multiple declarators group into a single SimpleDeclaration.
      {R"cpp(
      int *a, b;
  )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-int
  |-*
  |-a
  |-,
  |-b
  `-;
  )txt"},
      {R"cpp(
    typedef int *a, b;
  )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-typedef
  |-int
  |-*
  |-a
  |-,
  |-b
  `-;
  )txt"},
      // Multiple declarators inside a statement.
      {R"cpp(
void foo() {
      int *a, b;
      typedef int *ta, tb;
}
  )cpp",
       R"txt(
*: TranslationUnit
`-SimpleDeclaration
  |-void
  |-foo
  |-(
  |-)
  `-CompoundStatement
    |-{
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-int
    | | |-*
    | | |-a
    | | |-,
    | | `-b
    | `-;
    |-DeclarationStatement
    | |-SimpleDeclaration
    | | |-typedef
    | | |-int
    | | |-*
    | | |-ta
    | | |-,
    | | `-tb
    | `-;
    `-}
  )txt"},
      {R"cpp(
namespace a { namespace b {} }
namespace a::b {}
namespace {}

namespace foo = a;
    )cpp",
       R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-a
| |-{
| |-NamespaceDefinition
| | |-namespace
| | |-b
| | |-{
| | `-}
| `-}
|-NamespaceDefinition
| |-namespace
| |-a
| |-::
| |-b
| |-{
| `-}
|-NamespaceDefinition
| |-namespace
| |-{
| `-}
`-NamespaceAliasDefinition
  |-namespace
  |-foo
  |-=
  |-a
  `-;
)txt"},
      {R"cpp(
namespace ns {}
using namespace ::ns;
    )cpp",
       R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| `-}
`-UsingNamespaceDirective
  |-using
  |-namespace
  |-::
  |-ns
  `-;
       )txt"},
      {R"cpp(
namespace ns { int a; }
using ns::a;
    )cpp",
       R"txt(
*: TranslationUnit
|-NamespaceDefinition
| |-namespace
| |-ns
| |-{
| |-SimpleDeclaration
| | |-int
| | |-a
| | `-;
| `-}
`-UsingDeclaration
  |-using
  |-ns
  |-::
  |-a
  `-;
       )txt"},
      {R"cpp(
template <class T> struct X {
  using T::foo;
  using typename T::bar;
};
    )cpp",
       R"txt(
*: TranslationUnit
`-UnknownDeclaration
  |-template
  |-<
  |-UnknownDeclaration
  | |-class
  | `-T
  |->
  |-struct
  |-X
  |-{
  |-UsingDeclaration
  | |-using
  | |-T
  | |-::
  | |-foo
  | `-;
  |-UsingDeclaration
  | |-using
  | |-typename
  | |-T
  | |-::
  | |-bar
  | `-;
  |-}
  `-;
       )txt"},
      {R"cpp(
using type = int;
    )cpp",
       R"txt(
*: TranslationUnit
`-TypeAliasDeclaration
  |-using
  |-type
  |-=
  |-int
  `-;
       )txt"},
      {R"cpp(
;
    )cpp",
       R"txt(
*: TranslationUnit
`-EmptyDeclaration
  `-;
       )txt"},
      {R"cpp(
static_assert(true, "message");
static_assert(true);
    )cpp",
       R"txt(
*: TranslationUnit
|-StaticAssertDeclaration
| |-static_assert
| |-(
| |-UnknownExpression
| | `-true
| |-,
| |-UnknownExpression
| | `-"message"
| |-)
| `-;
`-StaticAssertDeclaration
  |-static_assert
  |-(
  |-UnknownExpression
  | `-true
  |-)
  `-;
       )txt"},
      {R"cpp(
extern "C" int a;
extern "C" { int b; int c; }
    )cpp",
       R"txt(
*: TranslationUnit
|-LinkageSpecificationDeclaration
| |-extern
| |-"C"
| `-SimpleDeclaration
|   |-int
|   |-a
|   `-;
`-LinkageSpecificationDeclaration
  |-extern
  |-"C"
  |-{
  |-SimpleDeclaration
  | |-int
  | |-b
  | `-;
  |-SimpleDeclaration
  | |-int
  | |-c
  | `-;
  `-}
       )txt"},
  };

  for (const auto &T : Cases) {
    SCOPED_TRACE(T.first);

    auto *Root = buildTree(T.first);
    std::string Expected = llvm::StringRef(T.second).trim().str();
    std::string Actual = llvm::StringRef(Root->dump(*Arena)).trim();
    EXPECT_EQ(Expected, Actual) << "the resulting dump is:\n" << Actual;
  }
}
} // namespace
