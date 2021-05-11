//===- file-table-tform.cpp - transform files with tables of strings ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This tool transforms a series of input file tables into single output file
// table according to operations passed on the command line. Operations'
// arguments are input files, and some operations like 'rename' take single
// input, others like 'replace' take two. Operations are executed in
// command-line order and consume needed amount of inputs left-to-right in the
// command-line order. Table files and operation example:
//   $ cat a.txt
//   [Code|Symbols|Properties]
//   a_0.bc|a_0.sym|a_0.props
//   a_1.bc|a_1.sym|a_1.props
//
//   $ cat b.txt:
//   [Files|Attrs]
//   a_0.spv|aa.attr
//   a_1.spv|bb.attr
//
//   $ file-table-tform --replace=Code,Files a.txt b.txt -o c.txt
//
//   $ cat c.txt
//   [Code|Symbols|Properties]
//   a_0.spv|a_0.sym|a_0.props
//   a_1.spv|a_1.sym|a_1.props
//
// The tool for now supports only linear transformation sequences like shown on
// the graph below. 'op*' represent operations, 'Input' is the main input file,
// 'Output' is the single output file, edges are directed and designate inputs
// and outputs of the operations.
//      File1       File3
//         \          \
// Input - op1 - op2 - op3 - Output
//         /
//      File2
// More complex transformation trees such as:
//   File0 - op0       File3
//             \          \
//     Input - op1 - op2 - op3 - Output
//             /
//          File2
// are not supported. For now, "File0 - op0" transformation must be done in a
// separate tool invocation.
// TODO support SQL-like transformation style if the tool ever evolves.

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/WithColor.h"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>

using namespace llvm;

static StringRef ToolName; // set in main first thing

cl::OptionCategory FileTableTformCat{"file-table-tform Options"};

static cl::list<std::string> Inputs(cl::Positional, cl::ZeroOrMore,
                                    cl::desc("<input filenames>"),
                                    cl::cat(FileTableTformCat));

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("<output filename>"),
                                   cl::value_desc("filename"),
                                   cl::cat(FileTableTformCat));

static constexpr char OPT_REPLACE[] = "replace";
static constexpr char OPT_RENAME[] = "rename";
static constexpr char OPT_EXTRACT[] = "extract";

static cl::list<std::string> TformReplace{
    OPT_REPLACE, cl::ZeroOrMore, cl::desc("replace a column"),
    cl::value_desc("<column name or ordinal>"), cl::cat(FileTableTformCat)};

static cl::list<std::string> TformRename{
    OPT_RENAME, cl::ZeroOrMore, cl::desc("rename a column"),
    cl::value_desc("<old_name>,<new_name>"), cl::cat(FileTableTformCat)};

static cl::list<std::string> TformExtract{
    OPT_EXTRACT, cl::ZeroOrMore,
    cl::desc("extract column(s) identified by names"),
    cl::value_desc("<name1>,<name2>,..."), cl::cat(FileTableTformCat)};

static cl::opt<bool> DropTitles{"drop_titles", cl::Optional,
                                cl::desc("drop column titles"),
                                cl::cat(FileTableTformCat)};

Error makeToolError(Twine Msg) {
  return make_error<StringError>("*** " + llvm::Twine(ToolName) +
                                     " ERROR: " + Msg,
                                 inconvertibleErrorCode());
}

Error makeIOError(Twine Msg) {
  return make_error<StringError>(
      "*** " + Twine(ToolName) + " SYSTEM ERROR: " + Msg, errc::io_error);
}

Error makeUserError(Twine Msg) {
  return createStringError(errc::invalid_argument,
                           "*** " + Twine(ToolName) + " usage ERROR: " + Msg);
}

struct TformCmd {
  using UPtrTy = std::unique_ptr<TformCmd>;
  StringRef Kind;
  SmallVector<StringRef, 2> Args;
  SmallVector<StringRef, 2> Inputs;

  TformCmd() = default;
  TformCmd(StringRef Kind) : Kind(Kind) {}

  static Expected<UPtrTy> create(StringRef Kind, StringRef RawArg = "") {
    UPtrTy Res = std::make_unique<TformCmd>(Kind);
    Error E = Res->parseArg(RawArg);
    if (E)
      return std::move(E);
    return std::move(Res);
  }

  using InpIt = cl::list<std::string>::iterator;

  Error consumeSingleInput(InpIt &Cur, const InpIt End) {
    if (Cur == End)
      return makeUserError("no input for '" + Twine(Kind) + "' command");
    if (!llvm::sys::fs::exists(*Cur))
      return makeIOError("file not found: " + Twine(*Cur));
    Inputs.push_back(*Cur);
    Cur++;
    return Error::success();
  }

  using Func = std::function<Error(TformCmd *)>;

  Error consumeInput(InpIt Cur, const InpIt End) {
    Func F =
        StringSwitch<Func>(Kind)
            .Case(OPT_REPLACE,
                  [&](TformCmd *Cmd) {
                    return Cmd->consumeSingleInput(Cur, End);
                  })
            .Case(OPT_RENAME, [&](TformCmd *Cmd) { return Error::success(); })
            .Case(OPT_EXTRACT, [&](TformCmd *Cmd) { return Error::success(); });
    return F(this);
  }

  Error parseArg(StringRef Arg) {
    Func F =
        StringSwitch<Func>(Kind)
            // need '-> Error' return type declaration in the lambdas below as
            // it can't be deduced automatically
            .Case(OPT_REPLACE,
                  [&](TformCmd *Cmd) -> Error {
                    // argument is <column name>
                    if (Arg.empty())
                      return makeUserError("empty argument in " +
                                           Twine(OPT_REPLACE));
                    Arg.split(Args, ',');
                    if (Args.size() != 2 || Args[0].empty() || Args[1].empty())
                      return makeUserError("invalid argument in " +
                                           Twine(OPT_REPLACE));
                    return Error::success();
                  })
            .Case(OPT_RENAME,
                  [&](TformCmd *Cmd) -> Error {
                    // argument is <old_name>,<new_name>
                    if (Arg.empty())
                      return makeUserError("empty argument in " +
                                           Twine(OPT_RENAME));
                    auto Names = Arg.split(',');
                    if (Names.first.empty() || Names.second.empty())
                      return makeUserError("invalid argument in " +
                                           Twine(OPT_RENAME));
                    Args.push_back(Names.first);
                    Args.push_back(Names.second);
                    return Error::success();
                  })
            .Case(OPT_EXTRACT, [&](TformCmd *Cmd) -> Error {
              // argument is <name1>,<name2>,... (1 or more)
              if (Arg.empty())
                return makeUserError("empty argument in " + Twine(OPT_RENAME));
              SmallVector<StringRef, 3> Names;
              Arg.split(Names, ',');
              if (std::find(Names.begin(), Names.end(), "") != Names.end())
                return makeUserError("empty name in " + Twine(OPT_RENAME));
              std::copy(Names.begin(), Names.end(), std::back_inserter(Args));
              return Error::success();
            });
    return F(this);
  }

  Error execute(util::SimpleTable &Table) {
    Func F =
        StringSwitch<Func>(Kind)
            .Case(OPT_REPLACE,
                  [&](TformCmd *Cmd) -> Error {
                    // argument is <column name>
                    assert(Cmd->Args.size() == 2 && Cmd->Inputs.size() == 1);
                    Expected<util::SimpleTable::UPtrTy> Table1 =
                        util::SimpleTable::read(Cmd->Inputs[0]);
                    if (!Table1)
                      return Table1.takeError();
                    Error Res =
                        Table.replaceColumn(Args[0], *Table1->get(), Args[1]);
                    return Res ? std::move(Res) : std::move(Error::success());
                  })
            .Case(OPT_RENAME,
                  [&](TformCmd *Cmd) -> Error {
                    // argument is <old_name>,<new_name>
                    assert(Args.size() == 2);
                    Error Res = Table.renameColumn(Args[0], Args[1]);
                    return Res ? std::move(Res) : std::move(Error::success());
                  })
            .Case(OPT_EXTRACT, [&](TformCmd *Cmd) -> Error {
              // argument is <name1>,<name2>,... (1 or more)
              assert(!Args.empty());
              Error Res = Table.peelColumns(Args);
              return Res ? std::move(Res) : std::move(Error::success());
            });
    return F(this);
  }
};

#define CHECK_AND_EXIT(E)                                                      \
  {                                                                            \
    Error LocE = E;                                                            \
    if (LocE) {                                                                \
      logAllUnhandledErrors(std::move(LocE), WithColor::error(errs()));        \
      return 1;                                                                \
    }                                                                          \
  }

int main(int argc, char **argv) {
  ToolName = argv[0]; // make tool name available for functions in this source
  InitLLVM X{argc, argv};

  cl::HideUnrelatedOptions(FileTableTformCat);
  cl::ParseCommandLineOptions(
      argc, argv,
      "File table transformation tool.\n"
      "Inputs and output of this tool is a \"file table\" files containing\n"
      "2D table of strings with optional row of column titles. Based on\n"
      "transformation actions passed via the command line, the tool "
      "transforms the first input file table and emits a new one as a result.\n"
      "\n"
      "Transformation actions are:\n"
      "- replace a column\n"
      "- rename a column\n"
      "- extract column(s)\n");

  std::map<int, TformCmd::UPtrTy> Cmds;

  // Partially construct commands (w/o input information). Can't fully construct
  // yet, as an order across all command line options-commands needs to be
  // established first to properly map inputs to commands.

  auto Lists = {std::addressof(TformReplace), std::addressof(TformRename),
                std::addressof(TformExtract)};

  for (const auto *L : Lists) {
    for (auto It = L->begin(); It != L->end(); It++) {
      Expected<TformCmd::UPtrTy> Cmd = TformCmd::create(L->ArgStr, *It);

      if (!Cmd)
        CHECK_AND_EXIT(Cmd.takeError());
      const int Pos = L->getPosition(It - L->begin());
      Cmds.emplace(Pos, std::move(Cmd.get()));
    }
  }
  // finish command construction first w/o execution to make sure command line
  // is valid
  auto CurInput = Inputs.begin();
  const auto EndInput = Inputs.end();
  // first input is the "current" - it will undergo the transformation sequence
  if (CurInput == EndInput)
    CHECK_AND_EXIT(makeUserError("no inputs"));
  std::string &InputFile = *CurInput++;

  for (auto &P : Cmds) {
    TformCmd::UPtrTy &Cmd = P.second;
    // this will advance cur iterator as far as needed
    Error E = Cmd->consumeInput(CurInput, EndInput);
    CHECK_AND_EXIT(std::move(E));
  }
  // commands are constructed, command line is correct - read input and execute
  // transformations on it

  Expected<util::SimpleTable::UPtrTy> Table =
      util::SimpleTable::read(InputFile);
  if (!Table)
    CHECK_AND_EXIT(Table.takeError());

  for (auto &P : Cmds) {
    TformCmd::UPtrTy &Cmd = P.second;
    Error Res = Cmd->execute(*Table->get());
    CHECK_AND_EXIT(std::move(Res));
  }
  // Finally, write the result
  std::error_code EC;
  raw_fd_ostream Out{Output, EC, sys::fs::OpenFlags::OF_None};

  if (EC)
    CHECK_AND_EXIT(createFileError(Output, EC));
  Table->get()->write(Out, !DropTitles);

  if (Out.has_error())
    CHECK_AND_EXIT(createFileError(Output, Out.error()));
  Out.close();
  return 0;
}
