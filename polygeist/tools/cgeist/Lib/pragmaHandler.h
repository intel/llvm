//===- pragmaHandler.ch - Pragmas used to emit MLIR---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H
#define MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"

/// POD holds information processed from the lower_to pragma.
struct LowerToInfo {
  llvm::StringMap<std::string> SymbolTable;
  llvm::SmallVector<llvm::StringRef, 2> InputSymbol;
  llvm::SmallVector<llvm::StringRef, 2> OutputSymbol;
};

/// The location of the scop, as delimited by scop and endscop
/// pragmas by the user.
/// "scop" and "endscop" are the source locations of the scop and
/// endscop pragmas.
/// "start_line" is the line number of the start position.
struct ScopLoc {
  ScopLoc() : End(0) {}

  clang::SourceLocation Scop;
  clang::SourceLocation EndScop;
  unsigned StartLine;
  unsigned Start;
  unsigned End;
};

/// Taken from pet.cc
/// List of pairs of #pragma scop and #pragma endscop locations.
struct ScopLocList {
  std::vector<ScopLoc> List;

  // Add a new start (#pragma scop) location to the list.
  // If the last #pragma scop did not have a matching
  // #pragma endscop then overwrite it.
  // "start" points to the location of the scop pragma.

  void addStart(clang::SourceManager &SM, clang::SourceLocation Start) {
    ScopLoc Loc;

    Loc.Scop = Start;
    int Line = SM.getExpansionLineNumber(Start);
    Start = SM.translateLineCol(SM.getFileID(Start), Line, 1);
    Loc.StartLine = Line;
    Loc.Start = SM.getFileOffset(Start);
    if (List.size() == 0 || List[List.size() - 1].End != 0)
      List.push_back(Loc);
    else
      List[List.size() - 1] = Loc;
  }

  // Set the end location (#pragma endscop) of the last pair
  // in the list.
  // If there is no such pair of if the end of that pair
  // is already set, then ignore the spurious #pragma endscop.
  // "end" points to the location of the endscop pragma.

  void addEnd(clang::SourceManager &SM, clang::SourceLocation End) {
    if (List.size() == 0 || List[List.size() - 1].End != 0)
      return;
    List[List.size() - 1].EndScop = End;
    int Line = SM.getExpansionLineNumber(End);
    End = SM.translateLineCol(SM.getFileID(End), Line + 1, 1);
    List[List.size() - 1].End = SM.getFileOffset(End);
  }

  // Check if the current location is in the scop.
  bool isInScop(clang::SourceLocation Target) {
    if (!List.size())
      return false;
    for (auto &ScopLoc : List)
      if ((Target >= ScopLoc.Scop) && (Target <= ScopLoc.EndScop))
        return true;
    return false;
  }
};

void addPragmaLowerToHandlers(clang::Preprocessor &PP, LowerToInfo &LTInfo);
void addPragmaScopHandlers(clang::Preprocessor &PP, ScopLocList &ScopLocList);
void addPragmaEndScopHandlers(clang::Preprocessor &PP,
                              ScopLocList &ScopLocList);

#endif // MLIR_TOOLS_MLIRCLANG_LIB_PRAGMAHANDLER_H
