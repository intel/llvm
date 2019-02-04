//===--- Annotations.h - Annotated source code for tests ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotations lets you mark points and ranges inside source code, for tests:
//
//    Annotations Example(R"cpp(
//       int complete() { x.pri^ }          // ^ indicates a point
//       void err() { [["hello" == 42]]; }  // [[this is a range]]
//       $definition^class Foo{};           // points can be named: "definition"
//       $fail[[static_assert(false, "")]]  // ranges can be named too: "fail"
//    )cpp");
//
//    StringRef Code = Example.code();              // annotations stripped.
//    std::vector<Position> PP = Example.points();  // all unnamed points
//    Position P = Example.point();                 // there must be exactly one
//    Range R = Example.range("fail");              // find named ranges
//
// Points/ranges are coordinates into `code()` which is stripped of annotations.
//
// Ranges may be nested (and points can be inside ranges), but there's no way
// to define general overlapping ranges.
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H

#include "Protocol.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace clangd {

class Annotations {
public:
  // Parses the annotations from Text. Crashes if it's malformed.
  Annotations(llvm::StringRef Text);

  // The input text with all annotations stripped.
  // All points and ranges are relative to this stripped text.
  llvm::StringRef code() const { return Code; }

  // Returns the position of the point marked by ^ (or $name^) in the text.
  // Crashes if there isn't exactly one.
  Position point(llvm::StringRef Name = "") const;
  // Returns the position of all points marked by ^ (or $name^) in the text.
  std::vector<Position> points(llvm::StringRef Name = "") const;

  // Returns the location of the range marked by [[ ]] (or $name[[ ]]).
  // Crashes if there isn't exactly one.
  Range range(llvm::StringRef Name = "") const;
  // Returns the location of all ranges marked by [[ ]] (or $name[[ ]]).
  std::vector<Range> ranges(llvm::StringRef Name = "") const;

private:
  std::string Code;
  llvm::StringMap<llvm::SmallVector<Position, 1>> Points;
  llvm::StringMap<llvm::SmallVector<Range, 1>> Ranges;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H
