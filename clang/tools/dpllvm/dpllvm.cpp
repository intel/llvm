//===-- dpllvm.cpp - DPC++ tool entry-point shim --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The DPC++ toolchain ships public entry-points whose names are prefixed with
// "dp" (dpclang, dpclang++, dpclang-cl, dpclang-cpp, dpsycl-ls, ...) so they
// don't collide with the identically-named tools of a system LLVM/Clang
// installation that may be on PATH.  Those entry-points are symlinks (real
// copies on Windows, which has no first-class symlinks) that resolve to this
// tool.
//
// dpllvm looks at how it was invoked (argv[0]), strips the leading "dp", and
// re-executes the correspondingly-named real binary that lives next to it in
// the same directory -- e.g. invoking "dpclang++" runs "clang++".  The real
// binary keeps its original name ("clang++", not "dpclang++") because the
// clang driver re-invokes itself by name (for -cc1, offloading sub-jobs, etc.)
// and must find a binary called "clang"/"clang++".
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

int main(int argc, char *argv[]) {
  using namespace llvm;
  using namespace llvm::sys;

  StringRef Executable = argv[0];
  StringRef Alias = path::filename(Executable);

  ExitOnError Exit((Alias + ": ").str());

  // Strip the "dp" prefix off the full filename to get the real tool's name.
  // Keep any extension: on Windows the entry-points are "dpclang++.exe" copies
  // of this tool and the real binary is "clang++.exe", so the extension must
  // be preserved for the sibling lookup (and for ExecuteAndWait) to succeed.
  StringRef RealName = Alias;
  if (!RealName.consume_front("dp"))
    Exit(createStringError("binary '" + Alias + "' not prefixed by 'dp'."));

  // Locate the directory this tool was installed into so we can find the real
  // binary sitting next to it, regardless of the current working directory or
  // how the tool was found on PATH.
  void *MainAddr = reinterpret_cast<void *>(main);
  std::string DpllvmPath = fs::getMainExecutable(argv[0], MainAddr);
  if (DpllvmPath.empty())
    Exit(createStringError(
        "couldn't determine the path to the DPC++ bin/ directory."));

  StringRef BinaryDir = path::parent_path(DpllvmPath);

  SmallString<256> BinaryPath(BinaryDir);
  path::append(BinaryPath, RealName);

  if (!fs::exists(BinaryPath))
    Exit(createStringError("binary '" + BinaryPath + "' does not exist."));

  SmallVector<StringRef, 128> Args = {BinaryPath};
  Args.append(argv + 1, argv + argc);

  std::string ErrMsg;
  int Result = ExecuteAndWait(BinaryPath, Args, /*Env=*/std::nullopt,
                              /*Redirects=*/{}, /*SecondsToWait=*/0,
                              /*MemoryLimit=*/0, &ErrMsg);
  if (!ErrMsg.empty())
    Exit(
        createStringError("failed to execute '" + BinaryPath + "': " + ErrMsg));
  return Result;
}
