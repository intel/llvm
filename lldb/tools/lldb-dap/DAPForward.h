//===-- DAPForward.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPFORWARD_H
#define LLDB_TOOLS_LLDB_DAP_DAPFORWARD_H

// IWYU pragma: begin_exports

namespace lldb_dap {
struct BreakpointBase;
struct ExceptionBreakpoint;
struct FunctionBreakpoint;
struct SourceBreakpoint;
struct Watchpoint;
struct InstructionBreakpoint;
struct DAP;
class BaseRequestHandler;
class ResponseHandler;
} // namespace lldb_dap

namespace lldb {
class SBAttachInfo;
class SBBreakpoint;
class SBBreakpointLocation;
class SBCommandInterpreter;
class SBCommandReturnObject;
class SBCommunication;
class SBDebugger;
class SBEvent;
class SBFrame;
class SBHostOS;
class SBInstruction;
class SBInstructionList;
class SBLanguageRuntime;
class SBLaunchInfo;
class SBLineEntry;
class SBListener;
class SBModule;
class SBProcess;
class SBStream;
class SBStringList;
class SBTarget;
class SBThread;
class SBValue;
class SBWatchpoint;
} // namespace lldb

namespace llvm {
namespace json {
class Object;
} // namespace json
} // namespace llvm

// IWYU pragma: end_exports

#endif
