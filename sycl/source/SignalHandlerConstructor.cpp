#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"

#include <cstdio>

static bool InitSignalHandler = []() {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  printf("Enabling signal handler");
  return true;
}();

void EnableHandler() {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  printf("Enabling signal handler");
}
