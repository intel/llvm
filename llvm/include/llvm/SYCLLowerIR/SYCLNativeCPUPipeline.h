#include "llvm/Target/TargetMachine.h"

namespace llvm {
void addSYCLNativeCPUBackendPasses(ModulePassManager& MPM, ModuleAnalysisManager& MAM);
} // namespace llvm
