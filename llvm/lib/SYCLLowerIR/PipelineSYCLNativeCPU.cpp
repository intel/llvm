#include "llvm/SYCLLowerIR/SYCLNativeCPUPipeline.h"
#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/RenameKernelSYCLNativeCPU.h"

#ifdef NATIVECPU_USE_OCK
#include "compiler/utils/builtin_info.h"
#include "compiler/utils/sub_group_analysis.h"
#include "compiler/utils/work_item_loops_pass.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

namespace llvm {
void addSYCLNativeCPUBackendPasses(llvm::ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM) {
  MPM.addPass(ConvertToMuxBuiltinsSYCLNativeCPUPass());
#ifdef NATIVECPU_USE_OCK
  // Todo set options properly
  compiler::utils::WorkItemLoopsPassOptions Opts;
  Opts.IsDebug = false;
  Opts.ForceNoTail = false;
  MAM.registerPass([&] { return compiler::utils::BuiltinInfoAnalysis(); });
  MAM.registerPass([&] { return compiler::utils::SubgroupAnalysis(); });
  MPM.addPass(compiler::utils::WorkItemLoopsPass(Opts));
  MPM.addPass(AlwaysInlinerPass());

#endif
  MPM.addPass(PrepareSYCLNativeCPUPass());
  MPM.addPass(RenameKernelSYCLNativeCPUPass());
}
} // namespace llvm
