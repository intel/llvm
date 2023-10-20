#if HAVE_LLVM > 0x0390
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#else
#include "llvm/Bitcode/ReaderWriter.h"
#endif

#include "llvm/Config/llvm-config.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

using namespace llvm;

static ExitOnError ExitOnErr;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::opt<bool> TextualOut("S", cl::desc("Emit LLVM textual assembly"),
                                cl::init(false));

int main(int argc, char **argv) {
  LLVMContext Context;
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "libclc builtin preparation tool\n");

  std::string ErrorMessage;
  Module *M = nullptr;

  {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    if (std::error_code  ec = BufferOrErr.getError()) {
      ErrorMessage = ec.message();
    } else {
      std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
      SMDiagnostic Err;
      std::unique_ptr<llvm::Module> MPtr =
#if HAVE_LLVM > 0x0390
          ExitOnErr(Expected<std::unique_ptr<llvm::Module>>(
              parseIR(BufferPtr.get()->getMemBufferRef(), Err, Context)));
#else
          parseIR(BufferPtr.get()->getMemBufferRef(), Err, Context);
#endif
      M = MPtr.release();
    }
  }

  if (!M) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

  // Strip the OpenCL version metadata. There are a lot of linked
  // modules in the library build, each spamming the same
  // version. This may also report a different version than the user
  // program is using. This should probably be uniqued when linking.
  if (NamedMDNode *OCLVersion = M->getNamedMetadata("opencl.ocl.version"))
      M->eraseNamedMetadata(OCLVersion);

  // wchar_size flag can cause a mismatch between libclc libraries and
  // modules using them. Since wchar is not used by libclc we drop the flag
  if (M->getModuleFlag("wchar_size")) {
    SmallVector<Module::ModuleFlagEntry, 4> ModuleFlags;
    M->getModuleFlagsMetadata(ModuleFlags);
    M->getModuleFlagsMetadata()->clearOperands();
    for (const Module::ModuleFlagEntry ModuleFlag : ModuleFlags)
      if (ModuleFlag.Key->getString() != "wchar_size")
        M->addModuleFlag(ModuleFlag.Behavior, ModuleFlag.Key->getString(),
                         ModuleFlag.Val);
  }

  // Set linkage of every external definition to linkonce_odr.
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  for (Module::global_iterator i = M->global_begin(), e = M->global_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  // AMDGPU remove incompatible functions pass replaces all uses of functions
  // that use GPU features incompatible with the current GPU with null then
  // deletes the function. This didn't use to cause problems, as all of libclc
  // functions were inlined prior to incompatible functions pass. Now that the
  // inliner runs later in the pipeline we have to remove all of the target
  // features, so libclc functions will not be earmarked for deletion.
  if (M->getTargetTriple().find("amdgcn") != std::string::npos) {
    AttributeMask AM;
    AM.addAttribute("target-features");
    AM.addAttribute("target-cpu");
    for (auto &F : *M)
      F.removeFnAttrs(AM);
  }

  std::error_code EC;
#if HAVE_LLVM >= 0x0600
  std::unique_ptr<ToolOutputFile> Out(
      new ToolOutputFile(OutputFilename, EC, sys::fs::OF_None));
#else
  std::unique_ptr<tool_output_file> Out(
      new tool_output_file(OutputFilename, EC, sys::fs::OF_None));
#endif
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  if (TextualOut)
    M->print(Out->os(), nullptr, true);
  else
#if HAVE_LLVM >= 0x0700
    WriteBitcodeToFile(*M, Out->os());
#else
    WriteBitcodeToFile(M, Out->os());
#endif

  // Declare success.
  Out->keep();
  return 0;
}
