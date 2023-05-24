//===-------- clang-offload-extract/ClangOffloadExtract.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the clang-offload-extract tool which allows extracting
// target images from linked fat offload binaries. For locating target images
// in the binary it uses information from the .tgtimg section which is added to
// the image by the clang-offload-wrapper tool. This section contains <address,
// size> pairs for all embedded target images.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#define IMAGE_INFO_SECTION_NAME ".tgtimg"

using namespace llvm;
using namespace llvm::object;

// Create a category to label the custom options separate from the
// builtin options
static cl::OptionCategory
    ClangOffloadExtractCategory("Utility-specific options");

// Create options for the input and output files, each with an
// appropriate default when not specified
static cl::opt<std::string> Input(cl::Positional, cl::init("a.out"),
                                  cl::desc("<input file>"),
                                  cl::cat(ClangOffloadExtractCategory));

static cl::opt<std::string>
    FileNameStem("stem", cl::init("target.exe"),
                 cl::desc(
                     R"(Specifies the stem for the output file(s).
The Output file name is composed from this stem and
the sequential number of each extracted image appended
to the stem:
  <stem>.<index>
      )"),
                 cl::cat(ClangOffloadExtractCategory));

// Since this wasn't productized before, I don't think we need an alias
// to deprecate the old options
// static cl::alias FileNameStemAlias(
//     "output", cl::desc("Deprecated option, replaced by option '--stem'"),
//     cl::aliasopt(FileNameStem), cl::cat(ClangOffloadExtractCategory));

// Path to the current binary
static std::string ToolPath;

// Report all accumulated yet unhandled errors
static void reportError(Error E) {
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), ToolPath));
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  ToolPath = argv[0];

  cl::HideUnrelatedOptions(ClangOffloadExtractCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-extract") << "\n";
  });
  cl::ParseCommandLineOptions(argc, argv,
                              R"(
A utility to extract all the target images from a
linked fat binary, and store them in separate files.
)");

  // Read input file. It should have one of the supported object file formats.
  Expected<OwningBinary<ObjectFile>> ObjectOrErr =
      ObjectFile::createObjectFile(Input);
  if (!ObjectOrErr) {
    reportError(ObjectOrErr.takeError());
    ObjectFile *Binary = ObjectOrErr->getBinary();

    // Do we plan to support 32-bit offload binaries?
    if (!(isa<ELF64LEObjectFile>(Binary) || isa<COFFObjectFile>(Binary)) ||
        Binary->getBytesInAddress() != sizeof(void *)) {
      reportError(
          createStringError(errc::invalid_argument,
                            "only 64-bit ELF or COFF inputs are supported"));
      return 1;
    }

    unsigned FileNum = 0;

    for (SectionRef Section : Binary->sections()) {
      // Look for the .tgtimg section in the binary.
      Expected<StringRef> NameOrErr = Section.getName();
      if (!NameOrErr) {
        reportError(NameOrErr.takeError());
        return 1;
      }
      if (*NameOrErr != IMAGE_INFO_SECTION_NAME)
        continue;

      // This is the section we are looking for.
      Expected<StringRef> DataOrErr = Section.getContents();
      if (!DataOrErr) {
        reportError(DataOrErr.takeError());
        return 1;
      }

      // This section contains concatenated <address, size> pairs describing
      // target images that are stored in the binary. Loop over these
      // descriptors and extract each target image.
      struct ImgInfoTy {
        uintptr_t Addr;
        uintptr_t Size;
      };

      auto ImgInfo = makeArrayRef<ImgInfoTy>(
          reinterpret_cast<const ImgInfoTy *>(DataOrErr->data()),
          DataOrErr->size() / sizeof(ImgInfoTy));

      for (auto &Img : ImgInfo) {
        // Ignore zero padding that can be inserted by the linker.
        if (!Img.Addr)
          continue;

        // Find section which contains this image.
        // TODO: can use more efficient algorithm than linear search. For
        // example sections and images could be sorted by address then one pass
        // performed through both at the same time.
        auto ImgSec = find_if(Binary->sections(), [&Img](SectionRef Sec) {
          if (!Sec.isData())
            return false;
          if (Img.Addr < Sec.getAddress() ||
              Img.Addr + Img.Size > Sec.getAddress() + Sec.getSize())
            return false;
          return true;
        });
        if (ImgSec == Binary->section_end()) {
          reportError(createStringError(
              inconvertibleErrorCode(),
              "cannot find section containing <0x%lx, 0x%lx> target image",
              Img.Addr, Img.Size));
          return 1;
        }

        Expected<StringRef> SecDataOrErr = ImgSec->getContents();
        if (!SecDataOrErr) {
          reportError(SecDataOrErr.takeError());
          return 1;
        }

        // Output file name is composed from the name prefix provided by the
        // user and the image number which is appended to the prefix.
        std::string FileName = FileNameStem + "." + std::to_string(FileNum++);

        // Tell user that we are saving an image.
        outs() << "Saving target image to \"" << FileName << "\"\n";

        // And write image data to the output.
        std::error_code EC;
        raw_fd_ostream OS(FileName, EC);
        if (EC) {
          reportError(createFileError(FileName, EC));
          return 1;
        }

        OS << SecDataOrErr->substr(Img.Addr - ImgSec->getAddress(), Img.Size);
        if (OS.has_error()) {
          reportError(createFileError(FileName, OS.error()));
          return 1;
        }
      }

      // Binary is not expected to have more than one .tgtimg section.
      break;
    }
    return 0;
    break;
  }
  return 0;
}
