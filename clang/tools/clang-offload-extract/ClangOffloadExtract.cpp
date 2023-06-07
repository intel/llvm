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
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#define IMAGE_INFO_SECTION_NAME ".tgtimg"

using namespace llvm;
using namespace llvm::object;

// Create a category to label utility-specific options; This will allow
// us to distinguish those specific options from generic options and
// irrelevant options
static cl::OptionCategory
    ClangOffloadExtractCategory("Utility-specific options");

// Create options for the input and output files, each with an
// appropriate default when not specified
static cl::opt<std::string> Input(cl::Positional, cl::init("a.out"),
                                  cl::desc("<input file>"),
                                  cl::cat(ClangOffloadExtractCategory));

static cl::opt<std::string>
    FileNameStem("stem", cl::init("target.bin"),
                 cl::desc(
                     R"(Specifies the stem for the output file(s).
The default stem when not specified is "target.bin".
The Output file name is composed from this stem and
the sequential number of each extracted image appended
to the stem:
  <stem>.<index>
      )"),
                 cl::cat(ClangOffloadExtractCategory));

// Create an alias for the deprecated option, so legacy use still works
static cl::alias FileNameStemAlias(
    "output", cl::desc("Deprecated option, replaced by option '--stem'"),
    cl::aliasopt(FileNameStem), cl::cat(ClangOffloadExtractCategory));

// Path to the current binary
static std::string ToolPath;

// Report error (and handle any deferred errors)
static void reportError(Error E, Twine message = "\n") {
  std::string S;
  raw_string_ostream OSS(S);
  logAllUnhandledErrors(std::move(E), OSS);

  errs() << raw_ostream::RED                         //
         << formatv("{0,-10}", "error")              //
         << raw_ostream::RESET                       //
         << S                                        //
         << formatv("{0}", fmt_pad(message, 10, 0)); //
  exit(1);
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  ToolPath = argv[0];

  // Hide non-generic options that are not in this utility's explicit
  // category;
  // Some include files bring in options that are not relevant for the
  // public interface of this utility (e.g. color handling or Intermediate
  // Representation objects)
  cl::HideUnrelatedOptions(ClangOffloadExtractCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-extract") << "\n";
  });
  cl::ParseCommandLineOptions(argc, argv,
                              R"(

A utility to extract all the target images from a
linked fat binary, and store them in separate files.
)",
                              nullptr, nullptr, true);

  // Read input file. It should have one of the supported object file
  // formats:
  // * Common Object File Format (COFF) : https://wiki.osdev.org/COFF
  // * Executable Linker Format (ELF)   : https://wiki.osdev.org/ELF
  // This utility works on a hierarchy of objects:
  // =                           OwningBinary<ObjectFile>   ObjectOrError
  // |_->getBinary()             ObjectFile                 *Binary
  //   |_->getBytesInAddress()   uint8_t                    -
  //   |_->section_end()         section_iterator           -
  //   |_->sections()            section_iterator_range     -
  //     |_:                     SectionRef                 Section
  //       |_.getName()          StringRef                  -
  //       |_.getContents()      StringRef                  -
  //       |_.isData()           bool                       -
  //       |_.getAddress()       uint64_t                   -
  //       |_.getSize()          uint64_t                   -
  Expected<OwningBinary<ObjectFile>> ObjectOrErr =
      ObjectFile::createObjectFile(Input);
  if (auto E = ObjectOrErr.takeError()) {
    reportError(std::move(E), "Input File: '" + Input + "'\n");
  }

  // LLVM::ObjectFile has no constructor, but we can extract it from the
  // LLVM::OwningBinary object
  ObjectFile *Binary = ObjectOrErr->getBinary();

  // Bitness       :  sizeof(void *)
  // 32-bit systems:  4
  // 64-bit systems:  8
  if (!(isa<ELF64LEObjectFile>(Binary) || isa<COFFObjectFile>(Binary)) //
      || Binary->getBytesInAddress() != sizeof(void *)                 //
  ) {
    reportError(
        createStringError(errc::invalid_argument,
                          "Only 64-bit ELF or COFF inputs are supported"),
        "Input File: '" + Input + "'");
  }

  // We are dealing with an appropriate fat binary;
  // Locate the section IMAGE_INFO_SECTION_NAME (which contains the
  // metadata on the embedded binaries)
  unsigned FileNum = 0;

  for (const auto &Section : Binary->sections()) {
    Expected<StringRef> NameOrErr = Section.getName();
    if (auto E = NameOrErr.takeError()) {
      reportError(std::move(E), "Input File: '" + Input + "'\n");
    }
    if (*NameOrErr != IMAGE_INFO_SECTION_NAME)
      continue;

    // This is the section we are looking for;
    // Extract the meta information:
    // The IMAGE_INFO_SECTION_NAME section contains packed <address,
    // size> pairs describing target images that are stored in the fat
    // binary.
    Expected<StringRef> DataOrErr = Section.getContents();
    if (auto E = DataOrErr.takeError()) {
      reportError(std::move(E), "Input File: '" + Input + "'\n");
    }
    // Data type to store the metadata for an individual target image
    struct ImgInfoType {
      uintptr_t Addr;
      uintptr_t Size;
    };

    // Store the metadata for all target images in an array of target
    // image information descriptors
    auto ImgInfo =
        ArrayRef(reinterpret_cast<const ImgInfoType *>(DataOrErr->data()),
                 DataOrErr->size() / sizeof(ImgInfoType));

    //  Loop over the image information descriptors to extract each
    // target image.
    for (const auto &Img : ImgInfo) {
      // Ignore zero padding that can be inserted by the linker.
      if (!Img.Addr)
        continue;

      // Find section which contains this image.
      // TODO: can use more efficient algorithm than linear search. For
      // example sections and images could be sorted by address then one pass
      // performed through both at the same time.
      // std::find_if
      // * searches for a true predicate in [first,last] =~ [first,end)
      // * returns end if no predicate is true
      // It is probably faster to track  success through a bool (ImgFound)
      bool ImgFound = false;
      auto ImgSec =
          find_if(Binary->sections(), [&Img, &ImgFound](SectionRef Sec) {
            bool pred = (                         //
                Sec.isData()                      //
                && (Img.Addr == Sec.getAddress()) //
                && (Img.Size == Sec.getSize())    //
            );
            ImgFound = ImgFound || pred;
            return pred;
          });
      if (!ImgFound) {

        reportError(
            createStringError(
                inconvertibleErrorCode(),
                "cannot find section containing <0x%lx, 0x%lx> target image",
                Img.Addr, Img.Size),
            "Input File: '" + Input + "'\n");
      }

      Expected<StringRef> SecDataOrErr = ImgSec->getContents();
      if (auto E = SecDataOrErr.takeError()) {
        reportError(std::move(E), "Input File: '" + Input + "'\n");
      }

      // Output file name is composed from the name prefix provided by the
      // user and the image number which is appended to the prefix
      std::string FileName = FileNameStem + "." + std::to_string(FileNum++);

      // Tell user that we are saving an image.
      outs() << "Saving target image to '" << FileName << "'\n";

      // Write image data to the output
      std::error_code EC;
      raw_fd_ostream OS(FileName, EC);
      if (EC) {
        reportError(createFileError(FileName, EC),
                    "Specify a different Output File ('--stem' option)\n");
      }

      OS << SecDataOrErr->substr(Img.Addr - ImgSec->getAddress(), Img.Size);
      if (OS.has_error()) {
        reportError(createFileError(FileName, OS.error()),
                    "Try a different Output File ('--stem' option)");
      }
    } // &Img: ImgInfo

    // Fat binary is not expected to have more than one .tgtimg section.
    break;
  }

  return 0;
}
