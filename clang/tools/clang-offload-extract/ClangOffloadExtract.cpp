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
#define IMAGE_SECTION_NAME_PREFIX "__CLANG_OFFLOAD_BUNDLE__"
// Windows truncates the names of sections to 8 bytes
#define IMAGE_SECTION_NAME_PREFIX_COFF "__CLANG_"

#define DEBUG_TYPE "clang-offload-extract"

using namespace llvm;
using namespace llvm::object;

// Command-line parsing
// Create a category to label utility-specific options; This will allow
// us to distinguish those specific options from generic options and
// irrelevant options
static cl::OptionCategory
    ClangOffloadExtractCategory("Utility-specific options");

// Create options for the input (fat binary) and output (target images)
// files, each with an appropriate default when not specified
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

static cl::opt<bool>
    Quiet("q", cl::init(false),
          cl::desc(R"(Do not print the names of generated files)"),
          cl::cat(ClangOffloadExtractCategory));

// Create an alias for the deprecated option, so legacy use still works
static cl::alias FileNameStemAlias(
    "output", cl::desc("Deprecated option, replaced by option '--stem'"),
    cl::aliasopt(FileNameStem), cl::cat(ClangOffloadExtractCategory));

// Report error (and handle any deferred errors)
// This is the main diagnostic used by the utility
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
  // * Create an array all the sections that have
  //   IMAGE_SECTION_NAME in the section name:
  auto OffloadSections = SmallVector<SectionRef>();
  // * Locate the section that starts with IMAGE_INFO_SECTION_NAME_PREFIX
  //   and extract the index for all the embedded binaries:
  auto OffloadIndex = SmallVector<SectionRef>();
  for (const auto &Section : Binary->sections()) {
    Expected<StringRef> InfoSecNameOrErr = Section.getName();
    if (auto E = InfoSecNameOrErr.takeError()) {
      reportError(std::move(E), "Input File: '" + Input + "'\n");
    }
    LLVM_DEBUG(dbgs() << "Section: " << *InfoSecNameOrErr << "\n");

    // We have a valid section name
    std::string SectionNameToCompare = isa<COFFObjectFile>(Binary)
                                           ? IMAGE_SECTION_NAME_PREFIX_COFF
                                           : IMAGE_SECTION_NAME_PREFIX;
    if (InfoSecNameOrErr->find(SectionNameToCompare) != std::string::npos) {
      // This section contains embedded binaries
      OffloadSections.push_back(Section);
    } else if (*InfoSecNameOrErr == IMAGE_INFO_SECTION_NAME) {
      // This section is the index
      OffloadIndex.push_back(Section);
    }
  }

  // Check if there are any sections with embedded binaries
  if (OffloadSections.size() == 0) {
    reportError(
        createStringError(inconvertibleErrorCode(),
                          "Could not locate sections with offload binaries"),
        "Fat Binary: '" + Input + "'\n");
  }
  // Check if we found the index section
  if (OffloadIndex.size() == 0) {
    reportError(
        createStringError(inconvertibleErrorCode(),
                          "Could not locate index for embedded binaries"),
        "Fat Binary: '" + Input + "'\n");
  }
  // Check if we have a valid index section
  Expected<StringRef> DataOrErr = OffloadIndex[0].getContents();
  if (auto E = DataOrErr.takeError()) {
    reportError(std::move(E), "Input File: '" + Input + "'\n");
  }

  // The input binary contains embedded offload binaries
  // The index section contains packed <address, size> pairs describing
  // target images that are stored in the fat binary.
  // Data type to store the index for an individual target image
  struct ImgInfoType {
    uintptr_t Addr;
    uintptr_t Size;
  };

  // Store the metadata for all target images in an array of target
  // image information descriptors
  // This can be done by reinterpreting the content of the section
  auto ImgInfo =
      ArrayRef(reinterpret_cast<const ImgInfoType *>(DataOrErr->data()),
               DataOrErr->size() / sizeof(ImgInfoType));

  //  Loop over the image information descriptors to extract each
  // target image the object file data
  unsigned FileNum = 0;
  unsigned ImgCnt = 1;

  for (const auto &Img : ImgInfo) {
    // Ignore zero padding that can be inserted by the linker.
    if (!Img.Addr)
      continue;

    // Find section which contains this image.
    // /!\ There might be multiple images in a section
    // std::find_if
    // * searches for a true predicate in [first,last] =~ [first,end)
    // * returns end if no predicate is true
    // It is probably faster to track  success through a bool (ImgFound)
    bool ImgFound = false;
    auto ImgSec = find_if(OffloadSections, [&Img, &ImgFound](auto Sec) {
      bool pred = (                                                        //
          Sec.isData()                                                     //
          && (Img.Addr >= Sec.getAddress())                                //
          && ((Img.Addr + Img.Size) <= (Sec.getAddress() + Sec.getSize())) //
      );
      ImgFound = ImgFound || pred;
      return pred;
    });
    if (!ImgFound) {

      reportError(
          createStringError(inconvertibleErrorCode(),
                            "Target image (Address:0x%lx, Size:0x%lx>) is "
                            "not contained in any section",
                            Img.Addr, Img.Size),
          "Fat Binary: '" + Input + "'\n");
    }

    Expected<StringRef> ImgSecNameOrErr = ImgSec->getName();
    if (auto E = ImgSecNameOrErr.takeError()) {
      reportError(std::move(E),
                  "Can not determine section name in Fat Binary '" + Input +
                      "'\n");
    }
    Expected<StringRef> SecDataOrErr = ImgSec->getContents();
    if (auto E = SecDataOrErr.takeError()) {
      reportError(std::move(E), "Can not extract contents for section '" +
                                    *ImgSecNameOrErr + "' in Fat Binary '" +
                                    Input + "'\n");
    }

    // Output file name is composed from the name prefix provided by the
    // user and the image number which is appended to the prefix
    std::string FileName = FileNameStem + "." + std::to_string(FileNum++);
    std::string OffloadName = ImgSecNameOrErr->data();
    std::string OffloadPrefix = isa<COFFObjectFile>(Binary)
                                    ? IMAGE_SECTION_NAME_PREFIX_COFF
                                    : IMAGE_SECTION_NAME_PREFIX;
    OffloadName.erase(0, OffloadPrefix.length());

    // Tell user that we are saving an image.
    if (!Quiet) {
      outs() << "Section '" + OffloadName + "': Image " << ImgCnt++
             << "'-> File '" + FileName + "'\n";
    }

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

  return 0;
}
