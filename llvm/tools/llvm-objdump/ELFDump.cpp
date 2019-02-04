//===-- ELFDump.cpp - ELF-specific dumper -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ELF-specific dumper for llvm-objdump.
///
//===----------------------------------------------------------------------===//

#include "llvm-objdump.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

template <class ELFT>
Expected<StringRef> getDynamicStrTab(const ELFFile<ELFT> *Elf) {
  typedef ELFFile<ELFT> ELFO;

  auto DynamicEntriesOrError = Elf->dynamicEntries();
  if (!DynamicEntriesOrError)
    return DynamicEntriesOrError.takeError();

  for (const typename ELFO::Elf_Dyn &Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_STRTAB) {
      auto MappedAddrOrError = Elf->toMappedAddr(Dyn.getPtr());
      if (!MappedAddrOrError)
        consumeError(MappedAddrOrError.takeError());
      return StringRef(reinterpret_cast<const char *>(*MappedAddrOrError));
    }
  }

  // If the dynamic segment is not present, we fall back on the sections.
  auto SectionsOrError = Elf->sections();
  if (!SectionsOrError)
    return SectionsOrError.takeError();

  for (const typename ELFO::Elf_Shdr &Sec : *SectionsOrError) {
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      return Elf->getStringTableForSymtab(Sec);
  }

  return createError("dynamic string table not found");
}

template <class ELFT>
static std::error_code getRelocationValueString(const ELFObjectFile<ELFT> *Obj,
                                                const RelocationRef &RelRef,
                                                SmallVectorImpl<char> &Result) {
  typedef typename ELFObjectFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFObjectFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFObjectFile<ELFT>::Elf_Rela Elf_Rela;

  const ELFFile<ELFT> &EF = *Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  auto SecOrErr = EF.getSection(Rel.d.a);
  if (!SecOrErr)
    return errorToErrorCode(SecOrErr.takeError());

  int64_t Addend = 0;
  // If there is no Symbol associated with the relocation, we set the undef
  // boolean value to 'true'. This will prevent us from calling functions that
  // requires the relocation to be associated with a symbol.
  //
  // In SHT_REL case we would need to read the addend from section data.
  // GNU objdump does not do that and we just follow for simplicity atm.
  bool Undef = false;
  if ((*SecOrErr)->sh_type == ELF::SHT_RELA) {
    const Elf_Rela *ERela = Obj->getRela(Rel);
    Addend = ERela->r_addend;
    Undef = ERela->getSymbol(false) == 0;
  } else if ((*SecOrErr)->sh_type != ELF::SHT_REL) {
    return object_error::parse_failed;
  }

  // Default scheme is to print Target, as well as "+ <addend>" for nonzero
  // addend. Should be acceptable for all normal purposes.
  std::string FmtBuf;
  raw_string_ostream Fmt(FmtBuf);

  if (!Undef) {
    symbol_iterator SI = RelRef.getSymbol();
    const Elf_Sym *Sym = Obj->getSymbol(SI->getRawDataRefImpl());
    if (Sym->getType() == ELF::STT_SECTION) {
      Expected<section_iterator> SymSI = SI->getSection();
      if (!SymSI)
        return errorToErrorCode(SymSI.takeError());
      const Elf_Shdr *SymSec = Obj->getSection((*SymSI)->getRawDataRefImpl());
      auto SecName = EF.getSectionName(SymSec);
      if (!SecName)
        return errorToErrorCode(SecName.takeError());
      Fmt << *SecName;
    } else {
      Expected<StringRef> SymName = SI->getName();
      if (!SymName)
        return errorToErrorCode(SymName.takeError());
      if (Demangle)
        Fmt << demangle(*SymName);
      else
        Fmt << *SymName;
    }
  } else {
    Fmt << "*ABS*";
  }

  if (Addend != 0)
    Fmt << (Addend < 0 ? "" : "+") << Addend;
  Fmt.flush();
  Result.append(FmtBuf.begin(), FmtBuf.end());
  return std::error_code();
}

std::error_code
llvm::getELFRelocationValueString(const ELFObjectFileBase *Obj,
                                  const RelocationRef &Rel,
                                  SmallVectorImpl<char> &Result) {
  if (auto *ELF32LE = dyn_cast<ELF32LEObjectFile>(Obj))
    return getRelocationValueString(ELF32LE, Rel, Result);
  if (auto *ELF64LE = dyn_cast<ELF64LEObjectFile>(Obj))
    return getRelocationValueString(ELF64LE, Rel, Result);
  if (auto *ELF32BE = dyn_cast<ELF32BEObjectFile>(Obj))
    return getRelocationValueString(ELF32BE, Rel, Result);
  auto *ELF64BE = cast<ELF64BEObjectFile>(Obj);
  return getRelocationValueString(ELF64BE, Rel, Result);
}

template <class ELFT>
static uint64_t getSectionLMA(const ELFFile<ELFT> *Obj,
                              const object::ELFSectionRef &Sec) {
  auto PhdrRangeOrErr = Obj->program_headers();
  if (!PhdrRangeOrErr)
    report_fatal_error(errorToErrorCode(PhdrRangeOrErr.takeError()).message());

  // Search for a PT_LOAD segment containing the requested section. Use this
  // segment's p_addr to calculate the section's LMA.
  for (const typename ELFFile<ELFT>::Elf_Phdr &Phdr : *PhdrRangeOrErr)
    if ((Phdr.p_type == ELF::PT_LOAD) && (Phdr.p_vaddr <= Sec.getAddress()) &&
        (Phdr.p_vaddr + Phdr.p_memsz > Sec.getAddress()))
      return Sec.getAddress() - Phdr.p_vaddr + Phdr.p_paddr;

  // Return section's VMA if it isn't in a PT_LOAD segment.
  return Sec.getAddress();
}

uint64_t llvm::getELFSectionLMA(const object::ELFSectionRef &Sec) {
  if (const auto *ELFObj = dyn_cast<ELF32LEObjectFile>(Sec.getObject()))
    return getSectionLMA(ELFObj->getELFFile(), Sec);
  else if (const auto *ELFObj = dyn_cast<ELF32BEObjectFile>(Sec.getObject()))
    return getSectionLMA(ELFObj->getELFFile(), Sec);
  else if (const auto *ELFObj = dyn_cast<ELF64LEObjectFile>(Sec.getObject()))
    return getSectionLMA(ELFObj->getELFFile(), Sec);
  const auto *ELFObj = cast<ELF64BEObjectFile>(Sec.getObject());
  return getSectionLMA(ELFObj->getELFFile(), Sec);
}

template <class ELFT>
void printDynamicSection(const ELFFile<ELFT> *Elf, StringRef Filename) {
  auto ProgramHeaderOrError = Elf->program_headers();
  if (!ProgramHeaderOrError)
    report_error(Filename, ProgramHeaderOrError.takeError());

  auto DynamicEntriesOrError = Elf->dynamicEntries();
  if (!DynamicEntriesOrError)
    report_error(Filename, DynamicEntriesOrError.takeError());

  outs() << "Dynamic Section:\n";
  for (const auto &Dyn : *DynamicEntriesOrError) {
    if (Dyn.d_tag == ELF::DT_NULL)
      continue;

    StringRef Str = StringRef(Elf->getDynamicTagAsString(Dyn.d_tag));

    if (Str.empty()) {
      std::string HexStr = utohexstr(static_cast<uint64_t>(Dyn.d_tag), true);
      outs() << format("  0x%-19s", HexStr.c_str());
    } else {
      // We use "-21" in order to match GNU objdump's output.
      outs() << format("  %-21s", Str.data());
    }

    const char *Fmt =
        ELFT::Is64Bits ? "0x%016" PRIx64 "\n" : "0x%08" PRIx64 "\n";
    if (Dyn.d_tag == ELF::DT_NEEDED) {
      Expected<StringRef> StrTabOrErr = getDynamicStrTab(Elf);
      if (StrTabOrErr) {
        const char *Data = StrTabOrErr.get().data();
        outs() << (Data + Dyn.d_un.d_val) << "\n";
        continue;
      }
      warn(errorToErrorCode(StrTabOrErr.takeError()).message());
      consumeError(StrTabOrErr.takeError());
    }
    outs() << format(Fmt, (uint64_t)Dyn.d_un.d_val);
  }
}

template <class ELFT> void printProgramHeaders(const ELFFile<ELFT> *o) {
  typedef ELFFile<ELFT> ELFO;
  outs() << "Program Header:\n";
  auto ProgramHeaderOrError = o->program_headers();
  if (!ProgramHeaderOrError)
    report_fatal_error(
        errorToErrorCode(ProgramHeaderOrError.takeError()).message());
  for (const typename ELFO::Elf_Phdr &Phdr : *ProgramHeaderOrError) {
    switch (Phdr.p_type) {
    case ELF::PT_DYNAMIC:
      outs() << " DYNAMIC ";
      break;
    case ELF::PT_GNU_EH_FRAME:
      outs() << "EH_FRAME ";
      break;
    case ELF::PT_GNU_RELRO:
      outs() << "   RELRO ";
      break;
    case ELF::PT_GNU_STACK:
      outs() << "   STACK ";
      break;
    case ELF::PT_INTERP:
      outs() << "  INTERP ";
      break;
    case ELF::PT_LOAD:
      outs() << "    LOAD ";
      break;
    case ELF::PT_NOTE:
      outs() << "    NOTE ";
      break;
    case ELF::PT_OPENBSD_BOOTDATA:
      outs() << "    OPENBSD_BOOTDATA ";
      break;
    case ELF::PT_OPENBSD_RANDOMIZE:
      outs() << "    OPENBSD_RANDOMIZE ";
      break;
    case ELF::PT_OPENBSD_WXNEEDED:
      outs() << "    OPENBSD_WXNEEDED ";
      break;
    case ELF::PT_PHDR:
      outs() << "    PHDR ";
      break;
    case ELF::PT_TLS:
      outs() << "    TLS ";
      break;
    default:
      outs() << " UNKNOWN ";
    }

    const char *Fmt = ELFT::Is64Bits ? "0x%016" PRIx64 " " : "0x%08" PRIx64 " ";

    outs() << "off    " << format(Fmt, (uint64_t)Phdr.p_offset) << "vaddr "
           << format(Fmt, (uint64_t)Phdr.p_vaddr) << "paddr "
           << format(Fmt, (uint64_t)Phdr.p_paddr)
           << format("align 2**%u\n",
                     countTrailingZeros<uint64_t>(Phdr.p_align))
           << "         filesz " << format(Fmt, (uint64_t)Phdr.p_filesz)
           << "memsz " << format(Fmt, (uint64_t)Phdr.p_memsz) << "flags "
           << ((Phdr.p_flags & ELF::PF_R) ? "r" : "-")
           << ((Phdr.p_flags & ELF::PF_W) ? "w" : "-")
           << ((Phdr.p_flags & ELF::PF_X) ? "x" : "-") << "\n";
  }
  outs() << "\n";
}

void llvm::printELFFileHeader(const object::ObjectFile *Obj) {
  if (const auto *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj))
    printProgramHeaders(ELFObj->getELFFile());
  else if (const auto *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj))
    printProgramHeaders(ELFObj->getELFFile());
  else if (const auto *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj))
    printProgramHeaders(ELFObj->getELFFile());
  else if (const auto *ELFObj = dyn_cast<ELF64BEObjectFile>(Obj))
    printProgramHeaders(ELFObj->getELFFile());
}

void llvm::printELFDynamicSection(const object::ObjectFile *Obj) {
  if (const auto *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj))
    printDynamicSection(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj))
    printDynamicSection(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj))
    printDynamicSection(ELFObj->getELFFile(), Obj->getFileName());
  else if (const auto *ELFObj = dyn_cast<ELF64BEObjectFile>(Obj))
    printDynamicSection(ELFObj->getELFFile(), Obj->getFileName());
}
