//===- MarkLive.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements --gc-sections, which is a feature to remove unused
// chunks from the output. Unused chunks are those that are not reachable from
// known root symbols or chunks. This feature is implemented as a mark-sweep
// garbage collector.
//
// Here's how it works. Each InputChunk has a "Live" bit. The bit is off by
// default. Starting with the GC-roots, visit all reachable chunks and set their
// Live bits. The Writer will then ignore chunks whose Live bits are off, so
// that such chunk are not appear in the output.
//
//===----------------------------------------------------------------------===//

#include "MarkLive.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputEvent.h"
#include "InputGlobal.h"
#include "SymbolTable.h"
#include "Symbols.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;

void lld::wasm::markLive() {
  if (!Config->GcSections)
    return;

  LLVM_DEBUG(dbgs() << "markLive\n");
  SmallVector<InputChunk *, 256> Q;

  std::function<void(Symbol*)> Enqueue = [&](Symbol *Sym) {
    if (!Sym || Sym->isLive())
      return;
    LLVM_DEBUG(dbgs() << "markLive: " << Sym->getName() << "\n");
    Sym->markLive();
    if (InputChunk *Chunk = Sym->getChunk())
      Q.push_back(Chunk);

    // The ctor functions are all referenced by the synthetic CallCtors
    // function.  However, this function does not contain relocations so we
    // have to manually mark the ctors as live if CallCtors itself is live.
    if (Sym == WasmSym::CallCtors) {
      for (const ObjFile *Obj : Symtab->ObjectFiles) {
        const WasmLinkingData &L = Obj->getWasmObj()->linkingData();
        for (const WasmInitFunc &F : L.InitFunctions)
          Enqueue(Obj->getFunctionSymbol(F.Symbol));
      }
    }
  };

  // Add GC root symbols.
  if (!Config->Entry.empty())
    Enqueue(Symtab->find(Config->Entry));

  // We need to preserve any exported symbol
  for (Symbol *Sym : Symtab->getSymbols())
    if (Sym->isExported())
      Enqueue(Sym);

  // For relocatable output, we need to preserve all the ctor functions
  if (Config->Relocatable) {
    for (const ObjFile *Obj : Symtab->ObjectFiles) {
      const WasmLinkingData &L = Obj->getWasmObj()->linkingData();
      for (const WasmInitFunc &F : L.InitFunctions)
        Enqueue(Obj->getFunctionSymbol(F.Symbol));
    }
  }

  // Follow relocations to mark all reachable chunks.
  while (!Q.empty()) {
    InputChunk *C = Q.pop_back_val();

    for (const WasmRelocation Reloc : C->getRelocations()) {
      if (Reloc.Type == R_WASM_TYPE_INDEX_LEB)
        continue;
      Symbol *Sym = C->File->getSymbol(Reloc.Index);

      // If the function has been assigned the special index zero in the table,
      // the relocation doesn't pull in the function body, since the function
      // won't actually go in the table (the runtime will trap attempts to call
      // that index, since we don't use it).  A function with a table index of
      // zero is only reachable via "call", not via "call_indirect".  The stub
      // functions used for weak-undefined symbols have this behaviour (compare
      // equal to null pointer, only reachable via direct call).
      if (Reloc.Type == R_WASM_TABLE_INDEX_SLEB ||
          Reloc.Type == R_WASM_TABLE_INDEX_I32) {
        auto *FuncSym = cast<FunctionSymbol>(Sym);
        if (FuncSym->hasTableIndex() && FuncSym->getTableIndex() == 0)
          continue;
      }

      Enqueue(Sym);
    }
  }

  // Report garbage-collected sections.
  if (Config->PrintGcSections) {
    for (const ObjFile *Obj : Symtab->ObjectFiles) {
      for (InputChunk *C : Obj->Functions)
        if (!C->Live)
          message("removing unused section " + toString(C));
      for (InputChunk *C : Obj->Segments)
        if (!C->Live)
          message("removing unused section " + toString(C));
      for (InputGlobal *G : Obj->Globals)
        if (!G->Live)
          message("removing unused section " + toString(G));
      for (InputEvent *E : Obj->Events)
        if (!E->Live)
          message("removing unused section " + toString(E));
    }
    for (InputChunk *C : Symtab->SyntheticFunctions)
      if (!C->Live)
        message("removing unused section " + toString(C));
    for (InputGlobal *G : Symtab->SyntheticGlobals)
      if (!G->Live)
        message("removing unused section " + toString(G));
  }
}
