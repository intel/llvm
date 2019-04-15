# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %p/Inputs/aarch64-tls-ie.s -o %ttlsie.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux %s -o %tmain.o
# RUN: ld.lld %tmain.o %ttlsie.o -o %tout
# RUN: llvm-objdump -d %tout | FileCheck %s
# RUN: llvm-readobj -s -r %tout | FileCheck -check-prefix=RELOC %s

# Initial-Exec to Local-Exec relax creates no dynamic relocations.
# RELOC:      Relocations [
# RELOC-NEXT: ]

# TCB size = 64 and foo is first element from TLS register.
# CHECK: Disassembly of section .text:
# CHECK: _start:
# CHECK-NEXT: 210000:  00 00 a0 d2   movz   x0, #0, lsl #16
# CHECK-NEXT: 210004:  80 08 80 f2   movk   x0, #68
# CHECK-NEXT: 210008:  00 00 a0 d2   movz   x0, #0, lsl #16
# CHECK-NEXT: 21000c:  00 08 80 f2   movk   x0, #64

.section .tdata
.align 2
.type foo_local, %object
.size foo_local, 4
foo_local:
.word 5
.text

.globl _start
_start:
 adrp    x0, :gottprel:foo
 ldr     x0, [x0, :gottprel_lo12:foo]
 adrp    x0, :gottprel:foo_local
 ldr     x0, [x0, :gottprel_lo12:foo_local]
