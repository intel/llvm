; RUN: %pp-llvm-ver --llvm-ver 8  < %s | FileCheck %s.check8
; RUN: %pp-llvm-ver --llvm-ver 9  < %s | FileCheck %s.check9
; RUN: %pp-llvm-ver --llvm-ver 10 < %s | FileCheck %s.check10
; RUN: %pp-llvm-ver --llvm-ver 11 < %s | FileCheck %s.check11

; CHECK:                    x0x
; CHECK-LABEL:              x1x
; CHECK-GE8LE11:            x2x
; CHECK-GE8LE11-LABEL:      x3x
; FOO:                      x4x
; FOO-LABEL:                x5x

; CHECK-GE9LE11-LABEL:  x6x
; CHECK-GE10LE11-LABEL: x7x
; CHECK-GE11LE11-LABEL: x8x

; CHECK-EQ9-EQ11-DAG: x9x
; CHECK-EQ8-EQ10-DAG: x10x

; CHECK-GT9LT11:     x11x
; CHECK-EQ8-GT9LT11: x12x

; FOO-EQ8-EQ9-EQ11: x13x

; FOO-EQ8EQ9: x14x
