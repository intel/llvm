; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 3 Name [[#r1:]] "r1"
; CHECK-SPIRV: 3 Name [[#r2:]] "r2"
; CHECK-SPIRV: 3 Name [[#r3:]] "r3"
; CHECK-SPIRV: 3 Name [[#r4:]] "r4"
; CHECK-SPIRV: 3 Name [[#r5:]] "r5"
; CHECK-SPIRV: 3 Name [[#r6:]] "r6"
; CHECK-SPIRV: 3 Name [[#r7:]] "r7"
; CHECK-SPIRV: 3 Name [[#r8:]] "r8"
; CHECK-SPIRV: 3 Name [[#r9:]] "r9"
; CHECK-SPIRV: 3 Name [[#r10:]] "r10"
; CHECK-SPIRV: 3 Name [[#r11:]] "r11"
; CHECK-SPIRV: 3 Name [[#r12:]] "r12"
; CHECK-SPIRV: 3 Name [[#r13:]] "r13"
; CHECK-SPIRV: 3 Name [[#r14:]] "r14"
; CHECK-SPIRV: 3 Name [[#r15:]] "r15"
; CHECK-SPIRV: 3 Name [[#r16:]] "r16"
; CHECK-SPIRV: 3 Name [[#r17:]] "r17"
; CHECK-SPIRV: 3 Name [[#r18:]] "r18"
; CHECK-SPIRV: 3 Name [[#r19:]] "r19"
; CHECK-SPIRV: 3 Name [[#r20:]] "r20"
; CHECK-SPIRV: 3 Name [[#r21:]] "r21"
; CHECK-SPIRV: 3 Name [[#r22:]] "r22"
; CHECK-SPIRV: 3 Name [[#r23:]] "r23"
; CHECK-SPIRV: 3 Name [[#r24:]] "r24"
; CHECK-SPIRV: 3 Name [[#r25:]] "r25"
; CHECK-SPIRV: 3 Name [[#r26:]] "r26"
; CHECK-SPIRV: 3 Name [[#r27:]] "r27"
; CHECK-SPIRV: 3 Name [[#r28:]] "r28"
; CHECK-SPIRV: 3 Name [[#r29:]] "r29"
; CHECK-SPIRV: 3 Name [[#r30:]] "r30"
; CHECK-SPIRV: 3 Name [[#r31:]] "r31"
; CHECK-SPIRV: 3 Name [[#r32:]] "r32"
; CHECK-SPIRV: 3 Name [[#r33:]] "r33"
; CHECK-SPIRV: 3 Name [[#r34:]] "r34"
; CHECK-SPIRV: 3 Name [[#r35:]] "r35"
; CHECK-SPIRV: 3 Name [[#r36:]] "r36"
; CHECK-SPIRV: 3 Name [[#r37:]] "r37"
; CHECK-SPIRV: 3 Name [[#r38:]] "r38"
; CHECK-SPIRV: 3 Name [[#r39:]] "r39"
; CHECK-SPIRV: 3 Name [[#r40:]] "r40"
; CHECK-SPIRV: 3 Name [[#r41:]] "r41"
; CHECK-SPIRV: 3 Name [[#r42:]] "r42"
; CHECK-SPIRV: 3 Name [[#r43:]] "r43"
; CHECK-SPIRV: 3 Name [[#r44:]] "r44"
; CHECK-SPIRV: 3 Name [[#r45:]] "r45"
; CHECK-SPIRV: 3 Name [[#r46:]] "r46"
; CHECK-SPIRV: 3 Name [[#r47:]] "r47"
; CHECK-SPIRV: 3 Name [[#r48:]] "r48"
; CHECK-SPIRV: 3 Name [[#r49:]] "r49"
; CHECK-SPIRV: 3 Name [[#r50:]] "r50"
; CHECK-SPIRV: 3 Name [[#r51:]] "r51"
; CHECK-SPIRV: 3 Name [[#r52:]] "r52"
; CHECK-SPIRV: 3 Name [[#r53:]] "r53"
; CHECK-SPIRV: 3 Name [[#r54:]] "r54"
; CHECK-SPIRV: 3 Name [[#r55:]] "r55"
; CHECK-SPIRV: 3 Name [[#r56:]] "r56"
; CHECK-SPIRV: 3 Name [[#r57:]] "r57"
; CHECK-SPIRV: 3 Name [[#r58:]] "r58"
; CHECK-SPIRV: 3 Name [[#r59:]] "r59"
; CHECK-SPIRV: 3 Name [[#r60:]] "r60"
; CHECK-SPIRV: 3 Name [[#r61:]] "r61"
; CHECK-SPIRV: 3 Name [[#r62:]] "r62"
; CHECK-SPIRV: 3 Name [[#r63:]] "r63"
; CHECK-SPIRV: 3 Name [[#r64:]] "r64"
; CHECK-SPIRV: 3 Name [[#r65:]] "r65"
; CHECK-SPIRV: 3 Name [[#r66:]] "r66"
; CHECK-SPIRV: 3 Name [[#r67:]] "r67"
; CHECK-SPIRV: 3 Name [[#r68:]] "r68"
; CHECK-SPIRV: 3 Name [[#r69:]] "r69"
; CHECK-SPIRV: 3 Name [[#r70:]] "r70"
; CHECK-SPIRV: 3 Name [[#r71:]] "r71"
; CHECK-SPIRV: 3 Name [[#r72:]] "r72"
; CHECK-SPIRV: 3 Name [[#r73:]] "r73"
; CHECK-SPIRV: 3 Name [[#r74:]] "r74"
; CHECK-SPIRV: 3 Name [[#r75:]] "r75"
; CHECK-SPIRV: 3 Name [[#r76:]] "r76"
; CHECK-SPIRV: 3 Name [[#r77:]] "r77"
; CHECK-SPIRV: 3 Name [[#r78:]] "r78"
; CHECK-SPIRV: 3 Name [[#r79:]] "r79"
; CHECK-SPIRV: 3 Name [[#r80:]] "r80"
; CHECK-SPIRV: 3 Name [[#r81:]] "r81"
; CHECK-SPIRV: 3 Name [[#r82:]] "r82"
; CHECK-SPIRV: 3 Name [[#r83:]] "r83"
; CHECK-SPIRV: 3 Name [[#r84:]] "r84"
; CHECK-SPIRV: 3 Name [[#r85:]] "r85"
; CHECK-SPIRV: 3 Name [[#r86:]] "r86"
; CHECK-SPIRV: 3 Name [[#r87:]] "r87"
; CHECK-SPIRV: 3 Name [[#r88:]] "r88"
; CHECK-SPIRV: 3 Name [[#r89:]] "r89"
; CHECK-SPIRV: 3 Name [[#r90:]] "r90"
; CHECK-SPIRV-NOT: 4 Decorate {{.*}} FPFastMathMode
; CHECK-SPIRV: 2 TypeBool [[#bool:]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r1]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r2]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r3]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r4]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r5]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r6]]
; CHECK-SPIRV: 5 FOrdEqual [[#bool]] [[#r7]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r8]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r9]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r10]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r11]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r12]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r13]]
; CHECK-SPIRV: 5 FOrdNotEqual [[#bool]] [[#r14]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r15]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r16]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r17]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r18]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r19]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r20]]
; CHECK-SPIRV: 5 FOrdLessThan [[#bool]] [[#r21]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r22]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r23]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r24]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r25]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r26]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r27]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[#bool]] [[#r28]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r29]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r30]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r31]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r32]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r33]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r34]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[#bool]] [[#r35]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r36]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r37]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r38]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r39]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r40]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r41]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[#bool]] [[#r42]]
; CHECK-SPIRV: 5 Ordered [[#bool]] [[#r43]]
; CHECK-SPIRV: 5 Ordered [[#bool]] [[#r44]]
; CHECK-SPIRV: 5 Ordered [[#bool]] [[#r45]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r46]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r47]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r48]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r49]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r50]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r51]]
; CHECK-SPIRV: 5 FUnordEqual [[#bool]] [[#r52]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r53]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r54]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r55]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r56]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r57]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r58]]
; CHECK-SPIRV: 5 FUnordNotEqual [[#bool]] [[#r59]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r60]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r61]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r62]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r63]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r64]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r65]]
; CHECK-SPIRV: 5 FUnordLessThan [[#bool]] [[#r66]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r67]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r68]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r69]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r70]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r71]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r72]]
; CHECK-SPIRV: 5 FUnordGreaterThan [[#bool]] [[#r73]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r74]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r75]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r76]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r77]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r78]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r79]]
; CHECK-SPIRV: 5 FUnordLessThanEqual [[#bool]] [[#r80]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r81]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r82]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r83]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r84]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r85]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r86]]
; CHECK-SPIRV: 5 FUnordGreaterThanEqual [[#bool]] [[#r87]]
; CHECK-SPIRV: 5 Unordered [[#bool]] [[#r88]]
; CHECK-SPIRV: 5 Unordered [[#bool]] [[#r89]]
; CHECK-SPIRV: 5 Unordered [[#bool]] [[#r90]]

; CHECK-LLVM: %r1 = fcmp oeq float %a, %b
; CHECK-LLVM: %r2 = fcmp oeq float %a, %b
; CHECK-LLVM: %r3 = fcmp oeq float %a, %b
; CHECK-LLVM: %r4 = fcmp oeq float %a, %b
; CHECK-LLVM: %r5 = fcmp oeq float %a, %b
; CHECK-LLVM: %r6 = fcmp oeq float %a, %b
; CHECK-LLVM: %r7 = fcmp oeq float %a, %b
; CHECK-LLVM: %r8 = fcmp one float %a, %b
; CHECK-LLVM: %r9 = fcmp one float %a, %b
; CHECK-LLVM: %r10 = fcmp one float %a, %b
; CHECK-LLVM: %r11 = fcmp one float %a, %b
; CHECK-LLVM: %r12 = fcmp one float %a, %b
; CHECK-LLVM: %r13 = fcmp one float %a, %b
; CHECK-LLVM: %r14 = fcmp one float %a, %b
; CHECK-LLVM: %r15 = fcmp olt float %a, %b
; CHECK-LLVM: %r16 = fcmp olt float %a, %b
; CHECK-LLVM: %r17 = fcmp olt float %a, %b
; CHECK-LLVM: %r18 = fcmp olt float %a, %b
; CHECK-LLVM: %r19 = fcmp olt float %a, %b
; CHECK-LLVM: %r20 = fcmp olt float %a, %b
; CHECK-LLVM: %r21 = fcmp olt float %a, %b
; CHECK-LLVM: %r22 = fcmp ogt float %a, %b
; CHECK-LLVM: %r23 = fcmp ogt float %a, %b
; CHECK-LLVM: %r24 = fcmp ogt float %a, %b
; CHECK-LLVM: %r25 = fcmp ogt float %a, %b
; CHECK-LLVM: %r26 = fcmp ogt float %a, %b
; CHECK-LLVM: %r27 = fcmp ogt float %a, %b
; CHECK-LLVM: %r28 = fcmp ogt float %a, %b
; CHECK-LLVM: %r29 = fcmp ole float %a, %b
; CHECK-LLVM: %r30 = fcmp ole float %a, %b
; CHECK-LLVM: %r31 = fcmp ole float %a, %b
; CHECK-LLVM: %r32 = fcmp ole float %a, %b
; CHECK-LLVM: %r33 = fcmp ole float %a, %b
; CHECK-LLVM: %r34 = fcmp ole float %a, %b
; CHECK-LLVM: %r35 = fcmp ole float %a, %b
; CHECK-LLVM: %r36 = fcmp oge float %a, %b
; CHECK-LLVM: %r37 = fcmp oge float %a, %b
; CHECK-LLVM: %r38 = fcmp oge float %a, %b
; CHECK-LLVM: %r39 = fcmp oge float %a, %b
; CHECK-LLVM: %r40 = fcmp oge float %a, %b
; CHECK-LLVM: %r41 = fcmp oge float %a, %b
; CHECK-LLVM: %r42 = fcmp oge float %a, %b
; CHECK-LLVM: %r43 = fcmp ord float %a, %b
; CHECK-LLVM: %r44 = fcmp ord float %a, %b
; CHECK-LLVM: %r45 = fcmp ord float %a, %b
; CHECK-LLVM: %r46 = fcmp ueq float %a, %b
; CHECK-LLVM: %r47 = fcmp ueq float %a, %b
; CHECK-LLVM: %r48 = fcmp ueq float %a, %b
; CHECK-LLVM: %r49 = fcmp ueq float %a, %b
; CHECK-LLVM: %r50 = fcmp ueq float %a, %b
; CHECK-LLVM: %r51 = fcmp ueq float %a, %b
; CHECK-LLVM: %r52 = fcmp ueq float %a, %b
; CHECK-LLVM: %r53 = fcmp une float %a, %b
; CHECK-LLVM: %r54 = fcmp une float %a, %b
; CHECK-LLVM: %r55 = fcmp une float %a, %b
; CHECK-LLVM: %r56 = fcmp une float %a, %b
; CHECK-LLVM: %r57 = fcmp une float %a, %b
; CHECK-LLVM: %r58 = fcmp une float %a, %b
; CHECK-LLVM: %r59 = fcmp une float %a, %b
; CHECK-LLVM: %r60 = fcmp ult float %a, %b
; CHECK-LLVM: %r61 = fcmp ult float %a, %b
; CHECK-LLVM: %r62 = fcmp ult float %a, %b
; CHECK-LLVM: %r63 = fcmp ult float %a, %b
; CHECK-LLVM: %r64 = fcmp ult float %a, %b
; CHECK-LLVM: %r65 = fcmp ult float %a, %b
; CHECK-LLVM: %r66 = fcmp ult float %a, %b
; CHECK-LLVM: %r67 = fcmp ugt float %a, %b
; CHECK-LLVM: %r68 = fcmp ugt float %a, %b
; CHECK-LLVM: %r69 = fcmp ugt float %a, %b
; CHECK-LLVM: %r70 = fcmp ugt float %a, %b
; CHECK-LLVM: %r71 = fcmp ugt float %a, %b
; CHECK-LLVM: %r72 = fcmp ugt float %a, %b
; CHECK-LLVM: %r73 = fcmp ugt float %a, %b
; CHECK-LLVM: %r74 = fcmp ule float %a, %b
; CHECK-LLVM: %r75 = fcmp ule float %a, %b
; CHECK-LLVM: %r76 = fcmp ule float %a, %b
; CHECK-LLVM: %r77 = fcmp ule float %a, %b
; CHECK-LLVM: %r78 = fcmp ule float %a, %b
; CHECK-LLVM: %r79 = fcmp ule float %a, %b
; CHECK-LLVM: %r80 = fcmp ule float %a, %b
; CHECK-LLVM: %r81 = fcmp uge float %a, %b
; CHECK-LLVM: %r82 = fcmp uge float %a, %b
; CHECK-LLVM: %r83 = fcmp uge float %a, %b
; CHECK-LLVM: %r84 = fcmp uge float %a, %b
; CHECK-LLVM: %r85 = fcmp uge float %a, %b
; CHECK-LLVM: %r86 = fcmp uge float %a, %b
; CHECK-LLVM: %r87 = fcmp uge float %a, %b
; CHECK-LLVM: %r88 = fcmp uno float %a, %b
; CHECK-LLVM: %r89 = fcmp uno float %a, %b
; CHECK-LLVM: %r90 = fcmp uno float %a, %b

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testFCmp(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %r1 = fcmp oeq float %a, %b
  %r2 = fcmp nnan oeq float %a, %b
  %r3 = fcmp ninf oeq float %a, %b
  %r4 = fcmp nsz oeq float %a, %b
  %r5 = fcmp arcp oeq float %a, %b
  %r6 = fcmp fast oeq float %a, %b
  %r7 = fcmp nnan ninf oeq float %a, %b
  %r8 = fcmp one float %a, %b
  %r9 = fcmp nnan one float %a, %b
  %r10 = fcmp ninf one float %a, %b
  %r11 = fcmp nsz one float %a, %b
  %r12 = fcmp arcp one float %a, %b
  %r13 = fcmp fast one float %a, %b
  %r14 = fcmp nnan ninf one float %a, %b
  %r15 = fcmp olt float %a, %b
  %r16 = fcmp nnan olt float %a, %b
  %r17 = fcmp ninf olt float %a, %b
  %r18 = fcmp nsz olt float %a, %b
  %r19 = fcmp arcp olt float %a, %b
  %r20 = fcmp fast olt float %a, %b
  %r21 = fcmp nnan ninf olt float %a, %b
  %r22 = fcmp ogt float %a, %b
  %r23 = fcmp nnan ogt float %a, %b
  %r24 = fcmp ninf ogt float %a, %b
  %r25 = fcmp nsz ogt float %a, %b
  %r26 = fcmp arcp ogt float %a, %b
  %r27 = fcmp fast ogt float %a, %b
  %r28 = fcmp nnan ninf ogt float %a, %b
  %r29 = fcmp ole float %a, %b
  %r30 = fcmp nnan ole float %a, %b
  %r31 = fcmp ninf ole float %a, %b
  %r32 = fcmp nsz ole float %a, %b
  %r33 = fcmp arcp ole float %a, %b
  %r34 = fcmp fast ole float %a, %b
  %r35 = fcmp nnan ninf ole float %a, %b
  %r36 = fcmp oge float %a, %b
  %r37 = fcmp nnan oge float %a, %b
  %r38 = fcmp ninf oge float %a, %b
  %r39 = fcmp nsz oge float %a, %b
  %r40 = fcmp arcp oge float %a, %b
  %r41 = fcmp fast oge float %a, %b
  %r42 = fcmp nnan ninf oge float %a, %b
  %r43 = fcmp ord float %a, %b
  %r44 = fcmp ninf ord float %a, %b
  %r45 = fcmp nsz ord float %a, %b
  %r46 = fcmp ueq float %a, %b
  %r47 = fcmp nnan ueq float %a, %b
  %r48 = fcmp ninf ueq float %a, %b
  %r49 = fcmp nsz ueq float %a, %b
  %r50 = fcmp arcp ueq float %a, %b
  %r51 = fcmp fast ueq float %a, %b
  %r52 = fcmp nnan ninf ueq float %a, %b
  %r53 = fcmp une float %a, %b
  %r54 = fcmp nnan une float %a, %b
  %r55 = fcmp ninf une float %a, %b
  %r56 = fcmp nsz une float %a, %b
  %r57 = fcmp arcp une float %a, %b
  %r58 = fcmp fast une float %a, %b
  %r59 = fcmp nnan ninf une float %a, %b
  %r60 = fcmp ult float %a, %b
  %r61 = fcmp nnan ult float %a, %b
  %r62 = fcmp ninf ult float %a, %b
  %r63 = fcmp nsz ult float %a, %b
  %r64 = fcmp arcp ult float %a, %b
  %r65 = fcmp fast ult float %a, %b
  %r66 = fcmp nnan ninf ult float %a, %b
  %r67 = fcmp ugt float %a, %b
  %r68 = fcmp nnan ugt float %a, %b
  %r69 = fcmp ninf ugt float %a, %b
  %r70 = fcmp nsz ugt float %a, %b
  %r71 = fcmp arcp ugt float %a, %b
  %r72 = fcmp fast ugt float %a, %b
  %r73 = fcmp nnan ninf ugt float %a, %b
  %r74 = fcmp ule float %a, %b
  %r75 = fcmp nnan ule float %a, %b
  %r76 = fcmp ninf ule float %a, %b
  %r77 = fcmp nsz ule float %a, %b
  %r78 = fcmp arcp ule float %a, %b
  %r79 = fcmp fast ule float %a, %b
  %r80 = fcmp nnan ninf ule float %a, %b
  %r81 = fcmp uge float %a, %b
  %r82 = fcmp nnan uge float %a, %b
  %r83 = fcmp ninf uge float %a, %b
  %r84 = fcmp nsz uge float %a, %b
  %r85 = fcmp arcp uge float %a, %b
  %r86 = fcmp fast uge float %a, %b
  %r87 = fcmp nnan ninf uge float %a, %b
  %r88 = fcmp uno float %a, %b
  %r89 = fcmp ninf uno float %a, %b
  %r90 = fcmp nsz uno float %a, %b
  ret void
}

attributes #0 = { convergent nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 0, i32 0}
!3 = !{!"none", !"none"}
!4 = !{!"float", !"float"}
!5 = !{!"", !""}
