; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-15
; RUN: llvm-spirv %t.bc --spirv-max-version=1.5 -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; The next 2 lines do not check the individual flags for each fcmp. Only for 'r1', 'r2' and 'r6'.
; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.6 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-16,CHECK-SPIRV-16-DEFAULT
; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.6 --spirv-ext=+SPV_KHR_float_controls2 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-16,CHECK-SPIRV-16-FC2

; RUN: llvm-spirv %t.bc --spirv-max-version=1.6 -o %t.spv
; RUN: llvm-spirv %t.bc --spirv-max-version=1.6 --spirv-ext=+SPV_KHR_float_controls2 -o %t.fc2.spv
; RUN: spirv-val %t.spv
; RUN: spirv-val %t.fc2.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefixes=CHECK-LLVM-16,CHECK-LLVM-16-DEFAULT
; RUN: llvm-spirv -r %t.fc2.spv -o - | llvm-dis -o - | FileCheck %s --check-prefixes=CHECK-LLVM-16,CHECK-LLVM-16-FC2
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv32-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

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
; CHECK-SPIRV-15-NOT: 4 Decorate {{.*}} FPFastMathMode
; CHECK-SPIRV-16-NOT: 4 Decorate [[#r1]] FPFastMathMode
; CHECK-SPIRV-16: Decorate [[#r2]] FPFastMathMode 1
; CHECK-SPIRV-16-DEFAULT: Decorate [[#r6]] FPFastMathMode 16
; CHECK-SPIRV-16-FC2: Decorate [[#r6]] FPFastMathMode 458767
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

; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp oeq float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp one float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp olt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ogt float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp ole float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp oge float %a, %b
; CHECK-LLVM: fcmp ord float %a, %b
; CHECK-LLVM: fcmp ord float %a, %b
; CHECK-LLVM: fcmp ord float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp ueq float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp une float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ult float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ugt float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp ule float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uge float %a, %b
; CHECK-LLVM: fcmp uno float %a, %b
; CHECK-LLVM: fcmp uno float %a, %b
; CHECK-LLVM: fcmp uno float %a, %b


; CHECK-LLVM-16: %r1 = fcmp oeq float %a, %b
; CHECK-LLVM-16: %r2 = fcmp nnan oeq float %a, %b
; CHECK-LLVM-16: %r3 = fcmp ninf oeq float %a, %b
; CHECK-LLVM-16: %r4 = fcmp nsz oeq float %a, %b
; CHECK-LLVM-16: %r5 = fcmp arcp oeq float %a, %b
; CHECK-LLVM-16-DEFAULT: %r6 = fcmp fast oeq float %a, %b
; CHECK-LLVM-16-FC2: %r6 = fcmp reassoc nnan ninf nsz arcp contract oeq float %a, %b
; CHECK-LLVM-16: %r7 = fcmp nnan ninf oeq float %a, %b
; CHECK-LLVM-16: %r8 = fcmp one float %a, %b
; CHECK-LLVM-16: %r9 = fcmp nnan one float %a, %b
; CHECK-LLVM-16: %r10 = fcmp ninf one float %a, %b
; CHECK-LLVM-16: %r11 = fcmp nsz one float %a, %b
; CHECK-LLVM-16: %r12 = fcmp arcp one float %a, %b
; CHECK-LLVM-16-DEFAULT: %r13 = fcmp fast one float %a, %b
; CHECK-LLVM-16-FC2: %r13 = fcmp reassoc nnan ninf nsz arcp contract one float %a, %b
; CHECK-LLVM-16: %r14 = fcmp nnan ninf one float %a, %b
; CHECK-LLVM-16: %r15 = fcmp olt float %a, %b
; CHECK-LLVM-16: %r16 = fcmp nnan olt float %a, %b
; CHECK-LLVM-16: %r17 = fcmp ninf olt float %a, %b
; CHECK-LLVM-16: %r18 = fcmp nsz olt float %a, %b
; CHECK-LLVM-16: %r19 = fcmp arcp olt float %a, %b
; CHECK-LLVM-16-DEFAULT: %r20 = fcmp fast olt float %a, %b
; CHECK-LLVM-16-FC2: %r20 = fcmp reassoc nnan ninf nsz arcp contract olt float %a, %b
; CHECK-LLVM-16: %r21 = fcmp nnan ninf olt float %a, %b
; CHECK-LLVM-16: %r22 = fcmp ogt float %a, %b
; CHECK-LLVM-16: %r23 = fcmp nnan ogt float %a, %b
; CHECK-LLVM-16: %r24 = fcmp ninf ogt float %a, %b
; CHECK-LLVM-16: %r25 = fcmp nsz ogt float %a, %b
; CHECK-LLVM-16: %r26 = fcmp arcp ogt float %a, %b
; CHECK-LLVM-16-DEFAULT: %r27 = fcmp fast ogt float %a, %b
; CHECK-LLVM-16-FC2: %r27 = fcmp reassoc nnan ninf nsz arcp contract ogt float %a, %b
; CHECK-LLVM-16: %r28 = fcmp nnan ninf ogt float %a, %b
; CHECK-LLVM-16: %r29 = fcmp ole float %a, %b
; CHECK-LLVM-16: %r30 = fcmp nnan ole float %a, %b
; CHECK-LLVM-16: %r31 = fcmp ninf ole float %a, %b
; CHECK-LLVM-16: %r32 = fcmp nsz ole float %a, %b
; CHECK-LLVM-16: %r33 = fcmp arcp ole float %a, %b
; CHECK-LLVM-16-DEFAULT: %r34 = fcmp fast ole float %a, %b
; CHECK-LLVM-16-FC2: %r34 = fcmp reassoc nnan ninf nsz arcp contract ole float %a, %b
; CHECK-LLVM-16: %r35 = fcmp nnan ninf ole float %a, %b
; CHECK-LLVM-16: %r36 = fcmp oge float %a, %b
; CHECK-LLVM-16: %r37 = fcmp nnan oge float %a, %b
; CHECK-LLVM-16: %r38 = fcmp ninf oge float %a, %b
; CHECK-LLVM-16: %r39 = fcmp nsz oge float %a, %b
; CHECK-LLVM-16: %r40 = fcmp arcp oge float %a, %b
; CHECK-LLVM-16-DEFAULT: %r41 = fcmp fast oge float %a, %b
; CHECK-LLVM-16-FC2: %r41 = fcmp reassoc nnan ninf nsz arcp contract oge float %a, %b
; CHECK-LLVM-16: %r42 = fcmp nnan ninf oge float %a, %b
; CHECK-LLVM-16: %r43 = fcmp ord float %a, %b
; CHECK-LLVM-16: %r44 = fcmp ninf ord float %a, %b
; CHECK-LLVM-16: %r45 = fcmp nsz ord float %a, %b
; CHECK-LLVM-16: %r46 = fcmp ueq float %a, %b
; CHECK-LLVM-16: %r47 = fcmp nnan ueq float %a, %b
; CHECK-LLVM-16: %r48 = fcmp ninf ueq float %a, %b
; CHECK-LLVM-16: %r49 = fcmp nsz ueq float %a, %b
; CHECK-LLVM-16: %r50 = fcmp arcp ueq float %a, %b
; CHECK-LLVM-16-DEFAULT: %r51 = fcmp fast ueq float %a, %b
; CHECK-LLVM-16-FC2: %r51 = fcmp reassoc nnan ninf nsz arcp contract ueq float %a, %b
; CHECK-LLVM-16: %r52 = fcmp nnan ninf ueq float %a, %b
; CHECK-LLVM-16: %r53 = fcmp une float %a, %b
; CHECK-LLVM-16: %r54 = fcmp nnan une float %a, %b
; CHECK-LLVM-16: %r55 = fcmp ninf une float %a, %b
; CHECK-LLVM-16: %r56 = fcmp nsz une float %a, %b
; CHECK-LLVM-16: %r57 = fcmp arcp une float %a, %b
; CHECK-LLVM-16-DEFAULT: %r58 = fcmp fast une float %a, %b
; CHECK-LLVM-16-FC2: %r58 = fcmp reassoc nnan ninf nsz arcp contract une float %a, %b
; CHECK-LLVM-16: %r59 = fcmp nnan ninf une float %a, %b
; CHECK-LLVM-16: %r60 = fcmp ult float %a, %b
; CHECK-LLVM-16: %r61 = fcmp nnan ult float %a, %b
; CHECK-LLVM-16: %r62 = fcmp ninf ult float %a, %b
; CHECK-LLVM-16: %r63 = fcmp nsz ult float %a, %b
; CHECK-LLVM-16: %r64 = fcmp arcp ult float %a, %b
; CHECK-LLVM-16-DEFAULT: %r65 = fcmp fast ult float %a, %b
; CHECK-LLVM-16-FC2: %r65 = fcmp reassoc nnan ninf nsz arcp contract ult float %a, %b
; CHECK-LLVM-16: %r66 = fcmp nnan ninf ult float %a, %b
; CHECK-LLVM-16: %r67 = fcmp ugt float %a, %b
; CHECK-LLVM-16: %r68 = fcmp nnan ugt float %a, %b
; CHECK-LLVM-16: %r69 = fcmp ninf ugt float %a, %b
; CHECK-LLVM-16: %r70 = fcmp nsz ugt float %a, %b
; CHECK-LLVM-16: %r71 = fcmp arcp ugt float %a, %b
; CHECK-LLVM-16-DEFAULT: %r72 = fcmp fast ugt float %a, %b
; CHECK-LLVM-16-FC2: %r72 = fcmp reassoc nnan ninf nsz arcp contract ugt float %a, %b
; CHECK-LLVM-16: %r73 = fcmp nnan ninf ugt float %a, %b
; CHECK-LLVM-16: %r74 = fcmp ule float %a, %b
; CHECK-LLVM-16: %r75 = fcmp nnan ule float %a, %b
; CHECK-LLVM-16: %r76 = fcmp ninf ule float %a, %b
; CHECK-LLVM-16: %r77 = fcmp nsz ule float %a, %b
; CHECK-LLVM-16: %r78 = fcmp arcp ule float %a, %b
; CHECK-LLVM-16-DEFAULT: %r79 = fcmp fast ule float %a, %b
; CHECK-LLVM-16-FC2: %r79 = fcmp reassoc nnan ninf nsz arcp contract ule float %a, %b
; CHECK-LLVM-16: %r80 = fcmp nnan ninf ule float %a, %b
; CHECK-LLVM-16: %r81 = fcmp uge float %a, %b
; CHECK-LLVM-16: %r82 = fcmp nnan uge float %a, %b
; CHECK-LLVM-16: %r83 = fcmp ninf uge float %a, %b
; CHECK-LLVM-16: %r84 = fcmp nsz uge float %a, %b
; CHECK-LLVM-16: %r85 = fcmp arcp uge float %a, %b
; CHECK-LLVM-16-DEFAULT: %r86 = fcmp fast uge float %a, %b
; CHECK-LLVM-16-FC2: %r86 = fcmp reassoc nnan ninf nsz arcp contract uge float %a, %b
; CHECK-LLVM-16: %r87 = fcmp nnan ninf uge float %a, %b
; CHECK-LLVM-16: %r88 = fcmp uno float %a, %b
; CHECK-LLVM-16: %r89 = fcmp ninf uno float %a, %b
; CHECK-LLVM-16: %r90 = fcmp nsz uno float %a, %b

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testFCmp(float %a, float %b) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %tmp = alloca i1, align 1
  %r1 = fcmp oeq float %a, %b
  store volatile i1 %r1, ptr %tmp, align 1
  %r2 = fcmp nnan oeq float %a, %b
  store volatile i1 %r2, ptr %tmp, align 1
  %r3 = fcmp ninf oeq float %a, %b
  store volatile i1 %r3, ptr %tmp, align 1
  %r4 = fcmp nsz oeq float %a, %b
  store volatile i1 %r4, ptr %tmp, align 1
  %r5 = fcmp arcp oeq float %a, %b
  store volatile i1 %r5, ptr %tmp, align 1
  %r6 = fcmp fast oeq float %a, %b
  store volatile i1 %r6, ptr %tmp, align 1
  %r7 = fcmp nnan ninf oeq float %a, %b
  store volatile i1 %r7, ptr %tmp, align 1
  %r8 = fcmp one float %a, %b
  store volatile i1 %r8, ptr %tmp, align 1
  %r9 = fcmp nnan one float %a, %b
  store volatile i1 %r9, ptr %tmp, align 1
  %r10 = fcmp ninf one float %a, %b
  store volatile i1 %r10, ptr %tmp, align 1
  %r11 = fcmp nsz one float %a, %b
  store volatile i1 %r11, ptr %tmp, align 1
  %r12 = fcmp arcp one float %a, %b
  store volatile i1 %r12, ptr %tmp, align 1
  %r13 = fcmp fast one float %a, %b
  store volatile i1 %r13, ptr %tmp, align 1
  %r14 = fcmp nnan ninf one float %a, %b
  store volatile i1 %r14, ptr %tmp, align 1
  %r15 = fcmp olt float %a, %b
  store volatile i1 %r15, ptr %tmp, align 1
  %r16 = fcmp nnan olt float %a, %b
  store volatile i1 %r16, ptr %tmp, align 1
  %r17 = fcmp ninf olt float %a, %b
  store volatile i1 %r17, ptr %tmp, align 1
  %r18 = fcmp nsz olt float %a, %b
  store volatile i1 %r18, ptr %tmp, align 1
  %r19 = fcmp arcp olt float %a, %b
  store volatile i1 %r19, ptr %tmp, align 1
  %r20 = fcmp fast olt float %a, %b
  store volatile i1 %r20, ptr %tmp, align 1
  %r21 = fcmp nnan ninf olt float %a, %b
  store volatile i1 %r21, ptr %tmp, align 1
  %r22 = fcmp ogt float %a, %b
  store volatile i1 %r22, ptr %tmp, align 1
  %r23 = fcmp nnan ogt float %a, %b
  store volatile i1 %r23, ptr %tmp, align 1
  %r24 = fcmp ninf ogt float %a, %b
  store volatile i1 %r24, ptr %tmp, align 1
  %r25 = fcmp nsz ogt float %a, %b
  store volatile i1 %r25, ptr %tmp, align 1
  %r26 = fcmp arcp ogt float %a, %b
  store volatile i1 %r26, ptr %tmp, align 1
  %r27 = fcmp fast ogt float %a, %b
  store volatile i1 %r27, ptr %tmp, align 1
  %r28 = fcmp nnan ninf ogt float %a, %b
  store volatile i1 %r28, ptr %tmp, align 1
  %r29 = fcmp ole float %a, %b
  store volatile i1 %r29, ptr %tmp, align 1
  %r30 = fcmp nnan ole float %a, %b
  store volatile i1 %r30, ptr %tmp, align 1
  %r31 = fcmp ninf ole float %a, %b
  store volatile i1 %r31, ptr %tmp, align 1
  %r32 = fcmp nsz ole float %a, %b
  store volatile i1 %r32, ptr %tmp, align 1
  %r33 = fcmp arcp ole float %a, %b
  store volatile i1 %r33, ptr %tmp, align 1
  %r34 = fcmp fast ole float %a, %b
  store volatile i1 %r34, ptr %tmp, align 1
  %r35 = fcmp nnan ninf ole float %a, %b
  store volatile i1 %r35, ptr %tmp, align 1
  %r36 = fcmp oge float %a, %b
  store volatile i1 %r36, ptr %tmp, align 1
  %r37 = fcmp nnan oge float %a, %b
  store volatile i1 %r37, ptr %tmp, align 1
  %r38 = fcmp ninf oge float %a, %b
  store volatile i1 %r38, ptr %tmp, align 1
  %r39 = fcmp nsz oge float %a, %b
  store volatile i1 %r39, ptr %tmp, align 1
  %r40 = fcmp arcp oge float %a, %b
  store volatile i1 %r40, ptr %tmp, align 1
  %r41 = fcmp fast oge float %a, %b
  store volatile i1 %r41, ptr %tmp, align 1
  %r42 = fcmp nnan ninf oge float %a, %b
  store volatile i1 %r42, ptr %tmp, align 1
  %r43 = fcmp ord float %a, %b
  store volatile i1 %r43, ptr %tmp, align 1
  %r44 = fcmp ninf ord float %a, %b
  store volatile i1 %r44, ptr %tmp, align 1
  %r45 = fcmp nsz ord float %a, %b
  store volatile i1 %r45, ptr %tmp, align 1
  %r46 = fcmp ueq float %a, %b
  store volatile i1 %r46, ptr %tmp, align 1
  %r47 = fcmp nnan ueq float %a, %b
  store volatile i1 %r47, ptr %tmp, align 1
  %r48 = fcmp ninf ueq float %a, %b
  store volatile i1 %r48, ptr %tmp, align 1
  %r49 = fcmp nsz ueq float %a, %b
  store volatile i1 %r49, ptr %tmp, align 1
  %r50 = fcmp arcp ueq float %a, %b
  store volatile i1 %r50, ptr %tmp, align 1
  %r51 = fcmp fast ueq float %a, %b
  store volatile i1 %r51, ptr %tmp, align 1
  %r52 = fcmp nnan ninf ueq float %a, %b
  store volatile i1 %r52, ptr %tmp, align 1
  %r53 = fcmp une float %a, %b
  store volatile i1 %r53, ptr %tmp, align 1
  %r54 = fcmp nnan une float %a, %b
  store volatile i1 %r54, ptr %tmp, align 1
  %r55 = fcmp ninf une float %a, %b
  store volatile i1 %r55, ptr %tmp, align 1
  %r56 = fcmp nsz une float %a, %b
  store volatile i1 %r56, ptr %tmp, align 1
  %r57 = fcmp arcp une float %a, %b
  store volatile i1 %r57, ptr %tmp, align 1
  %r58 = fcmp fast une float %a, %b
  store volatile i1 %r58, ptr %tmp, align 1
  %r59 = fcmp nnan ninf une float %a, %b
  store volatile i1 %r59, ptr %tmp, align 1
  %r60 = fcmp ult float %a, %b
  store volatile i1 %r60, ptr %tmp, align 1
  %r61 = fcmp nnan ult float %a, %b
  store volatile i1 %r61, ptr %tmp, align 1
  %r62 = fcmp ninf ult float %a, %b
  store volatile i1 %r62, ptr %tmp, align 1
  %r63 = fcmp nsz ult float %a, %b
  store volatile i1 %r63, ptr %tmp, align 1
  %r64 = fcmp arcp ult float %a, %b
  store volatile i1 %r64, ptr %tmp, align 1
  %r65 = fcmp fast ult float %a, %b
  store volatile i1 %r65, ptr %tmp, align 1
  %r66 = fcmp nnan ninf ult float %a, %b
  store volatile i1 %r66, ptr %tmp, align 1
  %r67 = fcmp ugt float %a, %b
  store volatile i1 %r67, ptr %tmp, align 1
  %r68 = fcmp nnan ugt float %a, %b
  store volatile i1 %r68, ptr %tmp, align 1
  %r69 = fcmp ninf ugt float %a, %b
  store volatile i1 %r69, ptr %tmp, align 1
  %r70 = fcmp nsz ugt float %a, %b
  store volatile i1 %r70, ptr %tmp, align 1
  %r71 = fcmp arcp ugt float %a, %b
  store volatile i1 %r71, ptr %tmp, align 1
  %r72 = fcmp fast ugt float %a, %b
  store volatile i1 %r72, ptr %tmp, align 1
  %r73 = fcmp nnan ninf ugt float %a, %b
  store volatile i1 %r73, ptr %tmp, align 1
  %r74 = fcmp ule float %a, %b
  store volatile i1 %r74, ptr %tmp, align 1
  %r75 = fcmp nnan ule float %a, %b
  store volatile i1 %r75, ptr %tmp, align 1
  %r76 = fcmp ninf ule float %a, %b
  store volatile i1 %r76, ptr %tmp, align 1
  %r77 = fcmp nsz ule float %a, %b
  store volatile i1 %r77, ptr %tmp, align 1
  %r78 = fcmp arcp ule float %a, %b
  store volatile i1 %r78, ptr %tmp, align 1
  %r79 = fcmp fast ule float %a, %b
  store volatile i1 %r79, ptr %tmp, align 1
  %r80 = fcmp nnan ninf ule float %a, %b
  store volatile i1 %r80, ptr %tmp, align 1
  %r81 = fcmp uge float %a, %b
  store volatile i1 %r81, ptr %tmp, align 1
  %r82 = fcmp nnan uge float %a, %b
  store volatile i1 %r82, ptr %tmp, align 1
  %r83 = fcmp ninf uge float %a, %b
  store volatile i1 %r83, ptr %tmp, align 1
  %r84 = fcmp nsz uge float %a, %b
  store volatile i1 %r84, ptr %tmp, align 1
  %r85 = fcmp arcp uge float %a, %b
  store volatile i1 %r85, ptr %tmp, align 1
  %r86 = fcmp fast uge float %a, %b
  store volatile i1 %r86, ptr %tmp, align 1
  %r87 = fcmp nnan ninf uge float %a, %b
  store volatile i1 %r87, ptr %tmp, align 1
  %r88 = fcmp uno float %a, %b
  store volatile i1 %r88, ptr %tmp, align 1
  %r89 = fcmp ninf uno float %a, %b
  store volatile i1 %r89, ptr %tmp, align 1
  %r90 = fcmp nsz uno float %a, %b
  store volatile i1 %r90, ptr %tmp, align 1
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
