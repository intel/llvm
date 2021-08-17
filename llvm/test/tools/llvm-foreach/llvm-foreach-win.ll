; REQUIRES: system-windows
;
; RUN: echo 'Content of first file' > %t1.tgt
; RUN: echo 'Content of second file' > %t2.tgt
; RUN: echo "%t1.tgt" > %t.list
; RUN: echo "%t2.tgt" >> %t.list
; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.list -- echo "{}" > %t.res
; RUN: FileCheck < %t.res %s
; RUN: llvm-foreach --parallel-exec --in-replace="{}" --in-file-list=%t.list -- echo "{}" > %t.res
; RUN: FileCheck < %t.res %s
; CHECK-DAG: [[FIRST:.+1.tgt]]
; CHECK-DAG: [[SECOND:.+2.tgt]]
;
; RUN: llvm-foreach --in-replace="{}" --out-replace=%t --out-ext=out --in-file-list=%t.list --out-file-list=%t.out.list -- xcopy /y "{}" %t
; RUN: FileCheck < %t.out.list %s --check-prefix=CHECK-LIST
; RUN: llvm-foreach --parallel-exec --in-replace="{}" --out-replace=%t --out-ext=out --in-file-list=%t.list --out-file-list=%t.out.list -- xcopy /y "{}" %t
; RUN: FileCheck < %t.out.list %s --check-prefix=CHECK-LIST
; CHECK-LIST-DAG: [[FIRST:.+\.out]]
; CHECK-LIST-DAG: [[SECOND:.+\.out]]
; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.out.list -- FileCheck --input-file="{}" %s --check-prefix=CHECK-CONTENT
; CHECK-CONTENT: Content of

; RUN: echo 'something' > %t3.tgt
; RUN: echo 'something again' > %t4.tgt
; RUN: echo "%t3.tgt" > %t1.list
; RUN: echo "%t4.tgt" >> %t1.list
; RUN: llvm-foreach --in-replace="{}" --in-replace="inrep" --in-file-list=%t.list --in-file-list=%t1.list --out-increment=%t_out.prj -- echo -first-part-of-arg={}.out -first-part-of-arg=inrep.out -another-arg=%t_out.prj > %t1.res
; RUN: FileCheck < %t1.res %s --check-prefix=CHECK-DOUBLE-LISTS
; RUN: llvm-foreach --parallel-exec --in-replace="{}" --in-replace="inrep" --in-file-list=%t.list --in-file-list=%t1.list --out-increment=%t_out.prj -- echo -first-part-of-arg={}.out -first-part-of-arg=inrep.out -another-arg=%t_out.prj > %t1.res
; RUN: FileCheck < %t1.res %s --check-prefix=CHECK-DOUBLE-LISTS
; CHECK-DOUBLE-LISTS-DAG: -first-part-of-arg=[[FIRST:.+1.tgt.out]] -first-part-of-arg=[[THIRD:.+3.tgt.out]] -another-arg={{.+}}_out.prj
; CHECK-DOUBLE-LISTS-DAG: -first-part-of-arg=[[SECOND:.+2.tgt.out]] -first-part-of-arg=[[FOURTH:.+4.tgt.out]] -another-arg={{.+}}_out.prj_1
