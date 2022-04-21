; UNSUPPORTED: system-windows
;
; RUN: echo 'Content of first file' > %t1.tgt
; RUN: echo 'Content of second file' > %t2.tgt
; RUN: echo "%t1.tgt" > %t.list
; RUN: echo "%t2.tgt" >> %t.list
; RUN: llvm-foreach --jobs=2 --in-replace="{}" --in-file-list=%t.list -- echo "{}" > %t.res
; RUN: FileCheck < %t.res %s
; CHECK-DAG: [[FIRST:.+1.tgt]]
; CHECK-DAG: [[SECOND:.+2.tgt]]

; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.list -- echo "{}" > %t.res
; RUN: FileCheck < %t.res %s --check-prefix=CHECK-ORDER
; CHECK-ORDER: [[FIRST:.+1.tgt]]
; CHECK-ORDER: [[SECOND:.+2.tgt]]

; RUN: llvm-foreach --in-replace="{}" --out-replace=%t --out-ext=out --in-file-list=%t.list --out-file-list=%t.out.list -- cp "{}" %t
; RUN: FileCheck < %t.out.list %s --check-prefix=CHECK-LIST
; RUN: llvm-foreach --jobs=2 --in-replace="{}" --out-replace=%t --out-ext=out --in-file-list=%t.list --out-file-list=%t.out.list -- cp "{}" %t
; RUN: FileCheck < %t.out.list %s --check-prefix=CHECK-LIST
; CHECK-LIST: [[FIRST:.+\.out]]
; CHECK-LIST: [[SECOND:.+\.out]]
; RUN: llvm-foreach --in-replace="{}" --in-file-list=%t.out.list -- cat "{}" > %t.order
; RUN: FileCheck < %t.order %s --check-prefix=CHECK-CONTENT
; CHECK-CONTENT: Content of first file
; CHECK-CONTENT-NEXT: Content of second file

; RUN: echo 'something' > %t3.tgt
; RUN: echo 'something again' > %t4.tgt
; RUN: echo "%t3.tgt" > %t1.list
; RUN: echo "%t4.tgt" >> %t1.list
; RUN: llvm-foreach --in-replace="{}" --in-replace="inrep" --in-file-list=%t.list --in-file-list=%t1.list --out-increment="%t_out.prj" -- echo -first-part-of-arg={}.out -first-part-of-arg=inrep.out -another-arg=%t_out.prj > %t1.res
; RUN: FileCheck < %t1.res %s --check-prefix=CHECK-DOUBLE-LISTS
; RUN: llvm-foreach --jobs=2 --in-replace="{}" --in-replace="inrep" --in-file-list=%t.list --in-file-list=%t1.list --out-increment="%t_out.prj" -- echo -first-part-of-arg={}.out -first-part-of-arg=inrep.out -another-arg=%t_out.prj > %t1.res
; RUN: FileCheck < %t1.res %s --check-prefix=CHECK-DOUBLE-LISTS
; CHECK-DOUBLE-LISTS-DAG: -first-part-of-arg=[[FIRST:.+1.tgt.out]] -first-part-of-arg=[[THIRD:.+3.tgt.out]] -another-arg={{.+}}_out.prj
; CHECK-DOUBLE-LISTS-DAG: -first-part-of-arg=[[SECOND:.+2.tgt.out]] -first-part-of-arg=[[FOURTH:.+4.tgt.out]] -another-arg={{.+}}_out.prj_1

; RUN: echo "%t1.tgt" > %t2.list
; RUN: echo "%t2.tgt" >> %t2.list
; RUN: echo "%t3.tgt" >> %t2.list
; RUN: echo "%t4.tgt" >> %t2.list
; RUN: llvm-foreach -j 2 --in-replace="{}" --in-file-list=%t2.list -- echo "{}" > %t2.res
; RUN: FileCheck < %t2.res %s --check-prefix=CHECK-PARALLEL-JOBS
; CHECK-PARALLEL-JOBS-DAG: [[FIRST:.+1.tgt]]
; CHECK-PARALLEL-JOBS-DAG: [[SECOND:.+2.tgt]]
; CHECK-PARALLEL-JOBS-DAG: [[THIRD:.+3.tgt]]
; CHECK-PARALLEL-JOBS-DAG: [[FOURTH:.+4.tgt]]
