declare i32 @llvm.ctpop.i32(i32 %n)
declare i8 @llvm.ctpop.i8(i8 %n)


define dso_local i32 @_Z17__popcount_helperi(i32 %x) {
entry:
  %call = call i32 @llvm.ctpop.i32(i32 %x) 
  ret i32 %call
}


define dso_local i8 @_Z17__popcount_helpera(i8 %x) {
entry:
  %call = call i8 @llvm.ctpop.i8(i8 %x) 
  ret i8 %call
}

