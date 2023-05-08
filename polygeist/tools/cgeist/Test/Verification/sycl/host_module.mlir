module {
memref.global constant @c : memref<i32, 1> {alignment = 4 : i64}
func.func @do_nothing() -> memref<i32, 1> {
  %0 = memref.get_global @c : memref<i32, 1>
  return %0 : memref<i32, 1>
}
}
