+// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -fsyntax-only %s -ast-dump -Wno-ignored-attributes | FileCheck %s
+
+// Check that when an invalid value is specified for the attribute
+// intel::named_sub_group_size, an invalid value is not added to the
+// function declaration.
+
+// CHECK: FunctionDecl {{.*}} f1 'void ()'
+// CHECK-NOT: IntelNamedSubGroupSizeAttr
+[[intel::named_sub_group_size(invalid)]] void f1();
+// CHECK: FunctionDecl {{.*}} f2 'void ()'
+// CHECK-NOT: IntelNamedSubGroupSizeAttr
+[[intel::named_sub_group_size("invalid string")]] void f2();
