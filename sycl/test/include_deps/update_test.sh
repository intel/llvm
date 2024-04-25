echo -e "// Use update_test.sh to (re-)generate the checks" > test.cpp
echo -e "// REQUIRES: linux" >> test.cpp
echo -e "// RUN: bash %S/deps_known.sh | FileCheck %s\n" >> test.cpp
bash deps_known.sh | sed 's@^@// CHECK-NEXT: @' | sed 's@CHECK-NEXT: Dependencies@CHECK-LABEL: Dependencies@' | sed 's@CHECK-NEXT: $@CHECK-EMPTY:@' >> test.cpp
