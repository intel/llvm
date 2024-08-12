#!/bin/bash

for file in *.cpp; do
  if grep -q "// RUN: %{build} -o %t.out" "$file"; then
    sed -i 's|// RUN: %{build} -o %t.out|// RUN: %{build} -D__SPIRV_USE_COOPERATIVE_MATRIX -o %t.out|' "$file"
    echo "Updated $file"
  else
    echo "No matching line found in $file"
  fi

  if grep -q '#include "' "$file"; then
    sed -i -E 's|#include "([^"]+)"|#include "../\1"|' "$file"
    echo "Updated all #include lines in $file"
  else
    echo "No #include lines found in $file"
  fi

done

