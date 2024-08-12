#!/bin/bash

find . -name "*.cpp" | while read -r file; do

  line_number=$(grep -n -m 1 "// REQUIRES: .*" "$file" | cut -d: -f1)

  if [ ! -z "$line_number" ]; then
    sed -i "${line_number}s|// REQUIRES: |// REQUIRES: gpu, |" "$file"
    echo "Updated line $line_number in $file"
  else
    echo "No matching line found in $file"
  fi

done

