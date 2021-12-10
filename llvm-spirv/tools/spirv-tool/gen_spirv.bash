#!/usr/bin/bash
# script to generate code for LLVM/SPIR-V translator based on khronos 
# header file spirv.hpp.
#

# The NameMaps that will be generated into SPIRVNameMapEnum.h
nameMapEnums="LinkageType Decoration BuiltIn Capability"

# The isValid functions that will be generated into SPIRVIsValidEnum.h
isValidEnums="ExecutionModel AddressingModel MemoryModel StorageClass \
              LinkageType AccessQualifier FunctionParameterAttribute BuiltIn"

# The isValidxxxMask functions that will be generated into SPIRVIsValidEnum.h
isValidMaskEnums="FunctionControlMask"


######################
#
# generate NameMap
#
######################

genNameMap() {
prefix=$1
echo "template <> inline void SPIRVMap<$prefix, std::string>::init() {"

cat $spirvHeader | sed -n -e "/^ *${prefix}[^a-z]/s:^ *${prefix}\([^= ][^= ]*\)[= ][= ]*\([0x]*[0-9][0-9]*\).*:\1 \2:p"  | while read a b; do
  stringRep="$a"
  if [[ $prefix == "BuiltIn" ]]; then
    stringRep="BuiltIn$a"
  fi
  printf "  add(${prefix}%s, \"%s\");\n" "$a" "$stringRep"
done

echo "}
SPIRV_DEF_NAMEMAP($prefix, SPIRV${prefix}NameMap)
"

}

###########################
#
# generate isValid function
#
###########################
genIsValid() {
prefix=$1
echo "inline bool isValid(spv::$prefix V) {
  switch (V) {"

  prevValue=
  cat $spirvHeader | sed -n -e "/^ *${prefix}[^a-z]/s:^ *${prefix}\([^= ][^= ]*\)[= ][= ]*\(.*\).*:\1 \2:p"  | while read a b; do
    if [[ "$a" == "Max" ]]; then
      # The "Max" enum value is not valid.
      continue
    fi
    if [[ "$b" == "$prevValue" ]]; then
      # This enum value is an alias for the previous.  Skip to avoid duplicate case values.
      continue
    fi

    printf "  case ${prefix}%s:\n" "$a"
    prevValue=$b
  done

echo "    return true;
  default:
    return false;
  }
}
"
}
genMaskIsValid() {
prefix=$1
subprefix=`echo $prefix | sed -e "s:Mask::g"`
echo "inline bool isValid$prefix(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;"

  cat $spirvHeader | sed -n -e "/^ *${subprefix}[^a-z]/s:^ *${subprefix}\([^= ][^= ]*\)Mask[= ][= ]*\(.*\).*:\1 \2:p"  | while read a b; do
  if [[ $a == None ]]; then
    continue
  fi
  printf "  ValidMask |= ${subprefix}%sMask;\n" $a
done

echo "
  return (Mask & ~ValidMask) == 0;
}
"
}

gen() {
type=$1
if [[ "$type" == NameMap ]]; then
  for prefix in ${nameMapEnums} ; do
    genNameMap "$prefix"
  done
elif [[ "$type" == isValid ]]; then
  for prefix in ${isValidEnums} ; do
    genIsValid "$prefix"
  done
  for prefix in ${isValidMaskEnums} ; do
    genMaskIsValid "$prefix"
  done
else
  echo "invalid type \"$type\"."
  exit
fi
}

genFile() {
  outputFile=$1
  genType=$2
  outputBasename="$(basename ${outputFile})"
  includeGuard="SPIRV_LIBSPIRV_`echo ${outputBasename} | tr '[:lower:]' '[:upper:]' | sed -e 's/[\.\/]/_/g'`"

  echo "//===- ${outputBasename} - SPIR-V ${genType} enums ----------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the \"Software\"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \\file
///
/// This file defines SPIR-V ${genType} enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using \`tools/spirv-tool/gen_spirv.bash\` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef ${includeGuard}
#define ${includeGuard}

#include \"SPIRVEnum.h\"
#include \"spirv.hpp\"
#include \"spirv_internal.hpp\"

using namespace spv;

namespace SPIRV {
" > ${outputFile}

  gen $genType >> ${outputFile}

  echo "} /* namespace SPIRV */

#endif // ${includeGuard}" >> ${outputFile}
}

####################
#
# main
#
####################

if [[ $# -ne 1 ]]; then
  echo "usage: gen_spirv path_to_spirv.hpp"
  exit
fi

spirvHeader=$1
genFile "lib/SPIRV/libSPIRV/SPIRVNameMapEnum.h" NameMap
genFile "lib/SPIRV/libSPIRV/SPIRVIsValidEnum.h" isValid
