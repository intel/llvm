#!/bin/bash
############################################################################################
#
# Script to launch a SYCL executable and collect information regarding the run
#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
############################################################################################
# Set the color variable
green='\033[0;32m'
yellow='\033[0;33m'
cyan='\033[0;36m'
red='\033[0;31m'
# Clear the color after that
clear='\033[0m'

#export XPTI_PERF_DIR=/home/vtovink/sycl_workspace/xpti/llvm/xptifw/lib/RelWithDebInfo
help()
{
    echo -e "Usage: sycl-perf.sh [-f,--format] ${yellow}<Value1> ${clear}[-c,--color] [-s,--streams] ${yellow}<Value2> ${clear}[-h,--help] -- <executable> <arguments>"
    echo -e "          ${green}-f,--format    Allowed values for ${yellow}<Value1>${green} are ${yellow}json,csv,table,stack,all${green}"
    echo "          -c,--color     Boolean option, if provided will display output in color for stack data"
    echo "          -s,--streams   Streams to monitor in the SYCL runtime. Multiple streams can be provided"
    echo -e "                         as comma separated values for ${yellow}<Value2>${green}"
    echo -e "                         Example:- ${yellow}sycl,sycl.pi,sycl.perf ${clear}"
    echo "  "
    echo -e "                         The script requires you to set the environment variable XPTI_PERF_DIR"
    echo -e "                         before executing this script."
    echo -e "                         Example: ${cyan}export XPTI_PERF_DIR=/path/to/xptifw/lib/RelWithDebInfo${clear}"
    echo "  "
    echo "  "
    exit 2
}


export XPTI_TRACE_ENABLE=1
if [ -v XPTI_PERF_DIR ]; then 
   export XPTI_FRAMEWORK_DISPATCHER="$XPTI_PERF_DIR""/libxptifw.so"
   export XPTI_SUBSCRIBERS="$XPTI_PERF_DIR""/libsycl_perf_collector.so"
else
   echo -e "${red}Environment variable XPTI_PERF_DIR is not set${clear}"
   help
fi


if [ "$#" -eq 0 ]; then
  help
fi

############################################################################################
#
# Argument capture 
#
############################################################################################

# Default color is normal text and we change the colors only when the option is provided
color=0

SHORT=f:,s:,h,c
LONG=format:,streams:,help,color
OPTS=$(getopt -a -n sycl-perf --options $SHORT --longoptions $LONG -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$OPTS"

while [ : ]; do
  case "$1" in
    -f | --format)
        format="$2"
        shift 2
        ;;
    -c | --color)
        color=1
        shift;
        ;;
    -s | --streams)
        streams="$2"
        shift 2
        ;;
    -h | --help)
     	help
     	;;
    --) 
        shift; 
        break 
        ;;
  esac
done

############################################################################################
#
# Argument validation and environment setup
#
############################################################################################

if [[ "$format" == "json" || "$format" == "csv" || "$format" == "table" || "$format" == "stack" || "$format" == "all" ]]; then
export XPTI_SYCL_PERF_OUTPUT=$format
else
echo "Bad --format="$format " provided, will default to collector implementation.."
fi

if [[ -n "$streams"  ]]; then
export XPTI_STREAMS=$streams
fi

if [[ $color == 1 || $color == 0 ]]; then
export XPTI_STDOUT_USE_COLOR=$color
fi

############################################################################################
#
# Executing the SYCL program to get the telemetry data from
#
############################################################################################

$@
