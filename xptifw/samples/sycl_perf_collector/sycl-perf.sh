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

help()
{
    echo -e "Usage: sycl-perf.sh [-f,--format] ${yellow}<Value1> ${clear}[-i,--ignore] ${yellow}<Value2> ${clear}[-p,--projection] ${yellow}<Value3> ${clear}[-c,--color] [-s,--streams] ${yellow}<Value4> ${clear}[-h,--help] [-v, --verbose] [-d, --debug] -- <executable> <arguments>"
    echo -e "          ${green}-f,--format     Allowed values for ${yellow}<Value1>${green} are ${yellow}json,table,stack,all,none${green}"
    echo "          -i,--ignore     First time execution of certain calls take an order of magnitude more"
    echo -e "                          time than subsequent calls listed in ${yellow}<Value2>${green}"
    echo -e "                          Example:- ${yellow}piPlatformGet,piProgramBuild ${green}"
    echo "          -p,--projection Sequence of comma separated instrumentation costs in nanoseconds that will be"
    echo -e "                          used to simulate and project the impact of the overhead due to instrumentation"
    echo -e "                          as provided in ${yellow}<Value3>${green}"
    echo -e "                          Example:- ${yellow}10,50,150 ${green}"
    echo "          -c,--color      Boolean option, if provided will display output in color for stack data"
    echo "          -s,--streams    Streams to monitor in the SYCL runtime. Multiple streams can be provided"
    echo -e "                          as comma separated values for ${yellow}<Value4>${green}"
    echo -e "                          Example:- ${yellow}sycl,sycl.pi,sycl.perf ${clear}${green}"
    echo "          -e,--calibrate  Boolean option, if provided will run the application with an empty collector."
    echo "          -v,--verbose    Boolean option, if provided will display verbose output of the collector status"
    echo "          -d,--debug      Boolean option, if provided will display debug output, including saved record information."
    echo "  "
    echo -e "                          The script requires you to set the environment variable XPTI_PERF_DIR"
    echo -e "                          before executing this script."
    echo -e "                          Example: ${cyan}export XPTI_PERF_DIR=/path/to/xptifw/lib/RelWithDebInfo${clear}"
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
calibrate_flag=0
verbose_flag=0
debug_flag=0

SHORT=f:,s:,i:,p:,h,c,e,v,d
LONG=format:,streams:,ignore:,projection:,help,color,calibrate,verbose,debug
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
    -i | --ignore)
        ignore_list="$2"
        shift 2
        ;;
    -p | --projection)
        projection="$2"
        shift 2
        ;;
    -c | --color)
        color=1
        shift;
        ;;
    -e | --calibrate)
        calibrate_flag=1
        shift;
        ;;
    -v | --verbose)
        verbose_flag=1
        shift;
        ;;
    -d | --debug)
        debug_flag=1
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

if [[ -n $format ]]; then
export XPTI_SYCL_PERF_OUTPUT=$format
fi

if [[ -n "$ignore_list"  ]]; then
export XPTI_IGNORE_LIST=$ignore_list
fi

if [[ -n "$projection"  ]]; then
export XPTI_SIMULATION=$projection
fi

if [[ -n "$streams"  ]]; then
export XPTI_STREAMS=$streams
fi

if [[ $color == 1 || $color == 0 ]]; then
export XPTI_STDOUT_USE_COLOR=$color
fi

if [[ $calibrate_flag == 1 || $calibrate_flag == 0 ]]; then
export XPTI_CALIBRATE=$calibrate_flag
fi

if [[ $verbose_flag == 1 || $verbose_flag == 0 ]]; then
export XPTI_VERBOSE=$verbose_flag
fi

if [[ $debug_flag == 1 || $debug_flag == 0 ]]; then
export XPTI_DEBUG=$debug_flag
fi

############################################################################################
#
# Executing the SYCL program to get the telemetry data from
#
############################################################################################

$@
