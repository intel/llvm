<%
import re
from templates import helper as th
import datetime
%><%
    year_now=datetime.date.today().year
%>#
# Copyright (C) ${year_now} Intel Corporation
#
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
# Exceptions.
# See LICENSE.TXT
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# See YaML.md for syntax definition
#
--- #--------------------------------------------------------------------------
type: header
desc: "Intel $OneApi Unified Runtime Experimental APIs for ${" ".join(name.split("-")).title()}"
ordinal: "99"
--- #--------------------------------------------------------------------------
type: macro
desc: |
      The extension string which defines support for ${name}
      which is returned when querying device extensions.
name: $X_${"_".join(name.split("-")).upper()}_EXTENSION_STRING_EXP
value: "\"$x_exp_${"_".join(name.split("-"))}\""
