<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
LIBRARY @TARGET_LIBNAME@
EXPORTS
%for tbl in th.get_pfntables(specs, meta, n, tags):
	${tbl['export']['name']}
%endfor
