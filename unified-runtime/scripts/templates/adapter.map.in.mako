<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
@TARGET_LIBNAME@ {
	global:
%for tbl in th.get_pfntables(specs, meta, n, tags):
		${tbl['export']['name']};
%endfor
	local:
		*;
};
