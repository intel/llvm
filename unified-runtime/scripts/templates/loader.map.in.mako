<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
@TARGET_LIBNAME@ {
	global:
%for func in th.get_loader_functions(specs, meta, n, tags):
		${func['name']};
%endfor
	local:
		*;
};
