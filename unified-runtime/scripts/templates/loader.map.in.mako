<%!
import re
from templates import helper as th
%><%
    n=namespace
%>\
@TARGET_LIBNAME@ {
	global:
%for line in th.get_loader_functions(specs, meta, n, tags):
		${line};
%endfor
	local:
		*;
};
