<%!
import re
from templates import helper as th
%><%
	n=namespace
%>\
LIBRARY @TARGET_LIBNAME@
EXPORTS
%for line in th.get_loader_functions(specs, meta, n, tags):
	${line}
%endfor
