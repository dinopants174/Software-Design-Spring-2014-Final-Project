import keyword
import re

# build keyword dictionaries
py_list = keyword.kwlist
matlab_list = ['break', 'case', 'catch', 'continue', 'disp', 'else', 'elseif', 'end', 'for', 'function', 'global', 'if', 'otherwise', 'persistent', 'return', 'switch', 'try', 'while']
c_list = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern','float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while']

py_dict = {}
matlab_dict = {}
c_dict = {}

for keyword in py_list:
	py_dict[keyword] = 0

for keyword in matlab_list:
	matlab_dict[keyword] = 0

for keyword in c_list:
	c_dict[keyword] = 0

# overlap = ''
# for keyword in py_list:
# 	if keyword in matlab_list:
# 		overlap += keyword + '/'

# analyze string

f = open('random_art.py','r')
input_string = f.read()
f.close()
string_to_list = re.split('[^a-zA-z_]+', input_string)

for word in string_to_list:
	if word in py_dict:
		py_dict[word] += 1
	if word in matlab_dict:
		matlab_dict[word] += 1
	if word in c_dict:
		c_dict[word] += 1

py_occurence = 0
for key in py_dict:
	py_occurence += py_dict[key]

matlab_occurence = 0
for key in matlab_dict:
	matlab_occurence += matlab_dict[key]

c_occurrence = 0
for key in c_dict:
	c_occurrence += c_dict[key]

print py_occurence /  float(matlab_occurence+py_occurence+c_occurrence)
print matlab_occurence /  float(matlab_occurence+py_occurence+c_occurrence)
print c_occurrence /  float(matlab_occurence+py_occurence+c_occurrence)
