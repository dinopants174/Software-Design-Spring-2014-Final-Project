import keyword
import re

# build keyword dictionaries
py_list = keyword.kwlist
matlab_list = ['break', 'case', 'catch', 'continue', 'disp', 'else', 'elseif', 'end', 'for', 'function', 'global', 'if', 'otherwise', 'persistent', 'return', 'switch', 'try', 'while']

py_dict = {}
matlab_dict = {}

for keyword in py_list:
	py_dict[keyword] = 0

for keyword in matlab_list:
	matlab_dict[keyword] = 0

# overlap = ''
# for keyword in py_list:
# 	if keyword in matlab_list:
# 		overlap += keyword + '/'

# analyze string

f = open('twoD.m','r')
input_string = f.read()
f.close()
string_to_list = re.split('[^a-zA-z_]+', input_string)

for word in string_to_list:
	if word in py_dict:
		py_dict[word] += 1
	if word in matlab_dict:
		matlab_dict[word] += 1

py_occurence = 0
for key in py_dict:
	py_occurence += py_dict[key]

matlab_occurence = 0
for key in matlab_dict:
	matlab_occurence += matlab_dict[key]

print py_occurence /  float(matlab_occurence+py_occurence)
print matlab_occurence /  float(matlab_occurence+py_occurence)
print py_dict
print matlab_dict