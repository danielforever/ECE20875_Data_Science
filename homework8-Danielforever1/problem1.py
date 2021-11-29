#Arguments:
#  filename: name of file to read in
#Returns: a list of strings, each string is one line in the file
#hints: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files
#       https://docs.python.org/3/library/stdtypes.html#str.splitlines
from hw8_1 import *
import sys


if __name__=="__main__":
	filename = sys.argv[1]
	dict = getDict(filename)
	print(getDict(filename))
	

