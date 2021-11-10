from os import listdir
from os.path import isfile, join
import os

mypath = 'images/faces/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)


file1 = onlyfiles[0]
fileName, fileExtension = os.path.splitext(file1)

print(fileName)
print(fileExtension)


for i in range(9,10):
	print(i)

print("asd")