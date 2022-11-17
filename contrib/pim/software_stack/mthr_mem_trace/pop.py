import os

name = str(input())

out_f = open("pim_"+name+".txt", "w")
in_f = open(name+".txt", "r")
flag = 0

lines = in_f.readlines()
for line in lines:
    items = line.split()
    if (items[0] == '3'):
        break

    if (flag == 1):
        out_f.write(line)
   
    if (items[0] == '2'):
        flag = flag + 1;


out_f.close()
in_f.close()
