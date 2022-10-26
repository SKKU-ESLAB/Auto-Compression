#!/bin/sh

echo "Assume that there is libpimss.so\n"

gcc -o test test.c -L. -lpimss -lstdc++
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.

