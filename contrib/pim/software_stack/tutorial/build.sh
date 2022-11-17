#!/bin/sh

echo "Assume that there is libpimss.so\n"

gcc -o test test.c -L. -lpimss

