#!/bin/sh

echo "I'm building C++"

make clean && make && ./test 0
