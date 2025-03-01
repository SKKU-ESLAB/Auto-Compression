#!/bin/bash

TODAY=$(date +"%y-%m-%d")

git add .
git add -f pip.txt
git commit -m "$TODAY"
git push