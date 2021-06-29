#!/bin/bash
input="./ytids.txt"
while IFS= read -r line
do
  echo "Downloading $line"
  youtube-dl --cookies=~/bin/cookies.txt -f 133 $line
done < "$input"
