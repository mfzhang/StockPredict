#!/bin/sh
files="/home/yoshihara/StockPredict/yoshihara/Model/result/*"
brand=",0101, ,6501, ,6502, ,6702, ,6753, ,6758, ,7201, ,7203, ,7751, ,8031, ,8058,"

clear
for filepath in $files
do
    echo ${filepath}
    for i in $brand
    do
        sort -t "," -k 2 ${filepath} |grep $i|head -1
    done
echo ---------------------------------------------------------------------------------------------------
done
