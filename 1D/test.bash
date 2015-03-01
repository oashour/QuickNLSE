#!/bin/bash

START=32
NO=50
END=$(expr $START \* $NO)
DELTA=$START
FILE="cpu_timing.m"

rm $FILE

echo -ne "steps = [$START:$DELTA:$END];\n"  >> $FILE 
echo -ne "time = [" >> $FILE

for ((c=$START; c<=$END; c+=$DELTA))
	do
		./new.out $c $FILE
	done

echo -ne "];\n" >> $FILE
echo -ne "plot(steps, time, '-*b');\n" >> $FILE
