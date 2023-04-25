#!/bin/bash
ITER=10

for((Xp=1;Xp<=30;Xp+=5))
do
  for((Yp=${Xp};Yp<=30;Yp+=5))
    do
	 for((i=0;i<ITER;i++))
	 do	 
          echo `/usr/bin/time -a -f"${Xp} ${Yp} %E" ./mandelbrot_threadPool -1.5 1.2 0.9 -1.2 ${Xp} ${Yp}`
      done
     done
done
