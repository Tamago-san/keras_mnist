#set multiplot layout 2,2
set grid
#set terminal windows enhanced
#set output

set xlabel "t"
#set style line 1 lt 2 lw 3
#set style line 2 lt 0 lw 2a

#set xlabel "t"
#set ylabel "y"


plot "a.dat"  using 1 with line linewidth 2.5 linecolor rgb "#0011ff"

