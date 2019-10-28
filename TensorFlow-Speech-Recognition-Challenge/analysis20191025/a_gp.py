#set terminal postscript enhanced
set terminal png # 出力先をPNGに設定
set output 'test.png' # 出力ファイル名をtest.pngに設定

set grid
set tics font "Arial,10"
set title font "Arial,14"
set xlabel font "Arial,14"
set ylabel font "Arial,14"
set y2label font "Arial,14"
set zlabel font "Arial,13"

set title "RC (tanh, 128node)"
set xlabel "g"
set ylabel "mse"
set y2label "accuracy"
set title offset 0,-0.7
set ylabel offset 3,0
set y2label offset -3,0
set key opaque box lc rgb "white" height 1

set logscale x
set format x "10^{%L}"

set ytics
set y2tics
#set y2tics 0.75,0.02
set ytics nomirror
set y2tics nomirror

set yr[:0.325]
set y2r[:0.92]
plot './data_out/rc_val_erracc_beta.dat' u 1:2 axis x1y1 w lp lw 2 lc rgb "#FF0000" title "validation error",\
'./data_out/rc_val_erracc_beta.dat'u 1:3  axis x1y2 w lp lw 2 lc  rgb"#0000FF" title "validation accuracy"
set output