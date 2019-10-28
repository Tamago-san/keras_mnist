unset key
set tics font "Arial,10"
set title font "Arial,15"
set xlabel font "Arial,15"
set ylabel font "Arial,15"
set y2label font "Arial,15"
set zlabel font "Arial,15"
set key font "Arial,15"
set palette rgb 33,13,10
set xrange[-2:8]
set yrange[-2.5:2.5]
#set size square
set size ratio 0.5
set cbrange[0:0.6]
unset colorbox
#unset c
idts=1200
idte=1200
iRe=10
dtdt=1
#./data_out/data_out/velocity.000500'u 1:2:($3*1.0):($4*1.0) w vec lc "#0000FF"
#
#set terminal gif animate delay 7 optimize size 640,320
#outlocation= sprintf("./image/Wout_step.gif")
#set output outlocation
set style fill transparent solid 0.5 noborder
unset key
do for [i=idts:idte:5] {
#    outlocation= sprintf("./image/VxVy/Wout_RE-%03d.png", iRe)
#    outlocation= sprintf("./image/VxVy/step-%05d.eps", i)
    outlocation= sprintf("./step-%05d.png", i)


    set terminal png
#    set terminal postscript eps color enhanced "san-serif" 20

    set output outlocation

#    titlename= sprintf("dtL/dtN=%02d , Re=%03d, step=%04d", dtdt, iRe,i)
    titlename= sprintf("dt=0.01 , Re=%03d, step=%04d", iRe,i)
    filename = sprintf("./data_out/data_out/velocity.%06d", i)
    set title titlename
    plot './data/circle.dat'w filledcurves lc rgb "gray",\
         filename  u 1:2:($3*1.0):($4*1.0):(abs($3)*2) w vector lw 2 lc palette

}

