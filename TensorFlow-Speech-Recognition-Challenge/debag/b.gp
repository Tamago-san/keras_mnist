set multiplot layout 2,1
unset key
set xrange[-4:8]
set yrange[-3:3]
#set xrange[-2:14]
#set yrange[-4:4]
set size ratio 0.5

set title 'nu=0.04, D=2, U=1.0, time=500'
plot './data_out/data_out/velocity.000500'u 1:2:($3*1.0):($4*1.0) w vec lc "#0000FF"
#,\
#     './data/circle.dat'w filledcurves

set title 'nu=0.04, D=2, U=1.0, time=1000'
plot './data_out/data_out/velocity.005000'u 1:2:($3*1.0):($4*1.0) w vec lc "#0000FF"
#,\
#     './data/circle.dat'w filledcurves

#set title 'nu=0.04, D=2, U=1.0, time=10000'
#plot './data_out/data_out/velocity.010000'u 1:2:($3*1.0):($4*1.0) w vec lc "#0000FF"
##,\
##     './data/circle.dat'w filledcurves
#
#set title 'nu=0.04, D=2, U=1.0, time=20000'
#plot './data_out/data_out/velocity.020000'u 1:2:($3*1.0):($4*1.0) w vec lc "#0000FF"
##,\
##     './data/circle.dat'w filledcurves
##pause -1

unset multiplot