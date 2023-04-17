set rootname _alg
set fpsinit 0
set fpsend 45

for {set fps $fpsinit} {$fps<=$fpsend} {incr fps} {
set realname $fps$rootname.gro
mol new $realname}

