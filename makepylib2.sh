OBJS4=''
echo $OBJS1
f2py -m radtran -c Fortran/eddington.f90 Fortran/radtran.f Fortran/band.f90 Fortran/emissivity-sp.f
