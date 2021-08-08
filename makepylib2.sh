OBJS4=''
echo $OBJS1
f2py -m radtran -c tablesModule.f90 Fortran/readTables_nonsph.f90 Fortran/bisection.f90 Fortran/absorption3D.f90 Fortran/eddington.f90 Fortran/radtran.f Fortran/gcloud.f Fortran/band.f90 Fortran/rosen.f Fortran/emissivity-sp.f
