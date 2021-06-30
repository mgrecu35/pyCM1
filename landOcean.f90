module f90DataTypes
  integer*2 :: lsflag(1080,2160)
  integer*2 :: sstdata(91,144,12)
end module f90DataTypes

subroutine readdbase()
  use f90DataTypes
  call read_geodat( lsflag, sstdata)
end subroutine readdbase

subroutine getlandsea(rlat, rlon, ilandsea)
  use f90DataTypes
  implicit none
 
  real, intent(in):: rlat, rlon
  integer, intent(out)::  ilandsea
  integer :: igetlandsea
 
  ilandsea=igetlandsea(rlat,rlon,lsflag) 

end subroutine getlandsea
