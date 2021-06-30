 subroutine initGeophys(nmemb1,sysdN)
   use geophysEns
   real :: sysdn
   integer :: nmemb1, nmfreq
   nmfreq=6
   print*, 'here1'
   call allocGeophys(6,61,9,nmemb1,nmfreq*nmemb1*2)
   print*, 'here2'
   call setdNwIcJcL(sysdN,nmemb1)
   print*, 'here3'
 end subroutine initGeophys
subroutine setijRad(i,j)
   use geophysEns
   integer i,j
   iRad=i
   jRad=j
 end subroutine setijRad

subroutine readtables(lsflag,sstdata)
  byte, intent(out):: lsflag(1080,2160)
  integer*2, intent(out) :: sstdata(91,144,12)

  nmfreq=6
  nmu=5
  call readtablesLiang2(nmu,nmfreq)
  call cloud_init(nmfreq)
  call read_geodat(lsflag,sstdata) 

end subroutine readtables
