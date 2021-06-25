subroutine pycm1_init(ibo,ieo,jbo,jeo,kbo,keo,&
  ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo)
  implicit none
  integer, intent(out) :: ibo,ieo,jbo,jeo,kbo,keo,&
    ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo
  integer :: ib,ie,jb,je,kb,ke,&
    ibm,iem,jbm,jem,kbm,kem,numq
  call cm1_init(ib,ie,jb,je,kb,ke,&
    ibm,iem,jbm,jem,kbm,kem,numq)
  ibo=ib
  ieo=ie
  jbo=jb
  jeo=je
  kbo=kb
  keo=ke
  ibmo=ibm
  iemo=iem
  jbmo=jbm
  jemo=jem
  kbmo=kbm
  kemo=kem
  numqo=numq
end subroutine pycm1_init

subroutine pytimestep()
  call timestep()
end subroutine pytimestep

subroutine getq(ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo,qao)
  use cm1vars
  implicit none
  integer :: ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo
  real, intent(out) :: qao(ibmo:iemo,jbmo:jemo,kbmo:kemo,numqo)
  qao=qa
end subroutine getq

subroutine setq(ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo,qao)
  use cm1vars
  implicit none
  integer :: ibmo,iemo,jbmo,jemo,kbmo,kemo,numqo
  real, intent(in) :: qao(ibmo:iemo,jbmo:jemo,kbmo:kemo,numqo)
  qa=qao
end subroutine getq
