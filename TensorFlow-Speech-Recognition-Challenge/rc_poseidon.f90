!-----------------------------------------------------------------------------
!FDC
!gfortran -shared -o rc_poseidon.so rc_poseidon.f90 -llapack -fPIC
!-----------------------------------------------------------------------------
module cylinder
  integer, parameter :: in_True=1
  integer, parameter :: NOW = 0
  integer, parameter :: BEFORE  = -1
  integer :: Future1 = 10 !0.1
  integer :: Future2 = 30 !0.3
  integer :: Future3 = 50 !0.5
  real(8),parameter  :: Fu1_dble =0.1
  real(8),parameter  :: Fu2_dble =0.3
  real(8),parameter  :: Fu3_dble =0.5
  integer, parameter :: OUT_NODE = 10
  integer, parameter :: IN_NODE = 1
  integer, parameter :: TRANING_TIME_L =  5000 !5000time
  integer, parameter :: RC_TIME_L = 100        !100time
  integer ::  TRANING_STEP = TRANING_TIME_L
  integer ::  RC_STEP = RC_TIME_L    !インプットの信号を入れるか入れないか。
  real(kind=8) :: TYU =0.d0 !信号の平行移動(1なら中心１)
  real(kind=8) :: TYO =1.2d0 !信号の大きさ（１なら大きさ１）
  real(kind=8) :: TYK =5.d0 !5なら信号の大きさ1に（5をデフォルトとする）（-0.5～+0.5）
  integer(kind=4), parameter :: NXmin=-50,NXmax=500 ! 計算領域の形状
  integer(kind=4), parameter :: NYmin=-50,NYmax=50  ! 計算領域の形状
  real(kind=8), parameter :: Xmax=50.d0             ! 計算領域の大きさ
  real(kind=8), parameter :: D=2.d0                 ! 円柱の直径
  real(kind=8), parameter :: U=2.d0                 ! 流入速度
  real(kind=8), parameter :: NU=0.04d0              ! 動粘性係数
  real(kind=8), parameter :: dt=0.01d0              ! 時間刻み
  integer(kind=4), parameter :: Nstep=200000         ! LYトータルステップ数
  integer(kind=4), parameter :: iout=100           ! 出力間隔
  integer(kind=4), parameter :: iout_display=100  ! コンパイル画面の出力間隔
  integer(kind=4), parameter :: iSep=2              ! 出力データの間引き
  real(kind=8) :: Xmin,dX
  real(kind=8) :: Ymax,Ymin
  integer(kind=4) :: iX1(NYmin:NYmax),iX2(NYmin:NYmax)
  integer(kind=4) :: iY1(NXmin:NXmax),iY2(NXmin:NXmax)
  integer(kind=4) :: iXb1,iXb2
  integer(kind=4) :: istep,idtldtN,iRe_int
  integer(kind=4), parameter :: Ly_skip_step = 20000 !撹乱が定常、周期まで25000ステップ（250タイム）
  integer(kind=4), parameter :: skip_step = 20000 !定常，周期まで(200time)
  integer(kind=4), parameter :: Ly_out = 100 !リアプノフ指数の測定ステップ数500ステップ
  real(kind=8) :: abs_dV0
  real(kind=8) :: Vx_tmp(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy_tmp(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Re_tmp_dble
  real(kind=8) :: U_tmp
  integer(kind=4), parameter :: mean_var_step = 100000
  character(6) :: Re_tmp_name
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !-------------------------------------------------------
	integer,parameter :: tate_y  = 20
	integer,parameter :: yoko_x  = 40
	integer,parameter :: Rvx_min = -10
	integer,parameter :: Rvx_max = +30
	integer,parameter :: Rvy_min = -10
	integer,parameter :: Rvy_max = +10
	integer,parameter :: tmpx = 1
	integer,parameter :: tmpy = 1
	integer,parameter :: RC_NODE = (Rvx_max-Rvx_min+1)*(Rvy_max-Rvy_min+1)-250
	integer,parameter :: heikouidou_x = -10
	integer,parameter :: heikouidou_y = -10
!-------------------------------------------------------
  !real(kind=8) :: r_tmp(1,1:RC_NODE)
  real(kind=8) :: W_out(RC_NODE,OUT_NODE)
  real(kind=8) :: DATA_tmp(mean_var_step,3)
  real(kind=8) :: U_data (1,IN_NODE)
  real(kind=8) :: S_data (1,OUT_NODE)
  real(kind=8) :: S_rc (1,OUT_NODE)
  real(kind=8) :: DATA_mean(IN_NODE+OUT_NODE)
  real(kind=8) :: DATA_var(IN_NODE+OUT_NODE)
  real(kind=8) :: DATA(BEFORE:5000,1:3)
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    integer, parameter :: par_node = 3
    real(8), parameter :: par_a = 10.d0
    real(8), parameter :: par_b = 28.d0
    real(8), parameter :: par_c = 8.d0/3.d0
    real(8)  :: dt_l=0.01 !Lorenz
    real(8)  :: dtldtN=1.d0
    real(8), parameter :: dt_Runge = 1.d-4
    real(8) :: x0 = 1.d0
    real(8) :: y0 = 10.d0
    real(8) :: z0 = 5.d0
  contains
!-----------------------------------------------------------------------------
  subroutine cal_rhs(Vx,Vy,P,Rx,Ry)
!-----------------------------------------------------------------------------
! ■ 運動方程式の右辺の計算
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: C1,C2
  C1=1.d0/(2.d0*dX)
  C2=1.d0/(dX**2)
  Rx(:,:)=0.d0
  Ry(:,:)=0.d0
  do iX=NXmin+1,NXmax-1
  do iY=NYmin+1,NYmax-1
     Rx(iX,iY)=-Vx(iX,iY)*C1*(Vx(iX+1,iY)-Vx(iX-1,iY))                 &
               -Vy(iX,iY)*C1*(Vx(iX,iY+1)-Vx(iX,iY-1))                 &
               +NU       *C2*(Vx(iX+1,iY)-2.d0*Vx(iX,iY)+Vx(iX-1,iY))  &
               +NU       *C2*(Vx(iX,iY+1)-2.d0*Vx(iX,iY)+Vx(iX,iY-1))  &
               -          C1*(P(iX+1,iY)-P(iX-1,iY))
     Ry(iX,iY)=-Vx(iX,iY)*C1*(Vy(iX+1,iY)-Vy(iX-1,iY))                 &
               -Vy(iX,iY)*C1*(Vy(iX,iY+1)-Vy(iX,iY-1))                 &
               +NU       *C2*(Vy(iX+1,iY)-2.d0*Vy(iX,iY)+Vy(iX-1,iY))  &
               +NU       *C2*(Vy(iX,iY+1)-2.d0*Vy(iX,iY)+Vy(iX,iY-1))  &
               -          C1*(P(iX,iY+1)-P(iX,iY-1))
  enddo
  enddo
  end subroutine cal_rhs
!-----------------------------------------------------------------------------
  subroutine ibm(Vx,Vy,Rx,Ry)
!-----------------------------------------------------------------------------
! ■ 埋め込み境界法
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
  do iX=iXb1,iXb2
  do iY=iY1(iX),iY2(iX)
     Rx(iX,iY)=Rx(iX,iY)-Vx(iX,iY)/dt
     Ry(iX,iY)=Ry(iX,iY)-Vy(iX,iY)/dt
  enddo
  enddo
  end subroutine ibm
!-----------------------------------------------------------------------------
  subroutine RC_ibm(Vx,Vy,Rx,Ry,O_O)
!-----------------------------------------------------------------------------
! ■ 埋め込み境界法
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: R
  real(kind=8) :: sin1,cos1
  integer :: step,O_O
  
  select case(O_O)
    case (1)
        do iX=iXb1,iXb2
        do iY=iY1(iX),iY2(iX)
            X=dX*real(iX,8)
            Y=dX*real(iY,8)
            R=sqrt(X**2+Y**2)
            sin1 =Y / R
            cos1 =X / R
            Rx(iX,iY)=Rx(iX,iY)+(-Y *(( U_data(1,IN_NODE)/TYK )*TYO + TYU) - Vx(iX,iY))/dt
            Ry(iX,iY)=Ry(iX,iY)+( X *(( U_data(1,IN_NODE)/TYK )*TYO + TYU) - Vy(iX,iY))/dt
        enddo
        enddo
    case (2)
        do iX=iXb1,iXb2
        do iY=iY1(iX),iY2(iX)
            X=dX*real(iX,8)
            Y=dX*real(iY,8)
            R=sqrt(X**2+Y**2)
            sin1 =Y / R
            cos1 =X / R
            Rx(iX,iY)=Rx(iX,iY)+(-Y *(( U_data(1,IN_NODE)/TYK )*TYO + TYU) - Vx(iX,iY))/dt
            Ry(iX,iY)=Ry(iX,iY)+( X *(( U_data(1,IN_NODE)/TYK )*TYO + TYU) - Vy(iX,iY))/dt
        enddo
        enddo
  end select
  end subroutine RC_ibm
  subroutine march(Vx,Vy,P,O_O)
!-----------------------------------------------------------------------------
! ■ 時間発展
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Rx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Ry(NXmin:NXmax,NYmin:NYmax)
  integer O_O
  call cal_rhs(Vx,Vy,P,Rx,Ry)
  select case(O_O)
    case (0)
        call ibm(Vx,Vy,Rx,Ry)
    case (1)
        call RC_ibm(Vx,Vy,Rx,Ry,1)
    case (2)
        call RC_ibm(Vx,Vy,Rx,Ry,2)
  end select

  Vx(:,:)=Vx(:,:)+Rx(:,:)*dt
  Vy(:,:)=Vy(:,:)+Ry(:,:)*dt
!
  call poisson(Vx,Vy,P)
  end subroutine march
 !-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
  subroutine poisson(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 圧力についての Poisson 方程式
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P2(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: NLx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: NLy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: RHS(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: C1,C2,C13,CRedX
  real(kind=8) :: alpha
  real(kind=8) :: delta
  real(kind=8) :: divU,divN
  integer(kind=4), parameter :: itrmax=10000
  real(kind=8), parameter :: eps=1.d-2
  C1=1.d0/(2.d0*dX)
  C2=dX**2
  C13=1.d0/3.d0
  CRedX=2.d0/(Re*dX)
  alpha=0.5d0
  do iX=NXmin+1,NXmax-1
  do iY=NYmin+1,NYmax-1
     NLx(iX,iY)=Vx(iX,iY)*C1*(Vx(iX+1,iY)-Vx(iX-1,iY)) &
               +Vy(iX,iY)*C1*(Vx(iX,iY+1)-Vx(iX,iY-1))
     NLy(iX,iY)=Vx(iX,iY)*C1*(Vy(iX+1,iY)-Vy(iX-1,iY)) &
               +Vy(iX,iY)*C1*(Vy(iX,iY+1)-Vy(iX,iY-1))
  enddo
  enddo
  do iX=NXmin+1,NXmax-1
  do iY=NYmin+1,NYmax-1
     divN=-C1*(NLx(iX+1,iY)-NLx(iX-1,iY)) &
          -C1*(NLy(iX,iY+1)-NLy(iX,iY-1))
     divU=C1*(Vx(iX+1,iY)-Vx(iX-1,iY)+Vy(iX,iY+1)-Vy(iX,iY-1))
     RHS(iX,iY)=divN+divU/dt
!    RHS(iX,iY)=divN ! non-MAC
  enddo
  enddo
! if (mod(istep,10).eq.0) then
!    iX=NXmax/4
!    iY=0
!    write(*,*) 'divU = ',C1*(Vx(iX+1,iY)-Vx(iX-1,iY)+Vy(iX,iY+1)-Vy(iX,iY-1))
! end if
  do iterate=1,itrmax  ! poisson iteration
     P2(:,:)=0.d0
     do iX=NXmin+1,NXmax-1
     do iY=NYmin+1,NYmax-1
        P2(iX,iY)=0.25d0*(P(iX-1,iY)+P(iX+1,iY)+P(iX,iY-1)+P(iX,iY+1)) &
                 -C2*0.25d0*RHS(iX,iY)
     enddo
     enddo
     delta=0.d0
     do iX=NXmin+1,NXmax-1
     do iY=NYmin+1,NYmax-1
        delta=max(delta,dabs(P2(iX,iY)-P(iX,iY)))
     enddo
     enddo
     P(:,:)=P(:,:)+alpha*(P2(:,:)-P(:,:))
     if (delta.lt.eps) goto 1
  enddo
  write(*,'(a,i8)') '**** poission solver did not converge at istep = ',istep
  stop
1 continue
! write(*,'(a,i6)') 'possison solver converged: iteration = ',iterate
  end subroutine poisson
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
  subroutine output(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 出力
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
  real(kind=8) C1
  character(6) :: cistep
  write(cistep,'(i6.6)') istep
  C1=1.d0/(2.d0*dX)
  open(20,file='./data/velocity.'//cistep)
! open(21,file='./data/pressure.'//cistep)
! open(22,file='./data/vorticity.'//cistep)
  do iX=NXmin,NXmax,iSep
  do iY=NYmin,NYmax,iSep
     write(20,'(4e14.6)') dX*dble(iX),dX*dble(iY),Vx(iX,iY),Vy(iX,iY)
!    write(21,'(3e14.6)') dX*dble(iX),dX*dble(iY),P(iX,iY)
!    write(22,'(3e14.6)') dX*dble(iX),dX*dble(iY), &
!    C1*((Vx(iX,iY+1)-Vx(iX,iY-1))-(Vy(iX+1,iY)-Vy(iX-1,iY)))
  enddo
! write(21,*)
! write(22,*)
  enddo
  end subroutine output
!-----------------------------------------------------------------------------
  subroutine initial_condition(Vx,Vy,P)
!-----------------------------------------------------------------------------
! ■ 初期条件 ! y成分に小さい乱数を入れる
!-----------------------------------------------------------------------------
  real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
  real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
  real(kind=8) X,Y
  U_tmp=Re_tmp_dble*NU/D !++!
  call random_number(Vy)
  Vy(:,:)=2.d0*Vy(:,:)-1.d0 ! [-1:1] の乱数
  Vy(:,:)=Vy(:,:)*0.001d0
  do iX=NXmin,NXmax
  do iY=NYmin,NYmax
     X=dX*real(iX,8)
     Y=dX*real(iY,8)
     R=sqrt(X**2+Y**2)
     if (R.gt.D/2.d0) then
        Vx(iX,iY)=U_tmp
     else
        Vx(iX,iY)=0.d0
     endif
  enddo
  enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Vy(NXmin,:)=0.d0
  Vy(:,NYmax)=0.d0
  Vy(:,NYmin)=0.d0
  call poisson(Vx,Vy,P)
  end subroutine initial_condition
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
  subroutine set_grid
!-----------------------------------------------------------------------------
! ■ 格子の設定
! iY1(iX) <---> iY2(iX) が円柱内 ( iX = [iXb1,iXb2] )
!-----------------------------------------------------------------------------
  real(kind=8) :: X,Y
  real(kind=8) :: twopi
  integer :: iX,iY
  dX=Xmax/dble(NXmax)
  Xmin=dble(NXmin)*dX
  Ymin=dble(NYmin)*dX
  Ymax=dble(NYmax)*dX
  iXb1=NXmax
  iXb2=NXmin
  do iX=NXmin,NXmax
  ichk=0
  do iY=NYmin,NYmax-1
     X=dX*dble(iX)
     Y=dX*dble(iY)
     if ((X**2+Y**2-(D/2.d0)**2)*(X**2+(Y+dX)**2-(D/2.d0)**2).lt.0.d0) then
        if (ichk.eq.0) then
           iY1(iX)=iY+1
           ichk=1
        else
           iY2(iX)=iY
        endif
     endif
  enddo
  if (ichk.eq.1) then
     iXb1=min(iXb1,iX)
     iXb2=max(iXb1,iX)
  endif
  enddo
  open(20,file='./data/grid.dat')
  do iX=iXb1,iXb2
  do iY=iY1(iX),iY2(iX)
     X=dX*dble(iX)
     Y=dX*dble(iY)
     write(20,'(2e14.6)') X,Y
  enddo
  enddo
  close(20)
  open(20,file='./data/circle.dat')
  twopi=8.d0*atan(1.d0)
  do i=0,200
     write(20,'(2e14.6)') (D/2.d0)*cos(real(i,8)*twopi/200), &
                          (D/2.d0)*sin(real(i,8)*twopi/200)
  end do
  close(20)
  end subroutine set_grid
    subroutine  Cal_abs_vector(d_Vx,d_Vy,abs_dV)
        real(kind=8) :: d_Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: d_Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: abs_dV
        real(8) X,Y
        abs_dV=0.d0
        do iX=NXmin,NXmax
        do iY=NYmin,NYmax
            abs_dV = abs_dV + d_Vx(iX,iY)**2 +d_Vy(iX,iY)**2
        enddo
        enddo
        abs_dV = abs_dV**(0.5d0)
    end subroutine Cal_abs_vector
    subroutine Normalize_vector(d_Vx,d_Vy,d_P,abs_dV)
        real(kind=8) :: d_Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: d_Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: d_P(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: abs_dV
          d_Vx(:,:) = d_Vx(:,:)/abs_dV
          d_Vy(:,:) = d_Vy(:,:)/abs_dV
          d_P(:,:)  = d_P(:,:)/abs_dV
    end subroutine Normalize_vector
	subroutine DATA_standard(o_o)
		integer i,j,o_o
		select case(o_o)
		    case(0)
		        DATA_mean(1)=mean(DATA_tmp(1:mean_var_step,1),mean_var_step)
		        DATA_mean(2)=mean(DATA_tmp(1:mean_var_step,2),mean_var_step)
		        DATA_mean(3)=mean(DATA_tmp(1:mean_var_step,3),mean_var_step)
		        DATA_var(1)=variance(DATA_tmp(1:mean_var_step,1),DATA_mean(1),mean_var_step)
		        DATA_var(2)=variance(DATA_tmp(1:mean_var_step,2),DATA_mean(2),mean_var_step)
		        DATA_var(3)=variance(DATA_tmp(1:mean_var_step,3),DATA_mean(3),mean_var_step)
!			    do i=1,TRANING_STEP
!				    U_tr(i,IN_NODE)=(U_tr(i,IN_NODE)-DATA_mean(1))/DATA_var(1)
!			    enddo
!			    do j=1,2
!			    do i=1,TRANING_STEP
!				    S_tr(i,j)=(S_tr(i,j)-DATA_mean(j+1))/DATA_var(j+1)
!			    enddo
!			    enddo
!			    do j=3,5
!			    do i=1,TRANING_STEP
!				    S_tr(i,j)=(S_tr(i,j)-DATA_mean(1))/DATA_var(1)
!			    enddo
!			    enddo
			    !U_tr(:,1)= ( U_tr(:,1)/TYK )*TYO +TYU
                !S_tr(:,1)= ( S_tr(:,1)/TYK )*TYO +TYU
                !S_tr(:,2)= ( S_tr(:,2)/TYK )*TYO +TYU
            case(1)
				U_data(1,1)=(U_data(1,1)-DATA_mean(1))/DATA_var(1)
				S_data(1,1)=(S_data(1,1)-DATA_mean(2))/DATA_var(2)
				S_data(1,2)=(S_data(1,2)-DATA_mean(3))/DATA_var(3)
				S_data(1,3)=(S_data(1,3)-DATA_mean(1))/DATA_var(1)
				S_data(1,4)=(S_data(1,4)-DATA_mean(1))/DATA_var(1)
				S_data(1,5)=(S_data(1,5)-DATA_mean(1))/DATA_var(1)
		    case(2)
				U_data(1,1)=(U_data(1,1)-DATA_mean(1))/DATA_var(1)
				S_data(1,1)=(S_data(1,1)-DATA_mean(2))/DATA_var(2)
				S_data(1,2)=(S_data(1,2)-DATA_mean(3))/DATA_var(3)
				S_data(1,3)=(S_data(1,3)-DATA_mean(1))/DATA_var(1)
				S_data(1,4)=(S_data(1,4)-DATA_mean(1))/DATA_var(1)
				S_data(1,5)=(S_data(1,5)-DATA_mean(1))/DATA_var(1)
				!U_rc(1,1)     = (U_rc     (1,1)/TYK )*TYO + TYU
				!S_rc_data(1,1)= (S_rc_data(1,1)/TYK )*TYO + TYU
				!S_rc_data(1,2)= (S_rc_data(1,2)/TYK )*TYO + TYU
		end select
	end subroutine DATA_standard
function mean(a,time) result(out)
		real(8), intent(in) :: a(:)
		real(8) :: out
		integer i,time
		out=0.d0
		do i=1,time
			out = out + a(i)
		enddo
		out= out/dble(time)
	end function mean
	function variance(a,a_mean,time) result(out)
		real(8), intent(in) :: a(:),a_mean
		real(8) :: out
		integer i,time
		out=0.d0
		do i=1,time
			out = out + (a(i)-a_mean)**2
		enddo
		out= (out/dble(time))**0.5
	end function variance
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine create_r_matrix(Vx,Vy,r_tmp,time)
        real(kind=8) :: r_tmp(1:RC_NODE)
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) X,Y, R,C1
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        integer time,iX,iY
        integer i,j,k,inode,inode2
        inode =1
        inode2 =1
        C1=1.d0/(2.d0*dX)
        
        do iX=Rvx_min,Rvx_max
        do iY=Rvy_min,Rvy_max
            X=dX*real(iX,8)
            Y=dX*real(iY,8)
            R=sqrt(X**2+Y**2)
            if (R > D/2.d0) then
                !r_tmp(inode)=C1*((Vx(iX,iY+1)-Vx(iX,iY-1))-(Vy(iX+1,iY)-Vy(iX-1,iY)))
                r_tmp(inode)=Vx(iX,iY)
                inode = inode+1
            else
              inode2=inode2+1
            endif
        enddo
        enddo
        r_tmp(RC_NODE)=dble(in_True)*(( U_data(1,IN_NODE)/TYK )*TYO + TYU)
!        write(*,*) inode,inode2
!        do i= 1,tate_y
!            do j = 1,yoko_x
!            r_tr(time,(i-1)*yoko_x+j)=Vx(tmpx*j+heikouidou_x,tmpy*i+heikouidou_y)
!            enddo
!        enddo
    end subroutine create_r_matrix
    subroutine create_Wout_matrix(RiRj,RiSj)
        integer i,j,k
        real(kind=8) RiRj(RC_NODE,RC_NODE)
	    real(kind=8) RiSj(RC_NODE,OUT_NODE)
	    real(kind=8) tmp_3(RC_NODE,RC_NODE)
	    real(kind=8) e(RC_NODE,RC_NODE)
	 	real(kind=8) inverse(RC_NODE,RC_NODE)
	 	real(kind=8) :: beta=1.d-3
	!_________________________________
	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
	    real(kind=8) work(64*RC_NODE)
		integer :: lwork = 64*RC_NODE
		integer ipiv(1:RC_NODE)
		integer info
	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    !---------------------------------
        	e=0.d0
!        	tmp_1 =0.d0
!        	tmp_2 =0.d0
        	tmp_3 =0.d0
        	w_out =0.d0
    		do i=1,RC_NODE
    			e(i,i)=1.d0
    		enddo
!        	do i=1,RC_NODE
!        		do j=1,RC_NODE
!        			do k=1,TRANING_STEP
!        				tmp_1(i,j) = tmp_1(i,j) + r_tr(k,i)*r_tr(k,j)
!        			enddo
!        		enddo
!        	enddo
    		RiRj(1:RC_NODE,1:RC_NODE)=RiRj(1:RC_NODE,1:RC_NODE)+beta*e(1:RC_NODE,1:RC_NODE)
!            write(*,*) RiRj
!tmp_2はRC_NODE行OUT_NODE列の行列
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列作成++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        	tmp_3=RiRj
    		call dgetrf(RC_NODE, RC_NODE, RiRj, RC_NODE, ipiv, info)
    		call dgetri(RC_NODE,RiRj, RC_NODE, ipiv, work, lwork, info)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列確かめ++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    	    inverse=0.d0
    	    do i=1,RC_NODE
    	    	do j=1,RC_NODE
    	    		do k=1,RC_NODE
    	    			inverse(i,j)=inverse(i,j) + RiRj(i,k)*tmp_3(k,j)
    	    		enddo
    	    		if(abs(inverse(i,j))<1.d-6 .and. i/=j) inverse(i,j)=0.d0
    	    	enddo
    	    enddo
    	    do i=1,RC_NODE
    	    	if(mod(i,100) == 0.d0) write(*,*) inverse(i,100)
    	    enddo
    	    do i=1,RC_NODE
    	    	do j=1,RC_NODE
    	    		if(abs(inverse(i,j))>1.d-6 .and. i/=j) then
    	    			write(*,*) i,j
    	    			write(*,*) inverse(i,j)
    	    		endif
    	    	enddo
    	    enddo
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!W_out作成（OUT_NODE行RC_NODE列）++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            do j=1,OUT_NODE
	    	do i=1,RC_NODE
	    		do k=1,RC_NODE
	    			w_out(i,j) = w_out(i,j) + RiRj(i,k)* RiSj(k,j)
	    		enddo
	    	enddo
	    	enddo
    end subroutine create_Wout_matrix
!-----------------------------------------------------------------------------
    subroutine CREATE_TRANING_DATA
!-----------------------------------------------------------------------------
! ■ 入力
!-----------------------------------------------------------------------------
        !real(kind=8) :: DATA(3,3)
        integer skip
        integer time
        integer i
        DATA(Future3,1) =x0
        DATA(Future3,2) =y0
        DATA(Future3,3) =z0
        do i=1,10000
!            call Runge_Kutta_method(DATA(BEFORE:Future3,1:3),&
!                    DATA(Future3,1),DATA(Future3,2),DATA(Future3,3),i)
        enddo
        do i=1,mean_var_step
!            call Runge_Kutta_method(DATA(BEFORE:Future3,1:3),&
!                    DATA(Future3,1),DATA(Future3,2),DATA(Future3,3),i)
            DATA_tmp(i,1) = DATA(NOW,1)
            DATA_tmp(i,2) = DATA(NOW,2)
            DATA_tmp(i,3) = DATA(NOW,3)
!            DATA_tmp(i,3) = DATA(Future1,1)
!            DATA_tmp(i,4) = DATA(Future2,1)
!            DATA_tmp(i,5) = DATA(Future3,1)
        enddo
        x0=DATA(Future3,1)
        y0=DATA(Future3,2)
        z0=DATA(Future3,3)
        call DATA_standard(0)

    end subroutine CREATE_TRANING_DATA
    subroutine CAL_RC_ERR(err_tmp)
        real(kind=8) ::  err_tmp(1:OUT_NODE)
        integer i,j,k
!        err_tmp = 0.d0
    	do i=1,OUT_NODE
    			err_tmp(i)=err_tmp(i) + (S_rc(1,i)-S_data(1,i) )**2
    	enddo
    end subroutine CAL_RC_ERR
    subroutine RC_OWN(Vx,Vy, time)
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: r_tmp(1:RC_NODE)
        real(kind=8) X,Y, R
        integer time,iX,iY
        integer i,j,k,inode
        inode =1
        
        call create_r_matrix(Vx,Vy,r_tmp,time)
        
        S_rc(:,:) = 0.d0
        do j=1,OUT_NODE
        do i=1,RC_NODE
            S_rc(1,j) = S_rc(1,j) + r_tmp(i)*W_out(i,j)
        enddo
        enddo
    end subroutine RC_OWN
    subroutine output_Wout
        real(kind=8) X,Y, R
        integer time,iX,iY
        integer i,j,k,inode
        character filename*128
        
        !filename = a
        inode=1
        write (filename, '("./data_Wout/Wout_RE."i3.3 )') iRe_int
        open(42,file=filename,status='replace')
        do iX=Rvx_min,Rvx_max
        do iY=Rvy_min,Rvy_max
            X=dX*real(iX,8)
            Y=dX*real(iY,8)
            R=sqrt(X**2+Y**2)
            if (R > D/2.d0) then
                write(42,"(7e14.6)") X,Y,W_out(inode,1:OUT_NODE)
                inode=inode+1
            else
                write(42,"(7e14.6)") X,Y,W_out(RC_NODE,1:OUT_NODE)
            endif
        enddo
        write(42,*) ""
        enddo
        close(42)
    end subroutine output_Wout
    subroutine mean_rirj(Vx,Vy,RiRj,RiSj,step)
        real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
        real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
        real(8) RiRj(RC_NODE,RC_NODE)
	    real(8) RiSj(RC_NODE,OUT_NODE)
	    real(8) r_tmp(1:rc_node)
        integer i,j,k,step
        character filename*128
        
        call create_r_matrix(Vx,Vy,r_tmp,step)
        do i=1,RC_NODE
        do j=1,RC_NODE
!        	do k=1,TRANING_STEP
        	!	RiRj(i,j) = (RiRj(i,j)*dble(step-1) + r_tmp(i)*r_tmp(j) )/dble(step)
        	RiRj(i,j) = RiRj(i,j)+r_tmp(i)*r_tmp(j)
                if(abs(RiRj(i,j))<1.d-5) RiRj(i,j)=0.d0
!enddo
        enddo
        enddo
!tmp_2はRC_NODE行OUT_NODE列の行列
        do j=1,OUT_NODE
	    do i=1,RC_NODE
!	    	do k=1,TRANING_STEP
	    		!RiSj(i,j) = (RiSj(i,j)*dble(step-1) + r_tmp(i)*s_data(1,j))/dble(step)
	    	RiSj(i,j) = RiSj(i,j) + r_tmp(i)*s_data(1,j)
!	    	enddo
                if(abs(RiSj(i,j))<1.d-5) RiSj(i,j)=0.d0
	    enddo
	    enddo
	    write (filename, '("./data_traning_rirj/rirj_RE."i3.3 )') iRe_int
	    if(step==1) open(42,file=filename ,status='replace')
	    if(step/=1) open(42,file=filename ,position='append')
!        rirj=rirj +  r_tmp(i,100)*r_tmp(i,10)
         if(mod(step,100)==0) write(42,*) step,RiRj(1,10),RiSj(10,3)
!        enddo
        close(42)
    end subroutine mean_rirj
!-----------------------------------------------------------------------------
!■ルンゲクッタ法での近似
!-----------------------------------------------------------------------------
!-----------------------------------------------------------------------------
end module cylinder
!-----------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_poseidon(in_node00,out_node00,rc_node00,traning_step00,rc_step00,&
                    u_tr,s_tr,u_rc,s_rc_data,w_out0)
  use cylinder
    implicit none
    integer(4), intent(inout) :: in_node00,out_node00,rc_node00,traning_step00,rc_step00
    real(8),    intent(inout) :: w_out0(rc_node00,out_node00)
    real(8),    intent(inout) ::u_tr(traning_step00,in_node00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_tr(traning_step00,out_node00)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::u_rc(rc_step00,in_node00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_rc_data(rc_step00,out_node00)  !出力次元数、列サイズはトレーニング時間
    real(kind=8) :: R_tr(traning_step00,rc_node00)
!    real(kind=8) :: W_out(rc_node00,out_node00)
!    real(kind=8) :: U_tr (traning_step00,in_node00)
!    real(kind=8) :: S_tr (traning_step00,out_node00)
 !   real(kind=8) :: U_rc (rc_step00,in_node00)
 !   real(kind=8) :: S_rc (rc_step00,out_node00)
 !   real(kind=8) :: S_rc_data(rc_step00,out_node00)
    real(kind=8) :: Vx(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: Vy(NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: P (NXmin:NXmax,NYmin:NYmax)
    real(kind=8) :: RCerr(1:OUT_NODE)
    real(kind=8) RiRj(RC_NODE,RC_NODE)
    real(kind=8) RiSj(RC_NODE,OUT_NODE)
    integer iX,iY,i,j
    integer Re_tmp_int,iERR
    character filename*128

!=================================================================================================================
!-----------------------------------------------------------------------------------------------------------------
!初期化
        !write (filename,'mkdir -p ./data')
        !call system( filename )
!    open(82,file='./data_lyap_end/lyapnov_end.dat',status='replace')
!    close(82)
!    open(46,file='RC_err.dat',status='replace')
!    close(46)
!    OUT_NODE = out_node00
!    IN_NODE = in_node00
!    TRANING_STEP =traning_step00
!    RC_STEP = rc_step
!    call allocate_mat(in_node00,out_node00,rc_node00,traning_step00,rc_step00)
!    do idtldtN=8,2,-2
!        write (filename, '("mkdir -p ./data_Wout/dt", i5.5)') idtldtN
!        call system( filename )
!        write (filename, '("mkdir -p ./data_traning_rirj/dt", i5.5)') idtldtN
!        call system( filename )
!        write (filename, '("mkdir -p ./data_RC_time_series_RE/dt", i5.5)') idtldtN
!        call system( filename )
!    enddo
!    do idtldtN=8,2,-2
!        write (filename, '("./data_err/RCerr_dt."i5.5 )') idtldtN
!        open (46, file=filename, status='replace')
!        close(46)
!    enddo
!----------------------------------------------------------------------------------------------------------------
!=================================================================================================================
    

    !OUT_NODE = out_node00
    !IN_NODE = in_node00
    !TRANING_STEP = traning_step00
    !RC_STEP = rc_step00
    
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    WRITE(*,*) "IN_NODE      ",in_node00
    WRITE(*,*) "OUT_NODE     ",out_node00
    WRITE(*,*) "RC_NODE      ",rc_node
    WRITE(*,*) "TRANING_STEP ",traning_step00
    WRITE(*,*) "RC_STEP      ",rc_step00
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
!=================================================================================================================
    call set_grid
    
!【Reのdoループ 】
    do Re_tmp_int = 10,10,2
!【TYU,TYOのdoループ 】
    do iERR = 3,3
!    TYU=dble(iERR)*0.1d0
    TYO=dble(iERR)*1.d0
!    Re_tmp_int =30
    
        Re_tmp_dble =dble(Re_tmp_int)
        RCerr=0.d0

        call initial_condition(Vx,Vy,P)
!  call output_V_P(Vx,Vy,P,0)
        write(Re_tmp_name,'(i3.3)') Re_tmp_int
        write(*,*) ""
        write(*,*) "============================================= "
        write(*,*) "START >>  Re ==" ,Re_tmp_dble
        write(*,*) "START >>  TYO ==" ,TYO
        write(*,*) "============================================= "



!流れが定常or周期状態になるまでスキップ
        write(*,*) "=========================================="
        write(*,*) "     SKIP STEP"
        write(*,*) "=========================================="
        do istep=1,skip_step
    !=================================================================================================================
    !-----------------------------------------------------------------------------------------------------------------
    !ディスプレイ表示(一つ前のステップの値が出力istep=10ならistep=9の値)
    !-----------------------------------------------------------------------------------------------------------------
            if (mod(istep,5000).eq.0) write(*,*) 'Skip_step = ',istep
            if (mod(istep,5000).eq.0) write(*,*) '    Re    = ',iRe_int
            if (mod(istep,5000).eq.0) write(*,*) '    U     = ',U_tmp
            if (mod(istep,5000).eq.0) write(*,*) "   TYO    = " ,TYO
            if (mod(istep,5000).eq.0) write(*,200) '-------------    Vx(NXmax/2,0)=',Vx(NXmax/2,0)
            call march(Vx,Vy,P,0)
            !write(32,*) dt*real(istep),Vx(NXmax/2,0)
        enddo
  
!        Vx_tmp(:,:) = Vx(:,:)
!        Vy_tmp(:,:) = Vy(:,:)


        write(*,*) "=========================================="
        write(*,*) "     TRANING STEP"
        write(*,*) "=========================================="
            do istep=1,TRANING_STEP
                if (mod(istep,iout_display).eq.0) write(*,*) 'TRANING_step = ',istep
                if (mod(istep,iout_display).eq.0) write(*,*) '    Re    = ',iRe_int
                if (mod(istep,iout_display).eq.0) write(*,*) '    U     = ',U_tmp
                if (mod(istep,iout_display).eq.0) write(*,*) "   TYO    = " ,TYO
                if (mod(istep,iout_display).eq.0) write(*,200) '-------------    Vx(NXmax/2,0)=',Vx(NXmax/2,0)
                U_data(1,1) = u_tr(istep,1)
                do i=1,10
                    S_data(1,i) = s_tr(istep,i)
                enddo
                !call DATA_standard(2)
                call march(Vx,Vy,P,1)
                if(call mean_rirj(Vx,Vy,RiRj,RiSj,istep)
                Vx_tmp(:,:) = Vx(:,:)
                Vy_tmp(:,:) = Vy(:,:)
            enddo
            write(*,*) "=========================================="
            write(*,*) "     INVERSE MATRIX CALCULATION"
            write(*,*) "=========================================="
    !        write(*,*) RiRj
            call create_Wout_matrix(RiRj,RiSj)
            call output_Wout
    !=================================================================================================================%%
            write(*,*) "=========================================="
            write(*,*) "     RC STEP"
            write(*,*) "=========================================="
            
            write (filename, '("./data_RC_time_series_RE/RC_time_series_RE."i3.3 )') iRe_int
            open(45,file=filename ,status='replace')
    !        open(46,file='RC_err.dat',position='append')
            do istep=1,rc_step00
                if (mod(istep,iout_display).eq.0) write(*,*) 'RC_step = ',istep
!                call Runge_Kutta_method(DATA(BEFORE:Future3,1:3),&
!                        DATA(Future3,1),DATA(Future3,2),DATA(Future3,3),istep)
                U_data(1,1) = u_tr(istep,1)
                do i=1,10
                    S_data(1,i) = s_tr(istep,i)
                enddo
!                call DATA_standard(2)
                call march(Vx,Vy,P,2)
                call RC_OWN(Vx,Vy,istep)
                if(istep<=int(30.d0/dt_l) ) then
                    if(mod(istep,ceiling(1.d0/(dt_l*1.d2) ) ) == 0) &
                            write(45,"(11e14.3)") U_data(1,1),S_rc(1,1:OUT_NODE),S_data(1,1:OUT_NODE)
                endif
                Vx_tmp(:,:) = Vx(:,:)
                Vy_tmp(:,:) = Vy(:,:)
                call CAL_RC_ERR(RCerr)
            enddo
            close(45)
    enddo
    enddo
!    s_rc(:,:)=S_rc(:,:)/s_rc(1,1)


 !   do i=1,out_node
 !   do j=1,rc_step
 !       s_rcT(i,j) = s_rc(j,i)
 !   enddo
 !   enddo
 !   do i=1,out_node
 !   do j=1,rc_node
 !       w_outT(i,j) = W_out(j,i)
 !   enddo
 !   enddo

100 format(a,i6,a,f15.10)
200 format(a,f15.10)
end subroutine rc_poseidon
!-----------------------------------------------------------------------------