!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!モデルの特性をそのままコピー
!u(t)=r(t),s(t)=r(t+delta_t)で学習
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_traning_own_fortran(in_node,out_node,rc_node,tr_step,rc_step,gusai,alpha,g,nu,&
                    u_tr,s_tr,u_rc,w_in,a,w_out)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rc_node,tr_step,rc_step
    real(8),    intent(inout) :: gusai,alpha,g,nu
    real(8),    intent(inout) ::u_tr(tr_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_tr(tr_step,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::w_out(rc_node,out_node)
    real(8),    intent(inout) ::w_in(in_node,rc_node)
    real(8),    intent(inout) ::a(rc_node,rc_node)
    real(8),    intent(inout) ::U_rc(rc_step,in_node)
    real(8)     r_befor(rc_node)
!    real(8)     a(rc_node,rc_node)
    real(8)     s_test(tr_step,out_node)
    real(8)     u_tmp(in_node)
    real(8)     r(tr_step,rc_node)
    real(8)     r_rc(rc_step,rc_node)
    real(8)     r_tmp(rc_node)
    real(8)     tmp_1(rc_node,rc_node)
    real(8)     tmp_2(rc_node,out_node)
    real(8)     tmp_3(rc_node,rc_node)
    real(8)     e(rc_node,rc_node)
    real(8)     inverse(rc_node,rc_node)
    real(8)     beta,p,av_degree
    real(8)     r_ave,r_max,abcd
    real(8) :: PI=3.14159265358979
    integer(4)  i,j ,k,f
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    real(8)     work(64*rc_node)
    integer(4)  lwork
    integer(4)  ipiv(1:rc_node)
    integer(4)  info
    lwork = 64*rc_node
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!

!初期化==========================================
    do i=1,rc_node
    do j=1,rc_node
        a(i,j)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
    enddo
    enddo
!    write(*,*) a(1:rc_node,1:rc_node)
    do i=1,in_node
    do j=1,rc_node
        w_in(i,j)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        w_in(:,:)=NU*w_in(:,:)
    enddo
    enddo
	call random_number(w_in)
    w_out=0.d0
    beta=1.d-3
    tmp_1 = 0.d0
    tmp_2 = 0.d0
    tmp_3 = 0.d0
    r_tmp=0.d0
    r_max=0.d0
    r_ave=0.d0
    av_degree=1.d0
    e=0.d0
    do i=1,rc_node
        e(i,i)=1.d0
    enddo
!初期化-終わり=======================================

    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    write(*,*) "in_node     ",in_node
    write(*,*) "out_node    ",out_node
    write(*,*) "rc_node     ",rc_node
    write(*,*) "tr_step     ",tr_step
    write(*,*) "gusai       ",gusai
    write(*,*) "alpha       ",alpha
    write(*,*) "g           ",g
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""

!rはtraning_time行rc_node列の正方行列
    do i=1,tr_step
        u_tmp(1:in_node) = u_tr(i,1:in_node)
        call rc_function2( u_tmp,r_tmp)
        call create_r_matrix(r,r_tmp,1,i)
    enddo
    r_befor(1:rc_node) = r_tmp(1:rc_node)
!tmp_1はrc_node行rc_node列の正方行列
    do i=1,rc_node
    do j=1,rc_node
        do k=1,tr_step
            tmp_1(i,j) = tmp_1(i,j) + r(k,i)*r(k,j)
        enddo
    enddo
    enddo
    tmp_1(1:rc_node,1:rc_node)=tmp_1(1:rc_node,1:rc_node)+beta*e(1:rc_node,1:rc_node)

!tmp_2はrc_node行out_node列の行列
    do f=1,out_node
    do i=1,rc_node
        do k=1,tr_step
            tmp_2(i,f) = tmp_2(i,f) + r(k,i)*s_tr(k,f)
        enddo
    enddo
    enddo
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列作成++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    tmp_3=tmp_1
    call dgetrf(rc_node, rc_node, tmp_1, rc_node, ipiv, info)
    call dgetri(rc_node,tmp_1, rc_node, ipiv, work, lwork, info)
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!逆行列確かめ++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    inverse=0.d0
    do i=1,rc_node
    do j=1,rc_node
        do k=1,rc_node
            inverse(i,j)=inverse(i,j) + tmp_1(i,k)*tmp_3(k,j)
        enddo
        if(abs(inverse(i,j))<1.d-6 .and. i/=j) inverse(i,j)=0.d0
    enddo
    enddo
    do i=1,rc_node
    do j=1,rc_node
        if(abs(inverse(i,j))>1.d-6 .and. i/=j) then
            write(*,*) i,j
            write(*,*) inverse(i,j)
        endif
    enddo
    enddo
    do f=1,out_node
        do i=1,rc_node
            do k=1,rc_node
                w_out(i,f) = w_out(i,f) + tmp_1(i,k)* tmp_2(k,f)
            enddo
        enddo
    enddo

    write(*,*) ""
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "       COUMPLETE TRANING  !    "
    write(*,*) "-------------------------------"
    write(*,*) "   abs(r_ave)", r_ave
    write(*,*) "   abs(r_max)", r_max
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "+++++++++++++++++++++++++++++++"
!    write(*,*) "==============================="
!    write(*,*) "    START UP RESERVOIR         "
!    write(*,*) "-------------------------------"
!    write(*,*) "in_node     ",in_node
!    write(*,*) "out_node    ",out_node
!    write(*,*) "rc_node     ",rc_node
!    write(*,*) "rc_step     ",rc_step
!    write(*,*) "gusai       ",gusai
!    write(*,*) "alpha       ",alpha
!    write(*,*) "g           ",g
!    write(*,*) "-------------------------------"
!    write(*,*) "==============================="
!    write(*,*) "+++++++++++++++++++++++++++++++"
!    write(*,*) ""
	do i=1,rc_step
	    u_tmp(1:in_node) = u_rc(i,1:in_node)
        call rc_function2( u_tmp ,r_tmp)
		call create_r_matrix(r_rc,r_tmp,1,i)
		do f = 1,out_node
		    s_test(i,f) = 0.d0
		    do k = 1,rc_node
		        s_test(i,f) = s_test(i,f) + r_rc(i,k) * w_out(k,f)
		    enddo
		enddo
    enddo
    open(54,file='./data_out/output_traning1.dat', status='replace')
    do i=1,rc_step
        write(54,*) i, s_test(i,1:out_node),u_rc(i,1:in_node)
    enddo
    close(54)
    contains
    subroutine rc_function2( u,r_next)
        real(8) ,intent(in) :: u(in_node)
        real(8) r(rc_node)
        real(8) r_next(rc_node)
        real(8) wu,ar
        integer j1,j2
        r=r_next
        do j1=1,rc_node
            ar =0.d0
            wu=0.d0
            do j2= 1,rc_node
                ar = ar + r(j2)*a(j2,j1)
            enddo
            do j2=1,in_node
                wu = wu + u(j2)*w_in(j2,j1)
            enddo
            r_next(j1) =  (1-alpha)*r(j1) + alpha*tanh(g*(wu+ar+gusai))
            if(i==tr_step/2)  then
                if(j1==1) then
                    r_ave=0.d0
                    r_max=0.d0
                endif
                r_ave = (r_ave*dble(j1-1) + abs(g*(wu+ar+gusai)) )/dble(j1)
                r_max = max(r_max,abs(g*(wu+ar+gusai)) )
            endif
        enddo
    end subroutine rc_function2
    subroutine create_r_matrix(r,r_tmp,o_o,time)
        real(8) r_tmp(rc_node)
        real(8) r(:,:)
        integer o_o,time
        
        select case (o_o)
            case (1)
                r(time,1:rc_node) = r_tmp(1:rc_node)
            case (2)
                r(time,1:rc_node/2) = r_tmp(1:rc_node/2)
                do j=rc_node/2 +1,rc_node
                    r(time,j) =(r_tmp(j) )**2
                enddo
        end select
    end subroutine create_r_matrix
    function rand_normal(mu,sigma)
    	real(8) mu,sigma
    	real(8) rand_normal
    	real(8) z,p1,p2
    	call random_number(p1)
    	call random_number(p2)
    	!write(*,*) p1,p2
        z=sqrt( -2.0*log(p1) ) * sin( 2.0*PI*p2 );
        rand_normal= mu + sigma*z
    end function rand_normal
end subroutine rc_traning_own_fortran