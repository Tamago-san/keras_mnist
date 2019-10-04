!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!モデルの特性をそのままコピー
!u(t)=r(t),s(t)=r(t+delta_t)で学習
!その後自立駆動。
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_poseidon(in_node00,out_node00,rc_node00,samp_num00,samp_step00,traning_step00,rc_step00,&
                    u_tr,s_tr,u_rc,s_rc_data,w_out0)
  use cylinder
    implicit none
    integer(4), intent(inout) :: in_node00,out_node00,rc_node00,traning_step00,rc_step00
    integer(4), intent(inout) :: samp_num00,samp_step00
    real(8),    intent(inout) :: w_out0(rc_node00,out_node00)
    real(8),    intent(inout) ::u_tr(traning_step00,in_node00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_tr(samp_num00,out_node00)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::u_rc(rc_step00,in_node00) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_rc_data(rc_step00,out_node00)  !出力次元数、列サイズはトレーニング時間
    real(kind=8) :: R_tr(traning_step00,rc_node00)
    real(8)     w_out(rc_node,out_node)
    real(8)     w_in(in_node,rc_node)
    real(8)     r_befor(rc_node)
    real(8)     a(rc_node,rc_node)
    real(8)     u_tmp(in_node)
    real(8)     r(traning_step,rc_node)
    real(8)     r_rc(rc_step,rc_node)
    real(8)     r_tmp(rc_node)
    real(8)     tmp_1(rc_node,rc_node)
    real(8)     tmp_2(rc_node,out_node)
    real(8)     tmp_3(rc_node,rc_node)
    real(8)     e(rc_node,rc_node)
    real(8)     inverse(rc_node,rc_node)
    real(8)     beta,p,av_degree
    real(8)     r_ave,r_max
    integer(4)  i,j ,k,f
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    real(8)     work(64*rc_node)
    integer(4)  lwork
    integer(4)  ipiv(1:rc_node)
    integer(4)  info
    lwork = 64*rc_node
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    r_max=0.d0
    r_ave=0.d0
    av_degree=0.5
    do i=1,rc_node
        do k=1,rc_node
	        call random_number(p)
!	        write(*,*) p
            if (p<av_degree) then
            	a(k,i)=2.d0*p-1.d0
            else
                a(k,i)=0.d0
            endif
        enddo
    enddo
!    write(*,*) a(1:rc_node,1:rc_node)
	call random_number(w_in)
	w_in(:,:)=(2.d0*w_in(:,:)-1.d0)
    do i=1,in_node
    do j=1,traning_step
        u_traning(j,i) = u_traningT(i,j)
    enddo
    enddo
    do i=1,out_node
    do j=1,traning_step
        s_traning(j,i) = s_traningT(i,j)
    enddo
    enddo
    do i=1,in_node
    do j=1,rc_step
        u_rc(j,i) = u_rcT(i,j)
    enddo
    enddo
    w_out=0.d0
    beta=1.d-3
!    open(54,file='output_traning1.dat', status='replace')
!    do i=1,traning_step
!        write(54,*) i,u_traning(i,1:in_node),s_traning(i,1:out_node)
!    enddo
    close(54)
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    write(*,*) "in_node     ",in_node
    write(*,*) "out_node    ",out_node
    write(*,*) "rc_node     ",rc_node
    write(*,*) "traning_step",traning_step
    write(*,*) "rc_step     ",rc_step
    write(*,*) "gusai       ",gusai
    write(*,*) "alpha       ",alpha
    write(*,*) "g           ",g
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
        tmp_1 = 0.d0
        tmp_2 = 0.d0
        tmp_3 = 0.d0
        r_tmp=0.d0
        e=0.d0
        do i=1,rc_node
            e(i,i)=1.d0
        enddo
!rはtraning_time行rc_node列の正方行列
        do i=1,traning_step
            u_tmp(1:in_node) = u_traning(i,1:in_node)
            call rc_function2( u_tmp,r_tmp)
            call create_r_matrix(r,r_tmp,1,i)
        enddo
        r_befor(1:rc_node) = r_tmp(1:rc_node)
!tmp_1はrc_node行rc_node列の正方行列
        do i=1,rc_node
            do j=1,rc_node
                do k=1,traning_step
                    tmp_1(i,j) = tmp_1(i,j) + r(k,i)*r(k,j)
                enddo
            enddo
        enddo
        tmp_1(1:rc_node,1:rc_node)=tmp_1(1:rc_node,1:rc_node)+beta*e(1:rc_node,1:rc_node)

!tmp_2はrc_node行out_node列の行列
        do f=1,out_node
        do i=1,rc_node
            do k=1,traning_step
                tmp_2(i,f) = tmp_2(i,f) + r(k,i)*s_traning(k,f)
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
        write(*,*) '========================================='
        write(*,*) '_______',f,'        _______'
        write(*,*) '========================================='
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
    do i=1,20
        write(*,*) w_out(i,out_node)
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
    write(*,*) "==============================="
    write(*,*) "    START UP RESERVOIR         "
    write(*,*) "-------------------------------"
    write(*,*) "in_node     ",in_node
    write(*,*) "out_node    ",out_node
    write(*,*) "rc_node     ",rc_node
    write(*,*) "rc_step     ",rc_step
    write(*,*) "gusai       ",gusai
    write(*,*) "alpha       ",alpha
    write(*,*) "g           ",g
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
	do i=1,rc_step
	    u_tmp(1:in_node) = u_rc(i,1:in_node)
        call rc_function2( u_tmp ,r_tmp)
		call create_r_matrix(r_rc,r_tmp,1,i)
		do f = 1,out_node
		    s_rc(i,f) = 0.d0
		    do k = 1,rc_node
		        s_rc(i,f) = s_rc(i,f) + r_rc(i,k) * w_out(k,f)
		    enddo
		enddo
    enddo
    do i=1,out_node
    do j=1,rc_step
        s_rcT(i,j) = s_rc(j,i)
    enddo
    enddo
    do i=1,out_node
    do j=1,rc_node
        w_outT(i,j) = w_out(j,i)
    enddo
    enddo
    
    contains
    subroutine rc_function2( u,r )
        real(8) u(in_node)
        real(8) r(rc_node)
        real(8) wu,ar
        integer j1,j2
        do j1=1,rc_node
            ar =0.d0
            wu=0.d0
            do j2= 1,rc_node
                ar = ar + r(j2)*a(j2,j1)
            enddo
            do j2=1,in_node
                wu = wu + u(j2)*w_in(j2,j1)
            enddo
            r(j1) =  (1-alpha)*r(j1) + alpha*tanh(g*(wu+ar+gusai))
            if(i==traning_step/2)  then
                r_ave = (r_ave + abs(g*(wu+ar+gusai)) )/2.d0
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
end subroutine rc_traning_own_fortran