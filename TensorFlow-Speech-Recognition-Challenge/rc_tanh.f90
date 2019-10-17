!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!モデルの特性をそのままコピー
!u(t)=r(t),s(t)=r(t+delta_t)で学習
!その後自立駆動。
!gfortran -shared -o rc_tanh.so rc_tanh.f90 -llapack -fPIC
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_tanh(in_node,out_node,rc_node,&
                    samp_num,traning_num,rc_num,samp_step,traning_step,rc_step,&
                    u_tr,s_tr,u_rc,s_rc_data,w_out,acc_array)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rc_node
    integer(4), intent(inout) :: traning_num,rc_num,traning_step,rc_step
    integer(4), intent(inout) :: samp_num,samp_step
    real(8),    intent(inout) :: w_out(rc_node,out_node)
    real(8),    intent(inout) ::u_tr(1:traning_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_tr(1:traning_num,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::u_rc(1:rc_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_rc_data(1:rc_num,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::acc_array(out_node,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8) s_rc(rc_num,out_node)  !出力次元数、列サイズはトレーニング時間
    
    real(8)     w_in(in_node,rc_node)
    real(8)     W_rc(rc_node,rc_node)
    
    real(8)     r_bef(rc_node)
    real(8)     r_now(rc_node)
    real(8)     r_ini(rc_node)
    
    real(8)     u_tmp(1,1:in_node)

    real(8)     s_tmp(1,1:out_node)
    real(8)     RiRj(RC_NODE,RC_NODE)
    real(8)     RiSj(RC_NODE,OUT_NODE)
    real(8)     tmp_1(rc_node,rc_node)
    real(8)     tmp_2(rc_node,out_node)
    
    real(8)     e(rc_node,rc_node)
    real(8)     inverse(rc_node,rc_node)
    real(8)     beta,p,av_degree
    real(8)     r_ave,r_max,acc,err
    real(8)  alpha,g,gusai,NU,RHO
    integer(4)  i,j ,k,f,istep,isample,result_data(1),result_rc(1),inode
    real(8) :: PI=3.14159265358979
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!

!=======================================
!初期化
!=======================================
    acc_array = 0.d0
    err = 0.d0
    r_max=0.d0
    r_ave=0.d0
    alpha = 0.9d0
    g = 1.0d0
    gusai = 0.001d0
    NU = 1.0d0
    RHO = 1.d0
    RiRj=0.d0
    RiSj=0.d0
!    call random_number(r_ini)
    do i=1,rc_node
    do j=1,rc_node
        w_rc(i,j)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        w_rc(i,j)=RHO*w_rc(i,j)
    enddo
    enddo
!    write(*,*) a(1:rc_node,1:rc_node)
    do i=1,in_node
    do j=1,rc_node
        w_in(i,j)=rand_normal(0.d0, 1.d0/(in_node*rc_node)**0.5d0)
        w_in(i,j)=NU*w_in(i,j)
    enddo
    enddo

    tmp_1 = 0.d0
    tmp_2 = 0.d0
    u_tmp = 0.d0
    s_tmp = 0.d0
    r_now=  0.d0
    r_bef=1.d0
    w_out=0.d0
    beta=1.d-6
    e=0.d0
    do i=1,rc_node
        e(i,i)=1.d0
    enddo
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) "==============================="
    write(*,*) "    welcome to  Fortran90 !    "
    write(*,*) "-------------------------------"
    write(*,*) "in_node     ",in_node
    write(*,*) "out_node    ",out_node
    write(*,*) "rc_node     ",rc_node
    write(*,*) "samp_step   ",samp_step
    write(*,*) "samp_num    ",samp_num
    write(*,*) "traning_step",traning_step
    write(*,*) "rc_step     ",rc_step
    write(*,*) "gusai       ",gusai
    write(*,*) "alpha       ",alpha
    write(*,*) "g           ",g
    write(*,*) "-------------------------------"
    write(*,*) "==============================="
    write(*,*) "+++++++++++++++++++++++++++++++"
    write(*,*) ""
    open(54,file='./data_out/output_traning_u.dat', status='replace')
    open(55,file='./data_out/output_traning_s.dat', status='replace')
!rはtraning_time行rc_node列の正方行列
    do istep=1,traning_step
        isample=(istep-1)/samp_step +1
        !write(*,*) isample
        if (mod(istep,5000).eq.0) &
            write(*,*) 'TRANING_step = ',istep,int(dble(istep)*100.d0/dble(traning_step)),"%"
        do i=1,in_node
            u_tmp(1,i) = u_tr(istep,i)
        enddo
        do i=1,out_node
            s_tmp(1,i) = s_tr(isample,i)
            !if(s_tmp(1,i)<1.d-10) s_tmp(1,i) =0.d0
        enddo
        write(54,*) isample,u_tmp(1,1:in_node)
        !if(mod(istep,samp_step)==0) write(55,*) istep,isample,nint(s_tmp(1,1:out_node))
        write(55,*) istep,isample,nint(s_tmp(1,1:out_node))
        call create_r_matrix
        !if(mod(istep,samp_step)==0) call mean_rirj(RiRj,RiSj,istep,isample)
        call mean_rirj(RiRj,RiSj,istep,isample)
        !if(mod(istep,samp_step)==0) r_bef =r_ini
    enddo
    close(54)
    close(55)
    write(*,*) "=========================================="
    write(*,*) "     INVERSE MATRIX CALCULATION"
    write(*,*) "=========================================="
    call create_Wout_matrix(RiRj,RiSj)
!    call output_Wout
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
	s_rc(:,:)=0.d0
    open(54,file='./data_out/output_test_u.dat', status='replace')
    open(55,file='./data_out/output_test_s.dat', status='replace')
	do istep=1,rc_step
	    isample=(istep-1)/samp_step +1
	    do i=1,in_node
            u_tmp(1,i) = u_rc(istep,i)
        enddo
        do i=1,out_node
            s_tmp(1,i) = s_rc_data(isample,i)
        enddo
        call create_r_matrix
        
        !s_rc(isample,:)=0.d0
        do j=1,out_node
        do i=1,rc_node
            s_rc(isample,j) = s_rc(isample,j) + r_now(i)*W_out(i,j)
        enddo
        enddo
!        open(40,file='./data_out/r.dat', status='replace')
!        write(40,*) r_now(10)
!        close(40)

        !write(*,"(13e14.3)") r_now(10),s_rc(isample,:)/istep
        write(54,*) isample,u_tmp(1,1:in_node)
        !if(mod(istep,samp_step)==0) write(55,*) istep,isample,nint(s_tmp(1,1:out_node))
        write(55,*) istep,isample,nint(s_tmp(1,1:out_node))
        !if(mod(istep,samp_step)==0) r_bef =r_ini
    enddo
    close(54)
    close(55)

    call rc_cal_acc
    call out_result
    write(*,*) "=========================================="
    write(*,*) "     RC ACC >>>>>",acc/dble(rc_num) *100.d0,'%'
    write(*,*) "=========================================="
    write(*,*) "=========================================="
    write(*,*) "     RC MSE >>>>>",err/dble(out_node*rc_num),'mse'
    write(*,*) "=========================================="
    
    contains
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
        
        subroutine mean_rirj(RiRj,RiSj,step,sample)
            real(8) RiRj(rc_node,rc_node)
    	    real(8) RiSj(rc_node,out_node)
            integer i,j,k,step,sample
            character filename*128
            
            
            do i=1,rc_node
            do j=1,rc_node
    	        !RiRj(i,j) = (RiRj(i,j)*dble(step-1) + r_now(i)*r_now(j) )/dble(step)
            	RiRj(i,j) = RiRj(i,j)+r_now(i)*r_now(j)
                !if(abs(RiRj(i,j))<1.d-5) RiRj(i,j)=0.d0
            enddo
            enddo
    !tmp_2はRC_NODE行OUT_NODE列の行列
            do j=1,out_node
    	    do i=1,rc_node
!                RiSj(i,j) = (RiSj(i,j)*dble(step-1) + r_now(i)*s_tmp(1,j))/dble(step)
    	    	RiSj(i,j) = RiSj(i,j) + r_now(i)*s_tmp(1,j)
                !if(abs(RiSj(i,j))<1.d-5) RiSj(i,j)=0.d0
    	    enddo
    	    enddo
    !	    write (filename, '("./data_traning_rirj/rirj_RE."i3.3 )') iRe_int
    !	    if(step==1) open(42,file=filename ,status='replace')
    !	    if(step/=1) open(42,file=filename ,position='append')
    !!        rirj=rirj +  r_now(i,100)*r_now(i,10)
    !         if(mod(step,100)==0) write(42,*) step,RiRj(1,10),RiSj(10,3)
    !!        enddo
    !        close(42)
        end subroutine mean_rirj
        subroutine create_r_matrix
!            real(kind=8) :: r_tm(1:rc_node)
            real(8) wu,ar
            integer j1,j2
            r_now = 0.d0
            do j1=1,rc_node
                ar =0.d0
                wu=0.d0
                do j2= 1,rc_node
                    ar = ar + r_bef(j2)*w_rc(j2,j1)
                    !write(*,*) r_bef(j2)
                enddo
                do j2=1,in_node
                    wu = wu + u_tmp(1,j2)*w_in(j2,j1)
                enddo
                r_now(j1) =  (1-alpha)*r_bef(j1) + alpha*tanh(g*(wu+ar+gusai))
            enddo
            r_bef = r_now
        end subroutine create_r_matrix
        subroutine create_Wout_matrix(RiRj,RiSj)
            integer i,j,k
            real(kind=8) RiRj(rc_node,rc_node)
    	    real(kind=8) RiSj(rc_node,out_node)
    	    real(kind=8) tmp_3(rc_node,rc_node)
    	 	real(kind=8) inverse(rc_node,rc_node)
!    	 	real(kind=8) :: beta=1.d-10
    	!_________________________________
    	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    	    real(kind=8) work(64*rc_node)
    		integer :: lwork
    		integer ipiv(1:rc_node)
    		integer info
    		lwork = 64*rc_node
    	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
        !---------------------------------
            tmp_3 = 0.d0
        	RiRj(1:rc_node,1:rc_node)=RiRj(1:rc_node,1:rc_node)+beta*e(1:rc_node,1:rc_node)
    !            write(*,*) RiRj
    !tmp_2はRC_NODE行OUT_NODE列の行列
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !逆行列作成++++++++++++++++++++++++++++++++++++++++++++++++++
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            tmp_3=RiRj
        	call dgetrf(rc_node, rc_node, RiRj, rc_node, ipiv, info)
        	call dgetri(rc_node,RiRj, rc_node, ipiv, work, lwork, info)
        	 !write(*,*) RiRj
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    !逆行列確かめ++++++++++++++++++++++++++++++++++++++++++++++++++
    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            inverse=0.d0
            do i=1,rc_node
            	do j=1,rc_node
            		do k=1,rc_node
            			inverse(i,j)=inverse(i,j) + RiRj(i,k)*tmp_3(k,j)
            		enddo
            		if(abs(inverse(i,j))<1.d-6 .and. i/=j) inverse(i,j)=0.d0
            	enddo
            enddo
            do i=1,rc_node
            	if(mod(i,100) == 0.d0) write(*,*) inverse(i,100)
            enddo
            do i=1,rc_node
            	do j=1,rc_node
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
            w_out =0.d0
            do j=1,OUT_NODE
        	do i=1,RC_NODE
        	    do k=1,RC_NODE
        	        w_out(i,j) = w_out(i,j) + RiRj(i,k)* RiSj(k,j)
        	    enddo
        	enddo
        	enddo
        	write(*,*) w_out
        end subroutine create_Wout_matrix
!--------------------------------------
        subroutine rc_cal_acc
            integer(4)  result_data(1),result_rc(1)
            do isample=1,rc_num
                result_rc  =MAXLOC(s_rc(isample,1:out_node))
                result_data=MAXLOC(s_rc_data(isample,1:out_node))
                write(56,*) result_rc,result_data
                acc_array(result_data(1),result_rc(1)) =acc_array(result_data(1),result_rc(1)) +1
                if(result_rc(1)==result_data(1)) then
                    acc = acc + 1.d0
                endif
                do inode = 1,out_node
                    err = err + (s_rc(isample,inode)/dble(samp_step)- s_rc_data(isample,inode))**2
                enddo
            enddo
        end subroutine rc_cal_acc
        subroutine out_result
            open(40,file='./data_out/output_rc_s.dat', status='replace')
            open(41,file='./data_out/output_rc_s_data.dat', status='replace')
            open(42,file='./data_out/output_rc_acc_array.dat', status='replace')
            do isample=1,rc_num
                write(40,"(13e14.3)") s_rc(isample,1:out_node),dble(MAXLOC(s_rc(isample,1:out_node)))
            enddo
            do isample=1,rc_num
                write(41,*) MAXLOC(s_rc(isample,1:out_node)),MAXLOC(s_rc_data(isample,1:out_node))
            enddo
            close(40)
            close(41)
            close(42)
        end subroutine out_result
end subroutine rc_tanh