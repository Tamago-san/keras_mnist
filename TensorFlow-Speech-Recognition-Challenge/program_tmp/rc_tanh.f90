!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!モデルの特性をそのままコピー
!u(t)=r(t),s(t)=r(t+delta_t)で学習
!その後自立駆動。
!gfortran -shared -o rc_tanh.so rc_tanh.f90 -llapack -fPIC
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine rc_tanh(in_node,out_node,rc_node,samp_num,samp_step,traning_step,rc_step,&
                    u_tr,s_tr,u_rc,s_rc_data,w_out)
    implicit none
    integer(4), intent(inout) :: in_node,out_node,rc_node,traning_step,rc_step
    integer(4), intent(inout) :: samp_num,samp_step
    real(8),    intent(inout) :: w_out(rc_node,out_node)
    real(8),    intent(inout) ::u_tr(traning_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_tr(samp_num,out_node)  !出力次元数、列サイズはトレーニング時間
    real(8),    intent(inout) ::u_rc(rc_step,in_node) !今は一次元、列サイズはトレーニング時間
    real(8),    intent(inout) ::s_rc_data(rc_step,out_node)  !出力次元数、列サイズはトレーニング時間
    
    real(8)     w_in(in_node,rc_node)
    real(8)     W_rc(rc_node,rc_node)
    
    real(8)     r_befor(rc_node)
    real(8)     r_tmp(rc_node)
    
    real(8)     u_tmp(1,1:in_node)

    real(8)     s_tmp(1,1:out_node)
    real(8)     RiRj(RC_NODE,RC_NODE)
    real(8)     RiSj(RC_NODE,OUT_NODE)
    real(8)     tmp_1(rc_node,rc_node)
    real(8)     tmp_2(rc_node,out_node)
    
    real(8)     e(rc_node,rc_node)
    real(8)     inverse(rc_node,rc_node)
    real(8)     beta,p,av_degree
    real(8)     r_ave,r_max
    real(8)  alpha,g,gusai,NU
    integer(4)  i,j ,k,f
    real(8) :: PI=3.14159265358979
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    real(8)     work(64*rc_node)
    integer(4)  lwork
    integer(4)  ipiv(1:rc_node)
    integer(4)  info
    lwork = 64*rc_node
!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!

!=======================================
!初期化
!=======================================
    r_max=0.d0
    r_ave=0.d0
    alpha = 0.9d0
    g = 1.d0
    gusai = 0.001d0
    NU = 1.d0
    do i=1,rc_node
    do j=1,rc_node
        w_rc(i,j)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
    enddo
    enddo
!    write(*,*) a(1:rc_node,1:rc_node)
    do i=1,in_node
    do j=1,rc_node
        w_in(i,j)=rand_normal(0.d0, 1.d0/(rc_node**0.5d0))
        w_in(:,:)=NU*w_in(:,:)
    enddo
    enddo

    tmp_1 = 0.d0
    tmp_2 = 0.d0
    u_tmp = 0.d0
    s_tmp = 0.d0
    r_tmp=  0.d0
    w_out=0.d0
    beta=1.d-3
    e=0.d0
    do i=1,rc_node
        e(i,i)=1.d0
    enddo
!    open(54,file='output_traning1.dat', status='replace')
!    do i=1,traning_step
!        write(54,*) i,u_traning(i,1:in_node),s_traning(i,1:out_node)
!    enddo
!    close(54)
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

!rはtraning_time行rc_node列の正方行列
    do istep=1,traning_step00
        isample=istep/samp_step00 +1
        if (mod(istep,iout_display).eq.0) write(*,*) 'TRANING_step = ',istep
        if (mod(istep,iout_display).eq.0) write(*,*) '    Re    = ',iRe_int
        if (mod(istep,iout_display).eq.0) write(*,*) '    U     = ',U_tmp
        if (mod(istep,iout_display).eq.0) write(*,*) "   TYO    = " ,TYO
        if (mod(istep,iout_display).eq.0) write(*,200) '-------------    Vx(NXmax/2,0)=',Vx(NXmax/2,0)
        do i=1,in_node
            u_tmp(1,i) = u_tr(istep,i)
        enddo
        do i=1,out_node
            s_tmp(1,i) = s_tr(isample,i)
        enddo
        call mean_rirj(RiRj,RiSj,istep,isample)
    enddo
            
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
	do i=1,rc_step
	    isample=istep/samp_step00 +1
	    do i=1,in_node
            u_tmp(1,i) = u_tr(istep,i)
        enddo
        do i=1,out_node
            s_tmp(1,i) = s_tr(isample,i)
        enddo
        call create_r_matrix(r_tmp)
        
        s_rc(isample,:) =0.d0
        do j=1,out_node
        do i=1,rc_node
            s_rc(isample,j) = S_rc(isample,j) + r_tmp(i)*W_out(i,j)
        enddo
        enddo
    enddo
    
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
            
            call create_r_matrix(r_tmp)
            do i=1,rc_node
            do j=1,rc_node
    !        	do k=1,TRANING_STEP
    !	            RiRj(i,j) = (RiRj(i,j)*dble(step-1) + r_tmp(i)*r_tmp(j) )/dble(step)
    !           enddo
            	RiRj(i,j) = RiRj(i,j)+r_tmp(i)*r_tmp(j)
                if(abs(RiRj(i,j))<1.d-5) RiRj(i,j)=0.d0
            enddo
            enddo
    !tmp_2はRC_NODE行OUT_NODE列の行列
            do j=1,out_node
    	    do i=1,rc_node
    !	    	do k=1,TRANING_STEP
    !               RiSj(i,j) = (RiSj(i,j)*dble(step-1) + r_tmp(i)*s_data(1,j))/dble(step)
    !	    	enddo
    	    	RiSj(i,j) = RiSj(i,j) + r_tmp(i)*s_tmp(1,j)
                if(abs(RiSj(i,j))<1.d-5) RiSj(i,j)=0.d0
    	    enddo
    	    enddo
    !	    write (filename, '("./data_traning_rirj/rirj_RE."i3.3 )') iRe_int
    !	    if(step==1) open(42,file=filename ,status='replace')
    !	    if(step/=1) open(42,file=filename ,position='append')
    !!        rirj=rirj +  r_tmp(i,100)*r_tmp(i,10)
    !         if(mod(step,100)==0) write(42,*) step,RiRj(1,10),RiSj(10,3)
    !!        enddo
    !        close(42)
        end subroutine mean_rirj
        subroutine create_r_matrix(r_tm)
            real(kind=8) :: r_tm(1:rc_node)
            real(8) wu,ar
            integer j1,j2
            r_tm = 0.d0
            do j1=1,rc_node
                ar =0.d0
                wu=0.d0
                do j2= 1,rc_node
                    ar = ar + r_before(j2)*a(j2,j1)
                enddo
                do j2=1,in_node
                    wu = wu + u_tm(1,j2)*w_in(j2,j1)
                enddo
                r_tm(j1) =  (1-alpha)*r_before(j1) + alpha*tanh(g*(wu+ar+gusai))
            enddo
            r_before = r_tmp
        end subroutine create_r_matrix
        subroutine create_Wout_matrix(RiRj,RiSj)
            integer i,j,k
            real(kind=8) RiRj(rc_node,rc_node)
    	    real(kind=8) RiSj(rc_node,out_node)
    	    real(kind=8) tmp_3(rc_node,rc_node)
    	 	real(kind=8) inverse(rc_node,rc_node)
    	 	real(kind=8) :: beta=1.d-3
    	!_________________________________
    	!!!!!!!!lapack!!!!!!!!!!!!!!!!!!!!
    	    real(kind=8) work(64*rc_node)
    		integer :: lwork = 64*rc_node
    		integer ipiv(1:rc_node)
    		integer info
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
        end subroutine create_Wout_matrix
!--------------------------------------
end subroutine rc_tanh