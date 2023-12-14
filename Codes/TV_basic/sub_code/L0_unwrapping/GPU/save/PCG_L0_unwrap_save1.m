function out_phase = PCG_L0_unwrap(In_phase,In_mask, max_iter, epsi_con ,fdctplan, epsi_0 , last_sol)

vector = @(matrix) (reshape(matrix, 1, size(matrix,1)*size(matrix,2) ,size(matrix,3)));
In_phase=In_phase(:,:,:); %remove higher dim if there is
[mx,my,mz] = size(In_phase) ;
% increasing dimentions for  DCT
phase_pad = In_phase;
mask_pad = In_mask;
[mx,my,~,~] = size(phase_pad) ;


% computing the weighted wrapped phase Laplacian
u = (zeros(size(phase_pad),'single','gpuArray')) ;
v = (zeros(size(phase_pad),'single','gpuArray')) ;

dx_uw = padarray(((diff(last_sol,1,1))),[1 0 0 0], 0 , 'pre') ;
dy_uw = padarray(((diff(last_sol,1,2))),[0 1 0 0], 0 , 'pre') ;
dx_w = padarray(angle(exp(1i*diff(phase_pad,1,1))),[1 0 0 0], 0 , 'pre') ;
dy_w = padarray(angle(exp(1i*diff(phase_pad,1,2))),[0 1 0 0], 0 , 'pre') ;

u=epsi_0./(abs(dx_uw-dx_w).^2+epsi_0);
v=epsi_0./(abs(dy_uw-dy_w).^2+epsi_0);

phase_diff_x = padarray(angle(exp(1i*diff(phase_pad,1,1))),[1 0 0 0], 0 , 'pre') ;
phase_diff_y = padarray(angle(exp(1i*diff(phase_pad,1,2))),[0 1 0 0], 0 , 'pre') ;
u_pdiff_x = u.*phase_diff_x ;
v_pdiff_y = v.*phase_diff_y ;
u_pdiff_x_diff = padarray(diff(u_pdiff_x,1,1),[1 0 0 0], 0 , 'post') ;
v_pdiff_y_diff = padarray(diff(v_pdiff_y,1,2),[0 1 0 0], 0 , 'post') ;
lap_w_wr = u_pdiff_x_diff + v_pdiff_y_diff ;


ind2d_x = gpuArray(single(1:mx)') ;
ind2d_x = repmat(ind2d_x, [1 my]) ;

ind2d_y = gpuArray(single(1:my)) ;
ind2d_y = repmat(ind2d_y, [mx 1]) ;


%%% main PCG algorithm
ph_pcguw = zeros(size(phase_pad),'single','gpuArray') ;


% defining weighted wrapped phase laplacian for one slice (r0)
lap2d_w_wr = squeeze(lap_w_wr(:,:,:)) ;

z_level=gpuArray(1:size(lap2d_w_wr,3));
Num=0*z_level+1;

% initializing the unwrapped phase solution to last solution
ph2d_pcguw = last_sol;%zeros(mx,my,mz,'single','gpuArray') ;
lap2d_w_wr = lap2d_w_wr-compute_Qp(last_sol,u,v,z_level);

mask_pad2d = squeeze(mask_pad(:,:,:)) ;

% needed for epsi calculation
lap2d_w_wr_NaN = lap2d_w_wr ;
lap2d_w_wr_NaN(mask_pad2d==0) = NaN ;
rmse0 = sqrt(nanmean(vector(lap2d_w_wr_NaN.^2))) ;

% initialization of the iteration steps
j = 1 ;
epsi = 1 + 10*epsi_con*ones(1,1,size(lap2d_w_wr,3),'single','gpuArray') ;

% pcg iterations


out_phase = zeros(size(phase_pad),'single','gpuArray') ;
%display( [ 'itt0' num2str(max_iter) ',' num2str(j) ',' num2str(max(epsi(:))) ',' num2str(epsi_con) ]);
while (j<max_iter && length(z_level)>0 && max(epsi(:)) > epsi_con)
    
    %ph_lapuw = R_R_ifdct2(R_R_fdct2(lap2d_w_wr,fdctplan)./(2*cos(ind2d_x*pi/mx) + 2*cos(ind2d_y*pi/my) - 4),fdctplan) ;
    %{
    ph_lapuw = gpuArray(dct2(gather(lap2d_w_wr)))./(2*cos((ind2d_x-1)*pi/mx) + 2*cos((ind2d_y-1)*pi/my) - 4) ;
    ph_lapuw(1,1,:)=0;
    ph_lapuw = gpuArray(idct2(gather(ph_lapuw))) ;
    %}
    ph_lapuw = R_R_fdct2(lap2d_w_wr,fdctplan)./(2*cos((ind2d_x-1)*pi/mx) + 2*cos((ind2d_y-1)*pi/my) - 4) ;
    ph_lapuw(1,1,:)=0;
    ph_lapuw = R_R_ifdct2(ph_lapuw ,fdctplan) ;
    
    % "b" taken from the c++ code
    b = sum(vector(lap2d_w_wr.*ph_lapuw)) ;
    
    % calculate p
    if j == 1
        p = ph_lapuw ;
    else
        beta =  b./b_old ;
        p = ph_lapuw + beta.*p_old ;
    end
    % remove constant bias from p
    
    %{
    p_NaN = p ;
    p_NaN(mask_pad2d==0)=NaN ;
    p_mean = nanmean(vector(p_NaN)) ;
    p = p - p_mean + pi ;
    %}
    
    % calculate Qp
    %{
    p_diff_x = padarray(diff(p,1,1),[1 0 0 0], 0 , 'pre') ;
    p_diff_y = padarray(diff(p,1,2),[0 1 0 0], 0 , 'pre') ;
    u_pdiff_x = u(:,:,z_level).*p_diff_x ;
    v_pdiff_y = v(:,:,z_level).*p_diff_y ;
    u_pdiff_x_diff = padarray(diff(u_pdiff_x,1,1),[1 0 0 0], 0 , 'post') ;
    v_pdiff_y_diff = padarray(diff(v_pdiff_y,1,2),[0 1 0 0], 0 , 'post') ;
    Qp = u_pdiff_x_diff + v_pdiff_y_diff ;
    %}
    Qp=compute_Qp(p,u,v,z_level);
    
    
    % "alpha" as understood from c++ code
    alpha = b./sum(vector(p.*Qp)) ;
    
    ph2d_pcguw = ph2d_pcguw + alpha.*p ;
    
    % removing global bias from unwrapped solution
    %{
    ph2d_pcguw_NaN = ph2d_pcguw ;
    ph2d_pcguw_NaN(mask_pad2d==0) = NaN ;
    ph2d_pcguw_mean = nanmean(vector(ph2d_pcguw_NaN)) ;
    ph2d_pcguw =  ph2d_pcguw - ph2d_pcguw_mean + pi ;
    %}
    
    % calculating residual of the laplacian of the weighted wrapped phase
    lap2d_w_wr = lap2d_w_wr - alpha.*Qp ;
    
    p_old = p ;
    b_old = b ;
    
    lap2d_w_wr_NaN = lap2d_w_wr ;
    lap2d_w_wr_NaN(mask_pad2d==0) = NaN ;
    epsi =  sqrt(nanmean(vector(lap2d_w_wr_NaN.^2)))./rmse0(:,:,z_level) ;
    j = j + 1 ;
    %select finished result
    
    Num(z_level)=Num(z_level)+1;
    
    z_complete=z_level(epsi(:) <= epsi_con);
    z_level=z_level(epsi(:) > epsi_con);
    
    %length(z_level)
    %length(epsi(:) > epsi_con)
    
    out_phase(:,:,z_complete)=ph2d_pcguw(:,:,epsi(:) <= epsi_con);
    ph2d_pcguw=ph2d_pcguw(:,:,epsi(:) > epsi_con);
    lap2d_w_wr=lap2d_w_wr(:,:,epsi(:) > epsi_con);
    b_old=b_old(:,:,epsi(:) > epsi_con);
    p_old=p_old(:,:,epsi(:) > epsi_con);
end
if(length(z_level(:))~=0)
    out_phase(:,:,z_level)=ph2d_pcguw;
end

out_phase = out_phase - pi ;

end

function Qp=compute_Qp(p,u,v,z_level)
    p_diff_x = padarray(diff(p,1,1),[1 0 0 0], 0 , 'pre') ;
    p_diff_y = padarray(diff(p,1,2),[0 1 0 0], 0 , 'pre') ;
    u_pdiff_x = u(:,:,z_level).*p_diff_x ;
    v_pdiff_y = v(:,:,z_level).*p_diff_y ;
    u_pdiff_x_diff = padarray(diff(u_pdiff_x,1,1),[1 0 0 0], 0 , 'post') ;
    v_pdiff_y_diff = padarray(diff(v_pdiff_y,1,2),[0 1 0 0], 0 , 'post') ;
    Qp = u_pdiff_x_diff + v_pdiff_y_diff ;
end