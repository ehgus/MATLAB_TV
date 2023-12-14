% Implementation of a 2D weighted least-squares phase unwrapping
% based on preconditioned conjugate gradient (PCG) method with a
% preconditioner taken from a laplace unwrapping with discrete consine
% transform (DCT) plus additional congruence operation.
%
% Written by: Barbara Dymerska
%
% Based on theory and c++ code from: "Two-Dimensional Phase Unwrapping, Theory, Algorithms,
% and Software" written by Dennis C. Ghiglia and Mark D. Pritt (chapter 5)
%
% Usage: 
% [ph_uw_file, ph_uw_nii] = PCG_unwrap_2D(ph_nii, ph_file, mask_nii, max_iter, epsi_con, N)
%
% input files and parameters:
% ph_nii - phase data in rad in nifti format loaded using load_nii
% ph_file - full path to the corresponding phase file
% mask_nii - binary mask or quality mask with range of values [0,1] in nifti format loaded using load_nii
% max_iter - maximum number of iterations allowed, e.g. 5
% epsi_con - convergence parameter, e.g. 0.00001
% N - number of congruence steps, suggested N = 20
%
% output files
% ph_uw_nii - unwrapped phase in rad
% ph_uw_file - full path where the unwrapped phase is saved


function [ph_uw_file, ph_uw_nii] = PCG_unwrap_2D(ph_nii, ph_file, mask_nii, max_iter, epsi_con, N)

% tic
weighted_laplacian = 'yes' ;


[mx,my,mz,TP] = size(ph_nii.img) ;

if size(mask_nii.img,4)~=TP
    mask_nii.img = repmat(single(mask_nii.img),[1 1 1 TP]) ;
end

pixdim = ph_nii.hdr.dime.pixdim(2:4) ;

% increasing dimentions for  DCT
phase_pad = padarray(single(ph_nii.img), [mx/2 my/2]) ;
mask_pad = padarray(single(mask_nii.img), [mx/2 my/2]) ;

mx_orig = mx ;
my_orig = my ;

[mx,my,~,~] = size(phase_pad) ;


switch weighted_laplacian
    case 'yes'
        % computing the weighted wrapped phase Laplacian
        u = zeros(size(phase_pad),'single') ;
        v = zeros(size(phase_pad),'single') ;
        
        for x = 1:mx-1
            u(x,:,:,:) = min(mask_pad(x+1,:,:,:), mask_pad(x,:,:,:)) ;
        end
        
        for y = 1:my-1
            v(:,y,:,:) = min(mask_pad(:,y+1,:,:), mask_pad(:,y,:,:)) ;
        end
        
        phase_diff_x = padarray(angle(exp(1i*diff(phase_pad,1,1))),[1 0 0 0], 0 , 'pre') ;
        phase_diff_y = padarray(angle(exp(1i*diff(phase_pad,1,2))),[0 1 0 0], 0 , 'pre') ;
        u_pdiff_x = u.*phase_diff_x ;
        v_pdiff_y = v.*phase_diff_y ;
        u_pdiff_x_diff = padarray(diff(u_pdiff_x,1,1),[1 0 0 0], 0 , 'post') ;
        v_pdiff_y_diff = padarray(diff(v_pdiff_y,1,2),[0 1 0 0], 0 , 'post') ;
        lap_w_wr = u_pdiff_x_diff + v_pdiff_y_diff ;
        
    case 'no'
        
        phase_diff_x = padarray(angle(exp(1i*diff(phase_pad,1,1))),[1 0 0 0], 0 , 'pre') ;
        phase_diff_y = padarray(angle(exp(1i*diff(phase_pad,1,2))),[0 1 0 0], 0 , 'pre') ;
        pdiff_x_diff = padarray(diff(phase_diff_x,1,1),[1 0 0 0], 0 , 'post') ;
        pdiff_y_diff = padarray(diff(phase_diff_y,1,2),[0 1 0 0], 0 , 'post') ;
        lap_w_wr = pdiff_x_diff + pdiff_y_diff ;
        
end

ind2d_x = (1:mx)' ;
ind2d_x = repmat(ind2d_x, [1 my]) ;

ind2d_y = 1:my ;
ind2d_y = repmat(ind2d_y, [mx 1]) ;


%%% main PCG algorithm
ph_pcguw = zeros(size(phase_pad),'single') ;
for t = 1:TP
    
    for z = 1:mz
        
        % defining weighted wrapped phase laplacian for one slice (r0)
        lap2d_w_wr = squeeze(lap_w_wr(:,:,z,t)) ;
        
        % initializing the unwrapped phase solution to zero
        ph2d_pcguw = zeros(mx,my,'single') ;
        
        mask_pad2d = squeeze(mask_pad(:,:,z,t)) ;
        
        % needed for epsi calculation
        lap2d_w_wr_NaN = lap2d_w_wr ;
        lap2d_w_wr_NaN(mask_pad2d==0) = NaN ;
        rmse0 = sqrt(nanmean(vector(lap2d_w_wr_NaN.^2))) ;
        
        % initialization of the iteration steps
        j = 1 ;
        epsi = 10*epsi_con ;

        % pcg iterations
        while (j<max_iter && epsi > epsi_con)
            
            % solving Pz_k=r_k equation using unweighted discrete cosine transform algorithm
            ph_lapuw = idct2(dct2(lap2d_w_wr)./(2*cos(ind2d_x*pi/mx) + 2*cos(ind2d_y*pi/my) - 4)) ;
            
            if j==1
                ph_lapuw_step1(:,:,z,t) =  ph_lapuw ;
            end
            
            % "b" taken from the c++ code
            b = sum(vector(lap2d_w_wr.*ph_lapuw)) ;
            
            % calculate p
            if j == 1
                p = ph_lapuw ;
            else
                beta =  b/b_old ;
                p = ph_lapuw + beta.*p_old ;
            end
            % remove constant bias from p
            p_NaN = p ;
            p_NaN(mask_pad2d==0)=NaN ;
            p_mean = nanmean(vector(p_NaN)) ;
            p = p - p_mean + pi ;
            
            % calculate Qp
            p_diff_x = padarray(diff(p,1,1),[1 0 0 0], 0 , 'pre') ;
            p_diff_y = padarray(diff(p,1,2),[0 1 0 0], 0 , 'pre') ;
            
            switch weighted_laplacian
                case 'yes'
                    
                    u_pdiff_x = u(:,:,z,t).*p_diff_x ;
                    v_pdiff_y = v(:,:,z,t).*p_diff_y ;
                    u_pdiff_x_diff = padarray(diff(u_pdiff_x,1,1),[1 0 0 0], 0 , 'post') ;
                    v_pdiff_y_diff = padarray(diff(v_pdiff_y,1,2),[0 1 0 0], 0 , 'post') ;
                    Qp = u_pdiff_x_diff + v_pdiff_y_diff ;
                    
                case 'no'
                    
                    pdiff_x_diff = padarray(diff(p_diff_x,1,1),[1 0 0 0], 0 , 'post') ;
                    pdiff_y_diff = padarray(diff(p_diff_y,1,2),[0 1 0 0], 0 , 'post') ;
                    Qp = pdiff_x_diff + pdiff_y_diff ;
                    
            end
            
            % "alpha" as understood from c++ code
            alpha = b/sum(vector(p.*Qp)) ;
            
            ph2d_pcguw = ph2d_pcguw + alpha*p ;
            
            % removing global bias from unwrapped solution
            ph2d_pcguw_NaN = ph2d_pcguw ; 
            ph2d_pcguw_NaN(mask_pad2d==0) = NaN ;
            ph2d_pcguw_mean = nanmean(vector(ph2d_pcguw_NaN)) ;
            ph2d_pcguw =  ph2d_pcguw - ph2d_pcguw_mean + pi ;
            
            % calculating residual of the laplacian of the weighted wrapped phase
            lap2d_w_wr = lap2d_w_wr - alpha*Qp ;
            
            
            ph_lapuw_old = ph_lapuw ;
            p_old = p ;
            b_old = b ;
            
            lap2d_w_wr_NaN = lap2d_w_wr ;
            lap2d_w_wr_NaN(mask_pad2d==0) = NaN ;
            epsi =  sqrt(nanmean(vector(lap2d_w_wr_NaN.^2)))/rmse0 ;
            j = j + 1 ;
            
        end
        ph_pcguw(:,:,z,t) = ph2d_pcguw ;
        disp(['Slice ' num2str(z) ' finished after iteration nr: ' num2str(j) ' with epsi ' num2str(epsi)])
    end
    
end



ph_lapuw_step1_rd = ph_lapuw_step1((mx_orig/2)+1 : mx-(mx_orig/2), (my_orig/2)+1 : my-(my_orig/2),:,:) ;
[dir,name,~] = fileparts(ph_file) ;
save_nii(make_nii(ph_lapuw_step1_rd, pixdim), fullfile(dir, sprintf('%s_lapuw.nii',name))) ;
clear ph_lapuw_step1

ph_pcguw_rd = ph_pcguw((mx_orig/2)+1 : mx-(mx_orig/2), (my_orig/2)+1 : my-(my_orig/2),:,:) ;
ph_pcguw_rd = ph_pcguw_rd - pi ;
ph_uw_file = fullfile(dir, sprintf('%s_pcguw.nii',name)) ;
save_nii(make_nii(ph_pcguw_rd, pixdim), ph_uw_file) ;

%%
% CONGRUENCE OPERATION
mask_smooth = smoothn(mask_nii.img,2) ;
mask_nii.img(mask_smooth>0.5) = 1 ;
mask_nii.img(mask_smooth<=0.5) = 0 ;
ph_uw_cong = zeros(mx_orig,my_orig,mz,TP,N, 'single') ;
edge_mask = zeros(mx_orig,my_orig,mz,TP,N, 'single') ;
D = zeros(mz,TP,N, 'single') ;
mask_edge_mask = zeros(mx_orig,my_orig,mz,TP, 'single') ;
for k = 1:N
    h = 2*pi*k/N ;
    ph_uw_cong(:,:,:,:,k) = ph_pcguw_rd(:,:,:,:) + h + angle(exp(1i*(ph_nii.img(:,:,:,:)-ph_pcguw_rd(:,:,:,:)-h))) ;    

%%% comment out the following 3 lines if you want to check how results look for each "h"
%     ph_uw_file = fullfile(dir, sprintf('%s_pcguw_cong_step%i.nii',name, k)) ;
%     ph_uw_nii = make_nii(ph_uw_cong(:,:,:,:,k), pixdim) ;
%     save_nii(ph_uw_nii, ph_uw_file)
    

    for t = 1 : TP
        for z = 1 : mz
            edge_mask(:,:,z,t,k) = edge(ph_uw_cong(:,:,z,t,k).*mask_nii.img(:,:,z,t),'Sobel',1.6) ;
            D(z,t,k) = sum(vector(edge_mask(:,:,z,t,k).*mask_nii.img(:,:,z))) ;
        end
    end
    
end

ph_uw_final = zeros(size(ph_nii.img), 'single') ;

for t = 1 : TP
    for z = 1 : mz
        [~, Kmin] =  min(vector(D(z,t,:))) ;
        ph_uw_final(:,:,z,t) = ph_uw_cong(:,:,z,t,Kmin) ;
    end
end


ph_uw_file = fullfile(dir, sprintf('%s_pcguw_cong.nii',name)) ;
ph_uw_nii = make_nii(ph_uw_final, pixdim) ;
save_nii(ph_uw_nii, ph_uw_file)

% disp(['pcguw finished after ' num2str(secs2hms(toc))])