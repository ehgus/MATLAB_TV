function out_phase = L0_unwrap(In_phase)

max_outer_iter=40;
max_inner_iter=40;
outer_cutoff=30;

epsi_0=0.003;
min_epsi_0=0.001;
improve_thresh=1;%if stagnation

itt=0;
epsi_con=0.0;

epsi_0=epsi_0.*ones(1,1,size(In_phase,3),'single','gpuArray');

plan = fdct2_prep(size(In_phase),true);
last_sol=ones(size(In_phase),'single','gpuArray');
In_mask=ones(size(In_phase),'single','gpuArray');

n_residue=ones(size(In_phase,3),'single','gpuArray');
old_n_residue=n_residue;
z_level=1:size(In_phase,3);

while (max(n_residue(:))>0 && itt<max_outer_iter)
    %pcg iteration
    last_sol=PCG_L0_unwrap(In_phase(:,:,z_level),In_mask(:,:,z_level), max_inner_iter, epsi_con , plan, epsi_0 , last_sol);
    
    %count residues
    residue_map = get_residue_gpu(In_phase(:,:,z_level)-last_sol);
    residue_idx=find(residue_map)-1;%keep it as double for safety of rounding
    residue_z_id=single(floor((residue_idx)/(size(residue_map,1)*size(residue_map,2))));%compute it as double for safety of rounding also carfull index starts at zero
    n_residue = histcounts(residue_z_id,gpuArray(0:(length(z_level(:))))-0.5);
    itt=itt+1;
    
    %if itt==outer_cutoff
        %epsi_0=min_epsi_0*ones(1,1,length(epsi_0(:)),'single','gpuArray');
        %improve=n_residue./old_n_residue;
        %epsi_0(improve>improve_thresh)=epsi_0(improve>improve_thresh)./10;
        %epsi_0=max(epsi_0,min_epsi_0*ones(1,1,length(epsi_0(:)),'single','gpuArray'));
        %improve
        %epsi_0
    %end
    
    
    finished=z_level(n_residue(:)<=0);
    z_level=z_level(n_residue(:)>0);
    
    out_phase(:,:,finished)=last_sol(:,:,n_residue(:)<=0);
    
    old_n_residue=n_residue;
    old_n_residue=old_n_residue(n_residue(:)>0);
    epsi_0=epsi_0(:,:,n_residue(:)>0);
    last_sol=last_sol(:,:,n_residue(:)>0);
    
    display([ 'Iteration : ' num2str(itt) ' | Residue : ' num2str(gather(sum(n_residue(:)))) ' | To unwrapp : ' num2str(gather(length(n_residue(:))))]);
    
end

% unwrapp to get congruence
if length(z_level)~=0
out_phase(:,:,z_level(:))=last_sol(:,:,:);
end
%out_phase=out_phase+unwrapp2_gpu(In_phase-out_phase);

if sum(n_residue(:))~=0
    for ii=1:size(In_phase,3)
        %     size(double(In_phase(:,:,ii)-out_phase(:,:,ii)))
        %     whos
        out_phase(:,:,ii)=out_phase(:,:,ii)+gpuArray(single(unwrap2(gather(double(In_phase(:,:,ii)-out_phase(:,:,ii))))));
    end
end
end