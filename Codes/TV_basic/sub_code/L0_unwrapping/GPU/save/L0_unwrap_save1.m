function out_phase = L0_unwrap(In_phase, max_outer_iter,max_inner_iter, epsi_0 )

itt=0;
epsi_con=0.0;

plan = fdct2_prep(size(In_phase),true);
last_sol=ones(size(In_phase),'single','gpuArray');
In_mask=ones(size(In_phase),'single','gpuArray');

n_residue=ones(size(In_phase,1),'single','gpuArray');

while (max(n_residue(:))>0 && itt<max_outer_iter)
    %pcg iteration
    last_sol=PCG_L0_unwrap(In_phase,In_mask, max_inner_iter, epsi_con , plan, epsi_0 , last_sol);
    
    %count residues
    residue_map = get_residue_gpu(In_phase-last_sol);
    residue_idx=find(residue_map)-1;%keep it as double for safety of rounding
    residue_z_id=single(floor((residue_idx)/(size(residue_map,1)*size(residue_map,2))));%compute it as double for safety of rounding also carfull index starts at zero
    n_residue = histcounts(residue_z_id,gpuArray(0:(size(residue_map,3)))-0.5);
    itt=itt+1;
    display([ 'Iteration : ' num2str(itt) ' | Residue : ' num2str(gather(sum(n_residue(:)))) ' | To unwrapp : ' num2str(gather(length(n_residue(:))))]);
end

% unwrapp to get congruence
out_phase=last_sol;
%out_phase=out_phase+unwrapp2_gpu(In_phase-out_phase);

end