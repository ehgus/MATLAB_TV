function plan=Unwrap_Fourier_GPU_plan(sizing)
[p, q] = meshgrid(1:sizing(1),1:sizing(2));
p=gpuArray(single(p));
q=gpuArray(single(q));
p=p-1;
q=q-1;
square=p.^2+q.^2;
plan=squeeze(cell(1,2));
plan{1}=square;
plan{2}=fdct2_prep(sizing,true);
end