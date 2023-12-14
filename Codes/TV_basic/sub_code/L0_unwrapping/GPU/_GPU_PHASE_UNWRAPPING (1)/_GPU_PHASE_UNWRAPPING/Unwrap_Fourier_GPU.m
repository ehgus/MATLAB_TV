function UNWRAPPED=Unwrap_Fourier_GPU_V3(Field,plan)
Phase=Field./abs(Field);
%UNWRAPPED= ...
%    R_R_fdct2(imag(C_C_ifdct2(plan{1}.*C_C_fdct2(Phase,plan{2}),plan{2})./Phase),plan{2})./plan{1};
UNWRAPPED= ...
    R_R_fdct2(imag(C_C_ifdct2(plan{1}.*C_C_fdct2(Phase,plan{2}),plan{2})./Phase),plan{2})./plan{1};
% {
if size(Field,3)>1
    UNWRAPPED(1,1,:)=0;
else
    UNWRAPPED(1,1)=0;
end
UNWRAPPED=R_R_ifdct2(UNWRAPPED,plan{2});
%}
end