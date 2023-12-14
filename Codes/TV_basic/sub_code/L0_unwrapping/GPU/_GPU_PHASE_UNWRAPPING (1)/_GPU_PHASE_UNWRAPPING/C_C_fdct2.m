function C = C_C_fdct2(IMG,plan)
C=R_R_fdct2(real(IMG),plan)+1i*R_R_fdct2(imag(IMG),plan);
end