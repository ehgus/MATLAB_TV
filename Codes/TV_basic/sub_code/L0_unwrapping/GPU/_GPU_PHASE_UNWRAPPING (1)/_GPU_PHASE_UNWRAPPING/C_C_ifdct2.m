function C = C_C_ifdct2(IMG,plan)
C=R_R_ifdct2(real(IMG),plan)+1i*R_R_ifdct2(imag(IMG),plan);
end