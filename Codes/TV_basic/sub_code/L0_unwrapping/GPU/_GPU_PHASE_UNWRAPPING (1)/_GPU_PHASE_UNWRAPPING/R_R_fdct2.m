function C = R_R_fdct2(IMG,plan)
%ALGORITHM INSPIRED FROM : JOHN MAKHOUL / A Fast Cosine Transform in One and Two Dimensions
C=real(IMG);
if size(IMG,3)>1
    C=reshape(C,size(IMG,1)*size(IMG,2),size(IMG,3));
    C(plan{3},:)=C(:,:);
    C=reshape(C,size(IMG,1),size(IMG,2),size(IMG,3));
else
    C(plan{3})=C(:);
end
C=fft2(C);
%C = real(sqrt(1/(16*size(IMG,1)*size(IMG,2)))*(MULT1.*(MULT2.*trans+conj(MULT2).*circshift(flip(trans,2),[0,1]))));
C =( plan{1}.*plan{2}.*(C));
D=circshift(flip(imag(C),2),[0,1]);
if size(IMG,3)>1
D(:,1,:)=0;%D(:,2);
else
D(:,1)=0;%D(:,2);   
end
C= real(C)-D;
C=real(sqrt(1/(16*size(IMG,1)*size(IMG,2))))*C;
if size(IMG,3)>1
C(:,1,:)=sqrt(2)*C(:,1,:);
C(1,:,:)=sqrt(1/2)*C(1,:,:);
else
C(:,1)=sqrt(2)*C(:,1);
C(1,:)=sqrt(1/2)*C(1,:);
end
end