function plan = fdct2_prep(sizing,bool_gpu)
sizing=squeeze([sizing(1),sizing(2)]);
if(~exist('bool_gpu','var'))
    bool_gpu=false;
end
[k2,k1]=meshgrid(single(1:sizing(2)),single(1:sizing(1)));
POS1=zeros(sizing,'single');
POS1(1:2:sizing(1),:,:)=k1(1:length(1:2:sizing(1)),:,:);
POS1(2:2:sizing(1),:,:)=flip(k1(end-length(2:2:sizing(1))+1:end,:,:),1);
%POS1=k1;
POS2=zeros(sizing,'single');
POS2(:,1:2:sizing(2),:)=k2(:,1:length(1:2:sizing(2)),:);
POS2(:,2:2:sizing(2),:)=flip(k2(:,end-length(2:2:sizing(2))+1:end,:),2);
%POS2=k2;
POS=single(round(sub2ind(sizing, POS1,POS2)));
%MULT=single(2*exp(-1i.* pi.*((k1-1)./(2*sizing(1))+(k2-1)./(2*sizing(2)))));
MULT1=single(2*exp(-1i.* pi.*((k1(:,1)-1)./(2*sizing(1)))));
MULT2=single(2*exp(-1i.* pi.*((k2(1,:)-1)./(2*sizing(2)))));
%MULT=MULT1.*MULT2;
plan=squeeze(cell(3,1));
if bool_gpu
    MULT1=gpuArray(MULT1);
    MULT2=gpuArray(MULT2);
    POS=gpuArray(POS);
end
plan{1}=MULT1;
plan{2}=MULT2;
plan{3}=POS;
end