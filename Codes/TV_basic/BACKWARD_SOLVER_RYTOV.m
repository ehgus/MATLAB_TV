classdef BACKWARD_SOLVER_RYTOV < BACKWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        utility;
    end
    methods
        function h=BACKWARD_SOLVER_RYTOV(params)
            init_params=struct('use_non_negativity',false,'non_negativity_iteration', 100);
            if nargin==1
                warning('off','all');
                init_params=update_struct(init_params, params);
                warning('on','all');
            end
            h@BACKWARD_SOLVER(init_params);
        end
        function [RI, ORytov]=solve(h,input_field,output_field)
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            
            % check fields and parameters
            assert(ndims(input_field) == 4, 'You need to provide the field with 4 dimenssion : dim1 x dim2 x polarisation x illuminationnumber')
            assert(size(input_field,1) == size(input_field,2), 'Please input a square field')
            assert(isequal(size(input_field),size(output_field)), 'Please input field and bg of same size')
            assert(h.parameters.resolution(1) == h.parameters.resolution(2), 'x/y input resolution must be isotropic')
            assert(h.parameters.size(1) == h.parameters.size(2), 'x/y output size must be isotropic')
            assert(h.parameters.size(1) == size(input_field,1) && h.parameters.size(2) == size(input_field,2), 'declare size in the parameter must be the same as the field size')
            
            [bg,sp]=vector2scalarfield(input_field,output_field);
            normalized_field = sp./bg;
            retPhase=angle(normalized_field);
            retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
            if h.parameters.verbose
                for i = 1:size(retPhase,3)
                    imagesc(retPhase(:,:,i));axis image; drawnow
                end
            end
            retAmplitude=abs(normalized_field);
            thetaSize=size(retPhase,3);

            %preset variables
            kx_res = h.utility.fourier_space.res{2};
            ky_res = h.utility.fourier_space.res{1};
            kz_res = h.utility.fourier_space.res{3};
            xsize = h.parameters.size(2);
            ysize = h.parameters.size(1);
            zsize = h.parameters.size(3);
            halfxsize = floor(xsize/2);
            halfysize = floor(ysize/2);

            %find angle
            f_dx=zeros(thetaSize,1);
            f_dy=zeros(thetaSize,1);
            for i=1:size(bg,3)
                Fbg=fft2(bg(:,:,i));
                [~,linear_index] = max(Fbg,[],'all','ComparisonMethod','abs');
                [mj,mi]=ind2sub(size(Fbg),linear_index);
                f_dy(i)=mj-ysize*floor(mj/halfysize)-1;
                f_dx(i)=mi-xsize*floor(mi/halfxsize)-1;
            end
            f_dz=round(real(sqrt((h.utility.k0_nm)^2-(f_dx*kx_res).^2-(f_dy*ky_res).^2))/kz_res);

            NA_circle = ifftshift(h.utility.NA_circle);
            xind=find(NA_circle);
            kz=ifftshift(reshape(h.utility.k3,ysize,xsize));
            fy=[0:floor((ysize-1)/2) -floor((ysize)/2):-1];
            fx=[0:floor((xsize-1)/2) -floor((xsize)/2):-1];
            fz=round(kz/kz_res);
            fy=fy(rem(xind-1,xsize)+1)';
            fx=fx(floor((xind-1)/xsize)+1)';
            fz=fz(xind);
            kz=kz(xind);

            ORytov=gpuArray(zeros(ysize,xsize,zsize,'single'));
            Count=gpuArray(zeros(ysize,xsize,zsize,'single')); 
            for i = 1:thetaSize
                FRytov=squeeze(log(retAmplitude(:,:,i))+1i*retPhase(:,:,i));
                UsRytov=fft2(FRytov); % unit: (um^2)
                
                UsRytov=circshift(UsRytov,[f_dy(i) f_dx(i)]);
                Fx=fx-f_dx(i);Fy=fy-f_dy(i);Fz=fz-f_dz(i);
                Uprime=kz/1i.*UsRytov(xind);% unit: (um^1) % kz is spatial frequency, so 2pi is multiplied for wave vector
                
                Fx=rem(Fx+xsize,xsize)+1;
                Fy=rem(Fy+ysize,ysize)+1;
                Fz=rem(Fz+zsize,zsize)+1;
                Kzp=sub2ind(size(Count),Fy,Fx,Fz);

                ORytov(Kzp)=ORytov(Kzp)+Uprime;
                Count(Kzp)=Count(Kzp)+(Uprime~=0);
            end
            ORytov(Count>0)=ORytov(Count>0)./Count(Count>0)/kz_res; % should be (um^-2)*(px*py*pz), so (px*py*pz/um^3) should be multiplied.
            Reconimg=gather(fftshift(ifftn(ORytov),3));
            Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
            clear Count
            if h.parameters.use_non_negativity
                Emask = ifftshift(h.utility.fourier_space.coorxy)<(2*h.parameters.NA/h.parameters.wavelength);
                Reconimg = gpuArray(fftshift(ifftn(ifftshift(ORytov))));
                for mm = 1:h.parameters.non_negativity_iteration
                    Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                    ORytov_new=fftshift(fftn(ifftshift(Reconimg)));
                    ORytov_new=Emask.*ORytov_new.*(abs(ORytov)==0)+ORytov;
                    Reconimg=fftshift(ifftn(ifftshift(ORytov_new)));
                    %disp([num2str(mm),' / ',num2str(h.parameters.non_negativity_iteration)])
                end
                Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
                ORytov = gather(ORytov);
                ORytov_new = gather(ORytov_new);
                Reconimg = gather(Reconimg);
            end
            RI=Reconimg;
        end
        function [field_trans_f] = refocus(h, field_trans, z) % z is [um]
            field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshfit(field_trans))) .* exp(z.*h.utility.refocusing_kernel))));
        end
    end
end