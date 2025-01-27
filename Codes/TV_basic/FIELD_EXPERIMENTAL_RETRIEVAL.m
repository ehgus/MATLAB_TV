classdef FIELD_EXPERIMENTAL_RETRIEVAL < handle
    properties (SetAccess = private, Hidden = true)
        parameters=struct(...
                'size',[100 100 100], ...
                'wavelength', 0.532, ...
                'NA',1.2, ...
                'RI_bg',1.336, ...
                'resolution',[0.1 0.1 0.1], ...
                'vector_simulation', true, ...
                'use_abbe_sine', true, ...
                'resolution_image', [0.1 0.1], ...
                'use_abbe_correction', true, ...
                'cutout_portion', 1/3, ...
                'other_corner',false, ...
                'conjugate_field',false, ...
                'verbose',false, ...
                'normalidx',1 ...
            );
        utility;
    end
    methods
        function h=FIELD_EXPERIMENTAL_RETRIEVAL(init_params)
            if nargin==1
                warning('off','all');
                h.parameters=update_struct(h.parameters,init_params);
                warning('on','all');
            end
        end

        function [input_field,output_field,updated_optical_parameters, k0s]=get_fields(h,input_field,output_field)
            % Image state checker (error)
            assert(isequal(size(input_field),size(output_field)), 'Background and sample field must be of same size')
            assert(size(input_field,1) == size(input_field,2), 'The image must be a square')
            assert(strcmp(class(input_field), class(output_field)), 'The element types of the arrays should be the same')
            assert(h.parameters.resolution_image(1) == h.parameters.resolution_image(2), 'Please enter an isotropic resolution for the image')
            assert(h.parameters.resolution(1) == h.parameters.resolution(2), 'Please enter an isotropic resolution for the output image')
            
            switch ndims(input_field) 
                case 2
                    input_field = reshape(input_field, size(input_field,1), size(input_field,2), 1);
                    output_field = reshape(output_field, size(output_field,1), size(output_field,2), 1);
                case 3
                otherwise
                    error('image must be an single image or stack of images'); 
            end
            
            % Image state checker (warning)
            if isinteger(input_field)
                % pass if element type is float
                maximum_value = intmax(class(input_field));
                is_overexposed = max(max(input_field,[],'all'),max(output_field,[],'all')) > maximum_value;
                if is_overexposed
                    warning('The images are overexposed')
                end
            end
            
            % step 1: Convert image into Fourier space
            input_field=fft2(input_field);
            output_field=fft2(output_field);
            [xsize, ysize, zsize]=size(input_field);
    
            % step 2: Center the field in the Fourier space
            assert(0 < h.parameters.cutout_portion && h.parameters.cutout_portion < 1/2, "cutout portion should be the in (0,1/2)")
            assert(1 <= h.parameters.normalidx && h.parameters.normalidx <= zsize, "Normal index should be the z index of the normal image")
            
            search_band_1=round(xsize*(1/2-h.parameters.cutout_portion)):round(xsize/2);
            if h.parameters.other_corner
                search_band_1=round(xsize/2):round(xsize*(1/2+h.parameters.cutout_portion));
            end
            search_band_2=round(ysize*(1/2-h.parameters.cutout_portion)):round(ysize/2);
            normal_bg=zeros(xsize, round(ysize/2));
            normal_bg(search_band_1, search_band_2)=input_field(search_band_1, search_band_2, h.parameters.normalidx);
                
            [~,linear_index] = max(abs(normal_bg),[],'all');
            [center_pos_1,center_pos_2]=ind2sub(size(normal_bg),linear_index);
            peak2origin=[1-center_pos_1,1-center_pos_2,0];
            input_field=circshift(input_field,peak2origin);
            output_field=circshift(output_field,peak2origin);
                
                
            % setp 3: Resize the fourier images to match to the desired resolution
            old_xsize = xsize;
            old_ysize = ysize;
            resolution_ratio=h.parameters.resolution(1:2)./h.parameters.resolution_image(1:2);
            xsize = 2*round(old_xsize/resolution_ratio(1)/2);
            ysize = 2*round(old_ysize/resolution_ratio(2)/2);
                
            if xsize ~= old_xsize || ysize ~= old_ysize
                old_field = {input_field, output_field};
                new_field = {zeros(xsize,ysize,zsize), zeros(xsize,ysize,zsize)};
                half_xsize = floor(min([old_xsize xsize])/2);
                half_ysize = floor(min([old_ysize ysize])/2);
                
                for i = 1:2
                    new_field{i}(1:half_xsize,1:half_ysize,:) = old_field{i}(1:half_xsize,1:half_ysize,:);
                    new_field{i}(1:half_xsize,end-half_ysize+1:end,:) = old_field{i}(1:half_xsize,end-half_ysize+1:end,:);
                    new_field{i}(end-half_xsize+1:end,1:half_ysize,:) = old_field{i}(end-half_xsize+1:end,1:half_ysize,:);
                    new_field{i}(end-half_xsize+1:end,end-half_ysize+1:end,:) = old_field{i}(end-half_xsize+1:end,end-half_ysize+1:end,:);
                end

                input_field = new_field{1};
                output_field = new_field{2};
                h.parameters.resolution(1)=h.parameters.resolution_image(1).*old_xsize/xsize;
                h.parameters.resolution(2)=h.parameters.resolution_image(2).*old_ysize/ysize;
            end

            % step 4: crop image in accordance with the NA
            h.parameters.size(1)=xsize;
            h.parameters.size(2)=ysize;
            
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
                
            shifted_NA_circle = ifftshift(h.utility.NA_circle);
            input_field=input_field.*shifted_NA_circle;
            output_field=output_field.*shifted_NA_circle;
                
            % Optional step: Find peaks
            % The data will be utilized in next processing
            if nargout>=4
                k0s = zeros(2,size(input_field,3));
                % Subpixel k0s correction
                limit = 1;
                [param1,param2,param3,~]=CPU_placement_finder_prepare(xsize,ysize,1,limit);
                if h.parameters.use_GPU
                    param1 = gpuArray(param1);
                    param2 = gpuArray(param2);
                    param3 = gpuArray(param3);
                end
                for jj = 1:size(input_field,3)
                    shifted_input_field=fftshift(input_field(:,:,jj));
                    [k0s(1,jj), k0s(2,jj),~] = peak_subpixel_positioner(abs(shifted_input_field),param1,param2,param3);
                end
            end
            
            % step 5: Convert processed data into real space
            input_field=ifft2(input_field);
            output_field=ifft2(output_field);
    
            input_field = input_field(3:(end-2),3:(end-2),:);
            output_field = output_field(3:(end-2),3:(end-2),:);
            h.parameters.size(1)=size(input_field,1);
            h.parameters.size(2)=size(input_field,2);

            if h.parameters.conjugate_field
                input_field=conj(input_field);
                output_field=conj(output_field);
            end
            
            % step 6: Subpixel phase correction
            retPhase=angle(output_field./input_field);
            if h.parameters.use_GPU
                retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
            else
                for jj = 1:size(retPhase,3)
                    retPhase(:,:,jj)=(unwrap2((double(retPhase(:,:,jj)))));
                end
            end
            retAmplitude=abs(output_field./input_field);
                
            for jj = 1:size(retPhase,3)
                retPhase(:,:,jj)=PhiShiftMS(retPhase(:,:,jj),1,1);
                if h.parameters.verbose
                    subplot(121);imagesc(retAmplitude(:,:,jj));axis image; axis off; colorbar
                    subplot(122);imagesc(retPhase(:,:,jj));axis image; axis off; colorbar; drawnow
                end
            end
            output_field = retAmplitude .* input_field .* exp(1i.* retPhase);
                
                
            input_field=reshape(input_field,size(input_field,1),size(input_field,2),1,[]);
            output_field=reshape(output_field,size(output_field,1),size(output_field,2),1,[]);
                
            % Ending: Update paramters for next processing
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
             
            warning('abbe cos coefficient');
            if nargout>=3
                updated_optical_parameters=struct( ...
                    'size',h.parameters.size, ...
                    'wavelength',h.parameters.wavelength, ...
                    'NA',h.parameters.NA, ...
                    'RI_bg',h.parameters.RI_bg, ...
                    'resolution',h.parameters.resolution, ...
                    'vector_simulation',h.parameters.vector_simulation, ...
                    'use_abbe_sine',h.parameters.use_abbe_sine, ...
                    'use_GPU',h.parameters.use_GPU ...
                );
            end
        end

        function ROI = find_ROI(h, input_field, output_field)
            input_field0 = input_field(:,:,1);
            output_field0 = output_field(:,:,1);
            resolution0 = h.parameters.resolution;
            h.parameters.resolution(1)=h.parameters.resolution_image(1);
            h.parameters.resolution(1)=h.parameters.resolution_image(1);
            h.parameters.resolution(2)=h.parameters.resolution_image(2);
            [input_field0, output_field0] = get_fields(h, input_field0, output_field0);
            retPhase =  squeeze(angle(output_field0./input_field0));
            if h.parameters.use_GPU
                retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
            else
                retPhase=(unwrap2((double(retPhase))));
            end
            while true
                close all
                figure; imagesc(retPhase, [0 max(retPhase(:))]); axis image; colormap gray; colorbar
                title('Choose square ROI')
                r = drawrectangle;
                ROI = r.Position;
                ROI = [round(ROI(2)) round(ROI(2))+round(ROI(3)) round(ROI(1)) round(ROI(1))+round(ROI(3)) ];
                close all
                figure, imagesc(max(squeeze(retPhase(ROI(1):ROI(2), ROI(3):ROI(4),:,:)),[],3), [0 max(retPhase(:))]), axis image, colormap gray
                satisfied = input('Satisfied? 1/0: ');
                if satisfied
                    close;
                    break;
                end
            end
            h.parameters.resolution = resolution0;
        end
        
        function [input_field_f, output_field_f,aberration_params] = get_aberration_pattern(h,input_field, output_field, correct_params)
        % Stitch pupil
            fE = fftshift(fft2(ifftshift(squeeze(output_field./input_field))));
            fMask = h.utility.fourier_space.coorxy < max(h.utility.fourier_space.coorxy(:)) * correct_params.overlap_factor;
            fdp =  fE.*conj(fE(:,:,1)); 
            fdp = exp(1i*angle(fdp)).*fMask;
            pPhase = fE * 0;
            for ii = 1:size(pPhase,3)
                disp(['Stitcing pupil : ', num2str(ii),' / ', num2str(size(pPhase,3))])
                pPhase(:,:,ii) = circshift(fdp(:,:,ii),round(correct_params.k0s(:,ii)));
            end
            pPhase = pPhase.*h.utility.NA_circle; % NAcrop
            %{
            figure(302), 
            subplot(1,3,1), imagesc(abs(fMask)), colorbar, axis image
            subplot(1,3,2), imagesc(angle(sum(pPhase,3))), colorbar, axis image
            subplot(1,3,3), imagesc(abs(sum(pPhase,3))), colorbar, axis image
            %}
            
        % Correct relative phase
%             pPhase2 = pPhase;
            [~,seq] = sort(sqrt((correct_params.k0s(1,:)).^2+(correct_params.k0s(2,:)).^2));

%             pStep = 5; % 3 takes longer time
%             phiList = linspace(0,2*pi,pStep).';

%             ft = fittype( 'a*cos(x-b)+c', 'independent', 'x', 'dependent', 'y' );
%             opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
            % opts.Display = 'Off';

            pPhase2Temp = pPhase;
            sumTemp = 0;
            %{
            for ii = 2:size(pPhase,3)
                disp(['Phase matching : ', num2str(ii),' / ', num2str(size(pPhase,3))])

                temp = zeros(pStep,1);    
                sumTemp = sumTemp + pPhase2Temp(:,:,seq(ii-1));

            %     figure(1), imagesc(abs(sum(sumTemp,3))), axis image, colorbar, pause()

                for pIter = 1:pStep

                    pPhaseTemp = pPhase2(:,:,seq(ii))*exp(1i*phiList(pIter));
                    temp(pIter) = sum(abs( sumTemp + pPhaseTemp ).^2, [1,2] );
            %         temp(pIter) = sum(abs( exp(1i*angle(sumTemp)) + exp(1i*angle(pPhaseTemp)) ).^2, [1,2] );
                end

                opts.StartPoint = [(max(temp)-min(temp))/2 0 mean(temp)];
                [fitresult, ~] = fit( phiList, temp, ft, opts );

                pPhase2Temp(:,:,seq(ii)) = pPhase2Temp(:,:,seq(ii)).*exp(1i*(fitresult.b));
            end

            pPhase2 = sum(pPhase2Temp,3);
            clear pPhase2Temp
            %}
%             %{
            for ii = 2:size(pPhase,3)
                disp(['Phase matching : ', num2str(ii),' / ', num2str(size(pPhase,3))])
              sumTemp = sumTemp + pPhase2Temp(:,:,seq(ii-1));
               fitresult.b = angle(sum(sumTemp.*conj(pPhase2Temp(:,:,seq(ii))).*and(abs(sumTemp)>0, abs(pPhase2Temp(:,:,seq(ii)))>0),[1,2]));
                pPhase2Temp(:,:,seq(ii)) = pPhase2Temp(:,:,seq(ii)).*exp(1i*(fitresult.b));
            end
            pPhase2 = sum(pPhase2Temp,3);
            clear pPhase2Temp
            %}
            
            %{
            figure(401), imagesc(abs(pPhase2)), axis square, colorbar
            figure(402), imagesc(angle(pPhase2)), axis square, colorbar
            %}
            
        % Numerical refocusing

            % distList = -20:2:20; % rough pre-compensation of defocus aberration
%             distList = -5:1:5;
            distList = correct_params.distList;
            cc = zeros(length(distList),1);
            for ii = 1:length(distList)
                cc(ii) = abs(sum(sum(exp(h.utility.refocusing_kernel*distList(ii)).*exp(-1i*angle(pPhase2)).*h.utility.NA_circle)));
            end
            % figure(1), plot(distList,cc)
            [~,cmax] = max(cc(:));
%             approx = distList(cmax)*h.utility.refocusing_kernel;
            approx = distList(cmax)*h.utility.refocusing_kernel;
            distList(cmax)
            approx = approx - approx(floor(end/2+1),floor(end/2+1)); 
%             distList(cmax)
%             error('stop')

            % dist = diff(f_dx);
            % if max(dist) > 0
            %     dist = dist(dist<0);
            % else
            %     dist = dist(dist>0);
            % end

            dist = mode(diff(round(correct_params.k0s(2,:)))); %%%%%% peak distance (most frequent)
            % dist = 4;
            dist = (abs(mean(dist)))*2; % ideally 2 but practically 1.1 ~ 2.0

       
            % adjust value based on how well low-pass filtered in Fig. 501
            psf = fftshift(ifft2(ifftshift(pPhase2.*exp(1i*approx))));
            
            lpf = psf(:,:,1)*0;
            lpf(floor(size(lpf,1)/2+1)-floor(size(lpf,1)/dist):floor(size(lpf,1)/2+1)+ceil(size(lpf,1)/dist), ...
                floor(size(lpf,2)/2+1)-floor(size(lpf,2)/dist):floor(size(lpf,2)/2+1)+ceil(size(lpf,2)/dist)) = 1;

% 
%             [~, pIdx] = max(abs(psf(:)));
%             [pIdxY,pIdxX] = ind2sub(size(psf),pIdx);
%             pIdxX = pIdxX - floor(size(lpf,2)/2+1);
%             pIdxY = pIdxY - floor(size(lpf,1)/2+1);
%             lpf = circshift(lpf,[pIdxY,pIdxX]);

            
            figure(500), imagesc(log(abs(psf))), axis square
            psf = psf.*lpf; %% phase ramp unknown
            figure(501), imagesc(log(abs(psf))), axis square
            %}
            
            % psf = psf.*~mk_ellipse(sx2/dist,sy2/dist,sx2,sy2);
            % psf = psf.*~mk_ellipse(naX*2/6,naY*2/6,sx2,sy2); % choose size based on fMask
            % psf = psf.*~mk_ellipse(naX*2/5,naY*2/5,sx2,sy2); % choose size based on fMask

            pPhase3Temp = fftshift(fft2(ifftshift(psf)));
            pPhase3Temp = pPhase3Temp.*exp(1i*approx);
            pPhase3 = pPhase3Temp;
            
            aberration_pattern = pPhase3;
            aberration_params.pattern = aberration_pattern.*exp(-1i*angle(aberration_pattern(floor(end/2+1),floor(end/2+1),1))).*h.utility.NA_circle;
            aberration_params.utility = h.utility;
            input_field_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(input_field))) .* exp(-1i.*angle(aberration_params.pattern)))));
            output_field_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(output_field))) .* exp(-1i.*angle(aberration_params.pattern)))));

        end
        function [input_field_f,field_trans_f] = correct_aberration(h, input_field, field_trans, aberration_params)
            pattern = aberration_params.pattern;
            % Resolution matching
            target_fov = round(size(input_field,1) * aberration_params.utility.fourier_space.res{1} / h.utility.fourier_space.res{1}/2)*2;
            if target_fov > size(input_field,1)
                pattern0 = zeros(target_fov,target_fov,size(input_field,3),size(input_field,4),'single');
                pattern = fftshift(fft2(ifftshift(pattern)));
                cropsize = floor(target_fov - size(pattern,1))/2;
                pattern0(cropsize+1:end-cropsize,cropsize+1:end-cropsize,:,:) = pattern;
                pattern = fftshift(ifft2(ifftshift(pattern0)));
            elseif target_fov < size(input_field,1)
                pattern = fftshift(fft2(ifftshift(pattern)));
                cropsize = floor(size(pattern,1) - target_fov)/2;
                pattern = pattern(cropsize+1:end-cropsize,cropsize+1:end-cropsize,:,:);
                pattern = fftshift(ifft2(ifftshift(pattern)));
            end
            
            % FOV matching
            if size(pattern,1) > size(input_field,1)
                cropsize = floor(size(pattern,1) - size(input_field,1))/2;
                pattern = pattern(cropsize+1:end-cropsize,cropsize+1:end-cropsize,:,:);
                pattern = fftshift(ifft2(ifftshift(pattern)));
            elseif size(pattern,1) < size(input_field,1)
                pattern0 = zeros(target_fov,target_fov,size(input_field,3),size(input_field,4),'single');
                cropsize = floor(target_fov - size(pattern,1))/2;
                pattern0(cropsize+1:end-cropsize,cropsize+1:end-cropsize,:,:) = pattern;
                pattern = pattern0;
            end
            coor1 = aberration_params.utility.fourier_space.coor{1} * size(pattern,1) / target_fov;
            coor2 = aberration_params.utility.fourier_space.coor{2} * size(pattern,2) / target_fov;
            coorxy=sqrt(coor1.^2+coor2.^2);
            NA_circle=coorxy<h.utility.kmax;
            pattern = pattern .* NA_circle;
%             imagesc([angle(pattern) angle(aberration_params.pattern)]),axis image, colorbar, pause
            
            input_field_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(input_field))) .* exp(-1i.*angle(pattern)))));
            field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(field_trans))) .* exp(-1i.*angle(pattern)))));
%             input_field_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(input_field))) .* exp(1i.*angle(pattern)))));
%             field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(field_trans))) .* exp(1i.*angle(pattern)))));
           
        end
        function [field_trans_f] = refocus(h, field_trans, z)
            field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(field_trans))) .* exp(z.*h.utility.refocusing_kernel))));
        end
        
        function [input_field_f,output_field_f]=remove_dirty_fields(h,input_field,output_field,prctile, manual_on)
            if nargin == 3
                prctile = 95;
                manual_on = false;
            elseif nargin == 4
                manual_on = false;
            end
            [bg,sp]=vector2scalarfield(input_field,output_field);
            
            retAmplitude=abs(sp./bg);
            retPhase=angle(sp./bg);
            retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
            broken_list1=zeros(1,size(retPhase,3));broken_list2=broken_list1; 
            p0 = retPhase(:,:,1);
           for kk=1:size(retPhase,3)
              f1=(retAmplitude(:,:,kk)-retAmplitude(:,:,1));
              [XX, YY] = gradient(f1);
              f1 = std(XX(:).^2+YY(:).^2);
              f2=(retPhase(:,:,kk) > max(p0(:)).*1.5) + (retPhase(:,:,kk) < min(p0(:)).*1.5);
              broken_list1(kk)=(gather(f1));
              broken_list2(kk)=sum(f2(:))/ prod(size(p0)) > prctile/100;
           end
           F_list=find(isoutlier(broken_list1)+broken_list2==0);
           F_list=unique([1 F_list]);
           length(F_list)

            %%%% Manual selections
            if manual_on
                figure(11), broken_list1 = [];
                i0 = 1;
                for kk=F_list % for good-quality field images
                    a=retAmplitude(:,:,kk);p=retPhase(:,:,kk);
                    ax4=subplot(121);imagesc(p),colorbar,axis image,axis off
                    ax5=subplot(122);imagesc(a),colorbar,axis image,axis off
                    sgtitle([num2str(i0) '/' num2str(length(F_list))])
                    colormap(ax4,parula); colormap(ax5,'gray'); drawnow
                    satisfied = input('Satisfied? 1/0:  '); 
                    if ~satisfied
                        broken_list1 = [broken_list1 i0];
                    end
                    i0 = i0 + 1;
                end
                F_list(broken_list1) = [];
            end
            %%%%
            
            input_field_f = input_field(:,:,F_list);
            output_field_f = output_field(:,:,F_list);
            input_field_f = reshape(input_field_f, [size(input_field_f,1) size(input_field_f,2) 1 size(input_field_f,3)]);
            output_field_f = reshape(output_field_f, [size(output_field_f,1) size(output_field_f,2) 1 size(output_field_f,3)]);
        end
    end
end