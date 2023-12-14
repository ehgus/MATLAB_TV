function slope=PhiShift(field_image)
    % PhiShift
    assert(ismatrix(field_image),'Two dimensional size of image is only allowed')
    angle_image=angle(field_image);
    
    slope = zeros(2);
    for i=1:2
        if i ==1
            diff = angle_image(2:end,:)-angle_image(1:end-1,:);
        else
            diff = angle_image(:,2:end)-angle_image(:,1:end-1);
        end

        max_diff = 2*pi/size(angle_image,i);
        diff = mod(diff+pi,2*pi)-pi;
        diff = diff(abs(diff)<max_diff);
        diff = sort(diff);
        slope(i)=mean(diff);
    end
end