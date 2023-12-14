function unlock_tomocube_data(srcFile,dstDir)
%UNLOCK_TOMOCUBE_DATA conver ~~~,
%   자세한 설명 위치
    ImgCnt = length(h5info(srcFile,'/images').Datasets);
    for i = 0:ImgCnt-1
        h5loc = sprintf('/images/%06d',i);
        ImgBinary = h5read(srcFile,h5loc);
        dstFile = fullfile(dstDir,sprintf('%06d.png',i));
        fileID = fopen(dstFile,'w');
        fwrite(fileID,ImgBinary);
        fclose(fileID);
    end
end

