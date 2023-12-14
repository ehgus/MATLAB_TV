function [Fdata, RIdata] = load_meta_data()
    JSON_PATH = 'F:\Dohyeon Lee\Optiprep\22_data analysis\Utils\file_location.json';
    text = fileread(JSON_PATH);
    jsonData = jsondecode(text);
    RIdata = jsonData.RI;
    head_dir = jsonData.head_dir;
    Fdata  = rmfield(jsonData,{'RI','head_dir'});

    fieldNameList = fieldnames(Fdata);
    for i = 1:length(fieldNameList)
        fieldName = fieldNameList{i};
        spData = Fdata.(fieldName);
        sub_head_dir = spData.head_dir;
        spData = rmfield(spData,'head_dir');
        concentList = fieldnames(spData);
        for j = 1:length(concentList)
            concent = concentList{j};
            spData.(concent) = fullfile(head_dir,sub_head_dir,spData.(concent));
        end
        Fdata.(fieldName) = spData;
    end
end