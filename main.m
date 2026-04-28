clearvars;
close all;

files = listFilesIgnoreDirs("data/decoded/tudelft", "tmp");
output_folder = "data/mmsis/tudelft/";

if ~exist(output_folder, "dir")
    mkdir(output_folder);
end

masterMat = aeroReadSpaceWeatherData("SW-Reduced.csv");

S = load(masterMat);

% Replace F10.7 daily values > 400 with 81-day average and mark as INT
highFluxIdx = S.F107_OBS > 400;
if any(highFluxIdx)
    fprintf("Replacing %d F10.7 daily values > 400 with 81-day average\n", sum(highFluxIdx));
    S.F107_OBS(highFluxIdx) = S.F107_OBS_CENTER81(highFluxIdx);
    S.F107_DATA_TYPE(highFluxIdx) = {'INT'};
end

keep = {
    'YEAR','MONTH','DAY', ...
    'AP1','AP2','AP3','AP4','AP5','AP6','AP7','AP8','AP_AVG', ...
    'F107_OBS','F107_DATA_TYPE','F107_OBS_CENTER81'
};

fields = fieldnames(S);
S2 = rmfield(S, setdiff(fields, keep));

filtered_mat = "SW_filtered.mat";
save(filtered_mat, "-struct", "S2");

timestamp_col = "timestamp";
lat = "Latitude_deg_";
lon = "Longitude_deg_";
alt = "Altitude_m_";
lst = "LocalSolarTime_hours_";

tmp_dir = fullfile(output_folder, "tmp");
if ~exist(tmp_dir, "dir")
    mkdir(tmp_dir);
end

for k = 1:numel(files)
    fileparts_ = split(files{k}, filesep);
    mission = fileparts_{end - 1};
    filename = fileparts_{end};

    if endsWith(filename, ".csv")
        continue
    end

    fprintf("Processing %s\n", filename);

    mission_outdir = fullfile(output_folder, mission);
    if ~exist(mission_outdir, "dir")
        mkdir(mission_outdir);
    end

    outfile = fullfile(mission_outdir, replace(filename, "merged", "mmsis"));
    tmp_prefix = fullfile(tmp_dir, replace(filename, "merged", "mmsis"));

    info = parquetinfo(files{k});

    % Read once to get the time span for this dataframe/file
    dfTime = parquetread(files{k});
    startDate = min(dfTime.(timestamp_col));
    endDate = max(dfTime.(timestamp_col));
    clear dfTime;

    [~, baseName, ~] = fileparts(filename);
    weatherMat = fullfile(tmp_dir, baseName + "_spaceweather.mat");

    numRG = info.NumRowGroups;
    batchSize = 200;
    if (endDate.Year-startDate.Year)/numRG > 0.09
        batchSize=50
    end
    numBatches = ceil(numRG / batchSize);
    
    tmpFiles = strings(numBatches, 1);

    for batchIdx = 1:numBatches
        startRG = (batchIdx - 1) * batchSize + 1;
        endRG = min(batchIdx * batchSize, numRG);
        rgList = startRG:endRG;
        
        fprintf('  Batch %d/%d (RowGroups %d-%d): reading...\n', ...
            batchIdx, numBatches, startRG, endRG);
        
        % Read 10 RGs at once (or fewer for the last batch)
        dataframe = parquetread(files{k}, 'RowGroups', rgList);
        writeSpaceWeatherSubset( ...
            filtered_mat, min(dataframe.(timestamp_col)), ...
        max(dataframe.(timestamp_col)), weatherMat, 10, 10);
        fprintf('  Batch %d: space weather...\n', batchIdx);

        t = dataframe.(timestamp_col);

        if isempty(t.TimeZone)
            t.TimeZone = "UTC";
        end

        % Reduce calls drastically:
        % same day + same 3-hour UTC bin => same space weather output
        t3h = dateshift(t, "start", "day") + hours(floor(hour(t)));
        [uniqueT3h, ~, mapBack] = unique(t3h);

        [f107u, f107du, miu] = fluxSolarAndGeomagnetic(uniqueT3h, weatherMat);

        f107average = f107u(mapBack);
        f107daily = f107du(mapBack);
        magneticIndex = miu(mapBack, :);

         fprintf('  Batch %d: density calculation...\n', batchIdx);

        utcSec = second(t, "secondofday");

        [~, densities] = atmosnrlmsise00( ...
            dataframe.(alt), dataframe.(lat), dataframe.(lon), ...
            year(t), ...
            day(t, "dayofyear"), ...
            utcSec, ...
            dataframe.(lst), ...
            f107average, f107daily, magneticIndex);

        dataframe.matlab_density = densities(:, 6);
        dataframe.matlab_f107_81day_avg = f107average;  % 81-day centered average
        dataframe.matlab_f107_daily = f107daily;        % Previous day daily flux
        % Filename indicates range (e.g., rg001-010, rg011-020)
        if startRG == endRG
            tmpFile = sprintf('%s_rg%03d.parquet', tmp_prefix, startRG);
        else
            tmpFile = sprintf('%s_rg%03d-%03d.parquet', tmp_prefix, startRG, endRG);
        end
        
        parquetwrite(tmpFile, dataframe);
        tmpFiles(batchIdx) = tmpFile;
    end

    merge_parquet_files(tmpFiles, outfile);

    for i = 1:numel(tmpFiles)
        if isfile(tmpFiles(i))
            delete(tmpFiles(i));
        end
    end

    if isfile(weatherMat)
        delete(weatherMat);
    end
end