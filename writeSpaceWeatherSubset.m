function outMat = writeSpaceWeatherSubset( ...
    sourceMat, tStart, tEnd, outMat, padBeforeDays, padAfterDays)

    if nargin < 5 || isempty(padBeforeDays)
        padBeforeDays = 3;
    end
    if nargin < 6 || isempty(padAfterDays)
        padAfterDays = 1;
    end

    persistent SW swDates loadedSource

    sourceMat = char(sourceMat);

    if isempty(loadedSource) || ~strcmp(loadedSource, sourceMat)
        SW = load(sourceMat);
        loadedSource = sourceMat;
        swDates = datetime(SW.YEAR, SW.MONTH, SW.DAY, ...
            "TimeZone", "UTC");
    end

    if isempty(tStart.TimeZone)
        tStart.TimeZone = "UTC";
    end
    if isempty(tEnd.TimeZone)
        tEnd.TimeZone = "UTC";
    end

    firstDay = dateshift(tStart, "start", "day") - days(padBeforeDays);
    lastDay = dateshift(tEnd, "start", "day") + days(padAfterDays);

    rowMask = swDates >= firstDay & swDates <= lastDay;

    if ~any(rowMask)
        error("No space-weather rows found in requested range.");
    end

    keep = {
        "YEAR", "MONTH", "DAY", ...
        "AP1", "AP2", "AP3", "AP4", "AP5", "AP6", "AP7", "AP8", ...
        "AP_AVG", "F107_OBS", "F107_DATA_TYPE", "F107_OBS_CENTER81"
    };

    Sout = struct();

    for i = 1:numel(keep)
        name = keep{i};
        value = SW.(name);

        subs = repmat({':'}, 1, ndims(value));
        subs{1} = rowMask;

        Sout.(name) = value(subs{:});
    end

    save(outMat, "-struct", "Sout");
end