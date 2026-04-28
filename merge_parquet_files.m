function merge_parquet_files(tmpFiles, outfile)
    % if tmpFiles contains a single file, just copy and rename it
    if numel(tmpFiles)==1
        dataframe=parquetread(tmpFiles{1});
        parquetwrite(outfile,dataframe);
        return
    end
    pyenv('Version','.venv/bin/python');
    pl = py.importlib.import_module('polars');
    % Convert MATLAB string array → Python list
    pyList = py.list(tmpFiles);

    % Use lazy scan (does NOT load everything into memory)
    lazyFrames = py.list();
    for i = 1:length(pyList)
        fprintf("%s\n",tmpFiles{i})
        lf = pl.scan_parquet(pyList{i});
        lazyFrames.append(lf);
    end

    % Concatenate lazily and collect once
    combined = pl.concat(lazyFrames).collect();

    % Write final parquet
    combined.write_parquet(outfile);

    fprintf('Merged into %s\n', outfile);
end