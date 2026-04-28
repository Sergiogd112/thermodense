function filePaths = listFilesIgnoreDirs(rootFolder, excludeNames)
% rootFolder   : string or char scalar, folder to start recursion
% excludeNames : cell array of names to ignore (e.g., {'node_modules','build'})

if ischar(rootFolder), rootFolder = string(rootFolder); end
if ischar(excludeNames), excludeNames = {excludeNames}; end

% Get everything recursively
allEntries = dir(fullfile(rootFolder, '**', '*'));
isFile = ~[allEntries.isdir];
files = allEntries(isFile);

% Preallocate
filePaths = strings(0);

for k = 1:numel(files)
    f = files(k);
    folderParts = split(string(f.folder), filesep);            % path components
    if any(ismember(folderParts, string(excludeNames)))       % skip if any part matches
        continue
    end
    filePaths(end+1) = fullfile(f.folder, f.name);            % collect full path
end
filePaths = cellstr(filePaths); % return as cellstr (common for file lists)
end