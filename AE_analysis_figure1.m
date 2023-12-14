function createFigure1

% Read in data
raw_path = '/data/pt_02582/tsDCS_BIDS/';
data = readtable(fullfile(raw_path, 'questionnaire_adverse_effects.tsv'), 'FileType','text');

% Organize data (note that only columns 4-12 contain severity ratings)
reportsTmp = cell(9,1);
for col = 4:12
    for cat = 1:4 
        reportsTmp{col-3}(cat,1) = sum(table2array(data(:,col)) == cat);
    end
end

% Reorder according to number of occurrences
sortTmp = [cellfun(@(x) x(1), reportsTmp)'; 1:9]';
orderTmp = sortrows(sortTmp, 1);
pieOrder = orderTmp(:,2);
reports = {reportsTmp{pieOrder}}';

% Put titles in same order
namesTmp = {data.Properties.VariableDescriptions{4:12}};
names = {namesTmp{pieOrder}}';
replaceUnderscores = @(str) strrep(str, '_', ' ');
names = cellfun(replaceUnderscores, names, 'UniformOutput', false);
startUppercase = @(str) [upper(str(1)) lower(str(2:end))];
names = cellfun(startUppercase, names, 'UniformOutput', false);

% Plot pie charts 
f1 = figure(1); set(f1, 'color', [1 1 1]);
for j = 1:9
    f1; subplot(3,3,j); 
    colororder([0.5 0.5 0.5; 1 1 0; 1 0.5 0;1 0 0]); 
    piechart(reports{j});
    title(names{j});
end

% Save pie charts (in two different sizes)
print('Figure1_Small.pdf', '-dpdf');
set(f1, 'position', [1 1 800 500])
print('Figure1_Large.pdf', '-dpdf');
