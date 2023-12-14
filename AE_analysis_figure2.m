function createFigure2

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

% Put titles in same order
namesTmp = {data.Properties.VariableDescriptions{4:12}};
names = {namesTmp{pieOrder}}';
replaceUnderscores = @(str) strrep(str, '_', ' ');
names = cellfun(replaceUnderscores, names, 'UniformOutput', false);
startUppercase = @(str) [upper(str(1)) lower(str(2:end))];
names = cellfun(startUppercase, names, 'UniformOutput', false);

% Limit to top five symptoms
pieOrderCond = pieOrder(1:5);
namesCond = names(1:5);

% Get indices for conditions
tsdcs = data.condition;
a = find(strvcat(tsdcs) == 'A');
c = find(strvcat(tsdcs) == 'C');
s = find(strvcat(tsdcs) == 'S');

% Organize data, split up according to condition
conditions = [a c s];
reportsTmpC = cell(9,3);
for conds = 1:3    
    for col = 4:12
        for cat= 1:4
            reportsTmpC{col-3,conds}(cat,1) = sum(table2array(data(conditions(:,conds),col)) == cat);
        end
    end
end

% Create re-ordered data and plot pie charts and save
f1 = figure(1); set(f1, 'color', [1 1 1]);
for conds = 1:3
    reportsTmpCond = {reportsTmpC{:,conds}}';
    reportsCond = {reportsTmpCond{pieOrderCond}}';
    for j = 1:5
        f1; subplot(3,5,5*(conds-1)+j);
        colororder([0.5 0.5 0.5; 1 1 0; 1 0.5 0;1 0 0]);
        piechart(reportsCond{j});
        title(namesCond{j});
    end
end

% Save pie charts (in two different sizes)
print('Figure2_Small.pdf', '-dpdf');
set(f1, 'position', [1 1 800 500])
print('Figure2_Large.pdf', '-dpdf');
