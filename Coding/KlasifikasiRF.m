filename = 'lifestyle1.csv';
data = readtable(filename);

occupation_mapping = containers.Map({'Male', 'Female'}, {0, 1});
genderColumn = data.Gender;
genderNumeric = cellfun(@(x) occupation_mapping(x), genderColumn);
data.Gender = genderNumeric;

occupation_mapping = containers.Map({'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
occupationColumn = data.Occupation;
occupationNumeric = cellfun(@(x) occupation_mapping(x), occupationColumn);
data.Occupation = occupationNumeric;

bmi_mapping = containers.Map({'Overweight', 'Normal', 'Obese', 'Normal Weight'}, {0, 1, 2, 3});
bmiCategoryColumn = data.BMICategory; 
bmiNumeric = cellfun(@(x) bmi_mapping(x), bmiCategoryColumn);
data.BMICategory = bmiNumeric;

SleepDisorder_mapping = containers.Map({'None', 'Sleep Apnea', 'Insomnia'}, {0, 1, 2});
SleepDisorderColumn = data.SleepDisorder; 
SleepDisorderNumeric = cellfun(@(x) SleepDisorder_mapping(x), SleepDisorderColumn);
data.SleepDisorder = SleepDisorderNumeric;

x_data = data(:, {'Gender','Age','Occupation','SleepDuration','QualityofSleep', 'PhysicalActivityLevel','StressLevel','BMICategory','HeartRate', 'DailySteps','SystolicBP','DiastolicBP'});

y_target = data.('SleepDisorder');

columns_to_scale = {'Gender','Age','Occupation','SleepDuration','QualityofSleep', 'PhysicalActivityLevel','StressLevel','BMICategory','HeartRate', 'DailySteps','SystolicBP','DiastolicBP'};
data_to_scale = x_data{:, columns_to_scale};
min_vals = min(data_to_scale);
max_vals = max(data_to_scale);
scaled_data = (data_to_scale - min_vals) ./ (max_vals - min_vals);
scaled_data_table = array2table(scaled_data, 'VariableNames', columns_to_scale);
x_data(:, columns_to_scale) = scaled_data_table;

rng('default'); % For reproducibility
test_size = 0.33; % Test size
indices = randperm(height(x_data));
split_idx = round(test_size * height(x_data));

X_train = x_data(indices(split_idx+1:end), :);
X_test = x_data(indices(1:split_idx), :);

y_train = y_target(indices(split_idx+1:end));
y_test = y_target(indices(1:split_idx));

% numTrees = 100; % Number of trees in the forest
% RandomForest = TreeBagger(numTrees, X_train, y_train, 'Method', 'classification');
% Define all parameters
forestParams = struct(...
    'NumTrees', 100, ...
    'Method', 'classification', ...
    'MinLeafSize', 1, ...
    'FBoot', 1.0, ...
    'OOBPrediction', 'off', ...
    'OOBPredictorImportance', 'off', ...
    'Surrogate', 'off', ...
    'PredictorSelection', 'curvature' ...
);

% Train the random forest with all parameters
RandomForest = TreeBagger(...
    forestParams.NumTrees, ...
    X_train, ...
    y_train, ...
    'Method', forestParams.Method, ...
    'MinLeaf', forestParams.MinLeafSize, ... % Corrected parameter
    'FBoot', forestParams.FBoot, ...
    'OOBPrediction', forestParams.OOBPrediction, ...
    'OOBPredictorImportance', forestParams.OOBPredictorImportance, ...
    'Surrogate', forestParams.Surrogate, ...
    'PredictorSelection', forestParams.PredictorSelection ...
);

y_pred = predict(RandomForest, X_test);
y_pred_numeric = cellfun(@(x) str2double(x), y_pred);

forestAccuracy = sum(y_pred_numeric == y_test) / length(y_test);
disp(['Random Forest Accuracy: ', num2str(forestAccuracy)]);

C = confusionmat(y_test, y_pred_numeric);

% ... (Rest of the classification report calculations and display remain the same)
% Display confusion matrix
disp('Confusion Matrix:');
disp(C);
% Calculate classification report
num_classes = numel(unique(y_target));
class_names = unique(y_target);

% Convert numeric class labels to cell array of strings
% class_names_cell = cell(num_classes, 1);
% for i = 1:num_classes
%     class_names_cell{i} = reverse_class_mapping(class_names(i));
% end
class_names_cell = cell(num_classes, 1);
for i = 1:num_classes
    class_names_cell{i} = num2str(class_names(i));
end

% Initialize variables for classification report
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);
for i = 1:num_classes
    tp = C(i, i);
    fp = sum(C(:, i)) - tp;
    fn = sum(C(i, :)) - tp;
    
    precision(i) = tp / (tp + fp);
    recall(i) = tp / (tp + fn);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end

class_names = {'none', 'sleep apnea', 'insomnia'};

% Calculate accuracy, macro average, and weighted average
accuracy = sum(diag(C)) / sum(C(:));
macro_avg_precision = mean(precision);
macro_avg_recall = mean(recall);
weighted_avg_precision = sum(precision .* (sum(C, 2) / sum(C(:))));
weighted_avg_recall = sum(recall .* (sum(C, 2) / sum(C(:))));

% Display classification report headers
fprintf('Classification Report:\n');
fprintf('                    Precision  Recall\n');

% Display recall and precision for each class
for i = 1:num_classes
    fprintf('%-15s       %.2f      %.2f\n', class_names{i}, precision(i), recall(i));
end
fprintf('\n');

% Display macro and weighted averages
fprintf('Macro Avg             %.2f      %.2f\n', macro_avg_precision, macro_avg_recall);
fprintf('Weighted Avg          %.2f      %.2f\n', weighted_avg_precision, weighted_avg_recall);

% % Calculate accuracy, macro average, and weighted average
% accuracy = sum(diag(C)) / sum(C(:));
% macro_avg_precision = mean(precision);
% macro_avg_recall = mean(recall);
% macro_avg_f1 = mean(f1_score);
% weighted_avg_precision = sum(precision .* (sum(C, 2) / sum(C(:))));
% weighted_avg_recall = sum(recall .* (sum(C, 2) / sum(C(:))));
% weighted_avg_f1 = sum(f1_score .* (sum(C, 2) / sum(C(:))));
% % Display classification report
% disp('Classification Report:');
% disp('                    Precision  Recall  F1-Score   Support');
% for i = 1:num_classes
%     fprintf('%-15s       %.2f      %.2f      %.2f        %d\n', class_names_cell{i}, precision(i), recall(i), f1_score(i), sum(C(i, :)));
% end
% fprintf('\n');
% fprintf('Accuracy                                  %.2f       %d\n', accuracy, sum(C(:)));
% fprintf('Macro Avg             %.2f      %.2f      %.2f       %d\n', macro_avg_precision, macro_avg_recall, macro_avg_f1, sum(C(:)));
% fprintf('Weighted Avg          %.2f      %.2f      %.2f       %d\n', weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, sum(C(:)));