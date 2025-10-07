data = readtable('Social_Network_Ads(8).csv');

% Standardization
stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;
stand_estimated_salary = (data.EstimatedSalary - mean(data.EstimatedSalary)) / std(data.EstimatedSalary);
data.EstimatedSalary = stand_estimated_salary;

% classification_model = fitcknn(data,'Purchased~Age+EstimatedSalary');
classification_model = fitcnb(data,'Purchased~Age+EstimatedSalary');


cv = cvpartition(classification_model.NumObservations,'HoldOut',0.2);

cross_validated_model = crossval(classification_model,'cvpartition',cv);
Predictions = predict(cross_validated_model.Trained{1}, data(test(cv),1:end-1));
Results = confusionmat(cross_validated_model.Y(test(cv)), Predictions);
%% -------------- Visualizing training set results --------------
% ---------------------------- Code ---------------------------
 
labels = unique(data.Purchased);
classifier_name = 'K-Nearest Neigbor (Training Results)';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
training_data = data(training(cv),:);
Y = ismember(training_data.Purchased,labels{1});
 
 
scatter(training_data.Age(Y),training_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~Y),training_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
 
xlabel('Age');
ylabel('Estimated Salary');
 
title(classifier_name);
legend off, axis tight
 
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
%% -------------- Visualizing testing set results ----------------
% ---------------------------- Code ---------------------------
 
labels = unique(data.Purchased);
classifier_name = 'K-Nearest Neigbor (Testing Results)';
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Estimated_salary_range = min(data.EstimatedSalary(training(cv)))-1:0.01:max(data.EstimatedSalary(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range,Estimated_salary_range);
XGrid = [xx1(:) xx2(:)];
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
figure
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
hold on
testing_data = data(test(cv),:);
Y = ismember(testing_data.Purchased,labels{1});
 
scatter(testing_data.Age(Y),testing_data.EstimatedSalary(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Age(~Y),testing_data.EstimatedSalary(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
xlabel('Age');
ylabel('Estimated Salary');
title(classifier_name);
legend off, axis tight
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');
 
%________________________________________________________________

TP = Results(2,2);
TN = Results(1,1);
FP = Results(1,2);
FN = Results(2,1);

accuracy = (TP + TN) / (TP + TN + FP + FN);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Accuracy = %.2f%%\n', accuracy * 100);
fprintf('Precision = %.2f%%\n', precision * 100);
fprintf('Recall    = %.2f%%\n', recall * 100);
fprintf('F1 Score  = %.2f%%\n', f1_score * 100);
