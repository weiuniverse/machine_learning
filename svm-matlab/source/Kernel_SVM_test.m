%% code for 3.1
load('../hw2data/q3_1_data.mat');

%% 3.1.4
C = 0.1000;
clf = Kernel_SVM();
clf = clf.fit(trD,trLb,C);
[~,~,obj_val,alpha] = clf.get_param();

Val_pred = clf.predict(valD);
TP=sum((Val_pred>=0)&(valLb==1));
TN=sum((Val_pred<0)&(valLb==-1));
FP=sum((Val_pred>=0)&(valLb==-1));
FN=sum((Val_pred<0)&(valLb==1));
accuracy=(TP+TN)/length(valLb);
fprintf('C = %.2f:\n',C);
fprintf('Accuracy = %.2f %%.\n', accuracy*100);
fprintf('Objective_Value = %.2f.\n', obj_val);
fprintf('# of support vectors = %d.\n', sum(alpha>1e-4));
fprintf('confusion matrix:\n');
Mtx = [TP,FP;FN,TN]

%% 3.1.5
C = 10;
clf = Kernel_SVM();
clf = clf.fit(trD,trLb,C);
[~,~,obj_val,alpha] = clf.get_param();

Val_pred = clf.predict(valD);

TP=sum((Val_pred>=0)&(valLb==1));
TN=sum((Val_pred<0)&(valLb==-1));
FP=sum((Val_pred>=0)&(valLb==-1));
FN=sum((Val_pred<0)&(valLb==1));
accuracy=(TP+TN)/length(valLb);
fprintf('C = %.2f:\n',C);
fprintf('Accuracy = %.2f %%.\n', accuracy*100);
fprintf('Objective_Value = %.2f.\n', obj_val);
fprintf('# of support vectors = %d.\n', sum(alpha>0.001));
fprintf('confusion matrix:\n');
Mtx = [TP,FP;FN,TN]
