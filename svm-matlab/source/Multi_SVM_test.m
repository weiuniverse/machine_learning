%% Code for 3.2
load('../hw2data/q3_2_data.mat')
multi_clf = Multi_SVM();
max_epochs=500;
C = 10;
lr0 = 1;
lr1 = 100;
init_method = 'uniform';
t = cputime;
%W_struct = load('Weight.mat');
%Weight = W_struct.Weight;
%multi_clf= multi_clf.load_param(Weight);

index = randperm(2120);
index_val = index(1:120);
index_tr = index(2000:end);

tr_data = [trD valD(:,index_tr)];
%tr_data = trD;
trD_ex = tr_data.*tr_data;
tr_data = [tr_data;trD_ex];
tr_lb = [trLb;valLb(index_tr)];
%tr_lb = trLb;

val_data = valD(:,index_val);
%val_data = valD;
val_ex = val_data.*val_data;
val_data = [val_data;val_ex];
val_lb = valLb(index_val);
%val_lb = valLb;

multi_clf = multi_clf.fit(tr_data,tr_lb,C,lr0,lr1,max_epochs,init_method,val_data,val_lb);
e=cputime-t;
fprintf("All time: %.2f \n",e);

%% save model
L = multi_clf.Losses;
W = multi_clf.Weights;
save('Weight.mat','W')
save('Loss.mat','L')

%% 3.2.5 Plot loss function
figure(1)
L_fig = plot(L);
saveas(L_fig,'losses.png')

%% 3.2.6
%% 3.2.6(a) predict on validation
fprintf('\n');
fprintf('\n')
[~,pred] = multi_clf.predict(val_data);
error = pred' - val_lb;
T = find(error==0);
Accuracy = length(T)/length(error);
%fprintf('Validation Accuracy = %f \n',Accuracy);
fprintf('3.2.6(a) Validation Error = %f%% \n',(1-Accuracy)*100);

%% 3.2.6(b) predict on trainset
[~,tr_pred] = multi_clf.predict(tr_data);
tr_error = tr_pred' - trLb;
T = find(tr_error==0);
Tr_Accuracy = length(T)/length(tr_error);
%fprintf('Train Accuracy = %f \n',Tr_Accuracy);
fprintf('3.2.6(b): Train Error = %f%% \n',(1-Tr_Accuracy)*100);

%% 3.2.6(c) sum||w_j||^2
sum_w2=sum(diag(W*W'));
fprintf("3.2.6(c): sum||w_j||^2 = %f",sum_w2);

fprintf('\n')
fprintf('\n')
%% 3.2.7 get the submission
tstD_ex = tstD.*tstD;
tstData = [tstD;tstD_ex];
[score,test_pred] = multi_clf.predict(tstData);
index = 1:length(test_pred);
sub = [index' test_pred'];
csvwrite('sub.csv',sub,1,0);
