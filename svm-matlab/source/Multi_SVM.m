classdef Multi_SVM
% Code for 3.2
% Multiclass Support Vector Machine
% Implement with SGD
% author: Zhengwei Wei
% date : 2/28/2018
   properties
      Weights
      Losses
   end
   
   methods
      function obj = fit(obj,X,Y,C,lr0,lr1,max_epochs,init_method,valD,valLb)
         % X: d*n matrix; train data
         % Y: 1*n vector; train label
         % C; scalar >0
         % lr0: scalar, learning rate0 
         % lr1: scalar, learning rate1
         % max_epochs: scalar
         if C<0
             fprintf('Error:C should >0')
         end
         d = size(X,1);
         n = size(X,2);
         
         %W = zeros(d,k);
         k = length(unique(Y));
         W = obj.init_param(init_method,d,k);
         obj.Weights = W;
         losses = zeros(max_epochs,1);
         loss = 10000000;
         %obj_val = 0.5*w'*w + C*sum(max(w,0));
         fprintf("strat training: \n");
         epoch = 1;
         delta = 10000;
         best_accuracy = 0;
         while epoch<=max_epochs  && abs(delta)>1
             epoch = epoch+1;
             begin = cputime;
             %fprintf('epoch: %d/%d  ',epoch,max_epochs);
             lr = lr0/(lr1+epoch);
             indexes = randperm(n);
             loss_pre = loss;
             loss = 0;
             for i = indexes   
                 y = Y(i);
                 %size(W')
                 result = W'*X(:,i);
                 result_tmp = result;
                 result_tmp(y) = min(result_tmp)-1;
                 [~,y_hat] = max(result_tmp);
                 L_w = max(result(y_hat)-result(y)+1,0);
                 %L_w = max((W(:,y_hat)'-W(:,y)')*X(:,i)+1,0);
                 for j = 1:k
                     dw_tmp = W(:,j)/n;
                     if L_w==0
                         dw = dw_tmp;
                     else
                         if j==y
                             dw = dw_tmp - C*X(:,i);
                         elseif j==y_hat
                             dw = dw_tmp + C*X(:,i);
                         else
                             dw = dw_tmp;
                         end
                     end
                     W(:,j)=W(:,j)-lr*dw;
                     %loss = loss + W(:,j)'*W(:,j)/2; much slower
                 end
                 loss = loss + L_w;
             end
             Accuracy = obj.cmp_accuracy(W,valD,valLb);
             fprintf('Accuracy = %.2f',Accuracy);
             if Accuracy > best_accuracy
                 best_accuracy = Accuracy;
                 obj.Weights = W;
             end
             loss = loss + sum(diag(W*W'))/2;
             %loss_pre
             %loss
             delta = loss_pre - loss
             losses(epoch) = loss;
             e = cputime-begin;
             fprintf("epoch: %d/%d eplased time: %.2f \n",epoch,max_epochs,e);
         end
         
         obj.Weights = W;
         obj.Losses = losses;
      end
      
      function [score,pred] = predict(obj,X)     
         [score,pred] = max(obj.Weights'*X);
      end
      
      function W = init_param(obj,init_method,d,k)
         if strcmp(init_method,'uniform') 
             W = rand(d,k)-0.5;
         elseif strcmp(init_method,'normal') 
             W = randn(d,k);
         elseif strcmp(init_method,'load')
             W = obj.Weights;
         else
             W = zeros(d,k);
             fprintf(' initial with zero \n')
         end
      end
      
      function weights = get_param(obj)
          weights = obj.Weights;
      end
      
      function obj = load_param(obj,Weights)
         obj.Weights = Weights;
      end
      
      function Accuracy = cmp_accuracy(~,Weights,valD,valLb)
          [~,pred] = max(Weights'*valD);
          error = pred' - valLb;
          T = find(error==0);
          Accuracy = length(T)/length(error);
      end
   end
end