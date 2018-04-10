classdef Kernel_SVM
    % code for 3.1
    %KERNEL_SVM class for kernel SVM, implement with quadratic
    % usage:
    % clf = Kernel_SVM()
    % clf = clf.fit(X,Y,C)
    % Y_test = clf.predict(X_test)
    properties
        weights
        bias
        objective_value
        alpha
    end
    
    methods
        function obj = Kernel_SVM()
            %KERNEL_SVM
            %  init
            obj.bias=0;
        end
        function obj = fit(obj,X,Y,C)
            % fit SVM on (X,Y)
            % X: dxn matrix; data
            % Y; nx1 vector, Label
            K = X'*X;
            [obj.alpha,b,objective] = obj.optimization(K,Y,C);
            obj.weights = X*diag(Y)*obj.alpha;
            obj.bias = b;
            obj.objective_value = objective;
        end
        
        function [alpha,b,objective] = optimization(~,K,Y,C)
            %METHOD1 optimize the dual problem for SVM
            % K : nxn matrix
            % Y : nx1 vector; Label, +1/-1
            % C: scalar
            n = size(Y,1);
            H = diag(Y)*K*diag(Y);
            f = -ones(n,1);
            A = [];
            b = [];
            Aeq = Y';
            beq = 0;
            LB = zeros(n,1);
            UB = C*ones(n,1);
            opts.Display='off';
            [alpha,objective,~] = quadprog(H,f,A,b,Aeq,beq,LB,UB,[],opts);
            objective = -objective;
            % calculate bias
            [~,choice]=max(min(alpha,C-alpha));
            b=Y(choice)-K(choice,:)*diag(Y)*alpha;
        end
        
        function [w,b,obj_val,alp] = get_param(obj)
        % get the parameters w,b
            w = obj.weights;
            b = obj.bias;
            obj_val = obj.objective_value;
            alp = obj.alpha;
        end
        
        function Y_test = predict(X_test)
        % predict for test data
        % X_test: dxn matrix
            Y_test = X_test'*obj.weights+obj.bias;
        end
    end
end

