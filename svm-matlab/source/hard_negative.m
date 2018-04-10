%% code for 4.2 hard negative algorithm
run('../../vlfeat-0.9.15/toolbox/vl_setup');
addpath('../hw2data/')


% get the data
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
PosD=[trD(:,trLb==1)];
NegD=[trD(:,trLb==-1)];
num_PosD=size(PosD,2);
num_NegD=size(NegD,2);
trData=[PosD,NegD];
trLabel=[ones(num_PosD,1);-ones(num_NegD,C)];

% init SVM
C = 1;
clf = Kernel_SVM();
clf = clf.fit(trData,trLabel,C);
[w,b,obj_val,alpha]=clf.get_param();

max_epochs = 10;
Train_obj_val=zeros(1,max_epochs);
Val_ap=zeros(1,max_epochs);

for epoch = 1:max_epochs
    fprintf('epoch: %d/%d',epoch,max_epochs);
    HW2_Utils.genRsltFile(w, b, 'train', 'trainRslt.mat');
    [ap,prec,rec]=HW2_Utils.cmpAP('trainRslt.mat','train');
    load('trainRslt.mat','rects');
    load('trainAnno.mat','ubAnno');
    Image_num = length(rects);
    % get the features of all rects
    % part code borrow from HW2_Utils.getPosAndNeg()
    All_features = cell(1,Image_num);
    for i = 1:Image_num
        im = imread(sprintf('../hw2data/trainIms/%04d.jpg',i));
        rects_i = rects{i};
        features = zeros(size(trD,1),size(rects_i,2));
        for j = 1:size(rects_i,2)
            rect = rects_i(:,j);
            bbox = zeros(1,4);
            [w,h,c] = size(im);
            bbox(2) = int16(max(0,rect(2)));
            bbox(4) = int16(min(w,rect(4)));
            bbox(1) = int16(max(0,rect(1)));
            bbox(3) = int16(min(h,rect(3)));
            imReg = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
            imReg = imresize(imReg, HW2_Utils.normImSz);
            feature = HW2_Utils.cmpFeat(rgb2gray(imReg));
            % feature = HW2_Utils.l2Norm(feature);
            features(:,j) = feature;
        end
        All_features{i} = features;
    end
    All_features = cat(2,All_features{:});

    % A <-non support vectors in NegD
    % get NegD/A
    NegD_minus_A=NegD(:,alpha(end-num_NegD+1:end)>0.0001);

    % Hardest Negtive example -> B
    % code borrow from HW2_Utils.cmpAP()
    if length(rects) ~= length(ubAnno)
        error('result and annotation files mismatch. Are you using the right dataset?');
    end
    [detScores, isTruePos] = deal(cell(1, nIm));
    for i=1:nIm
        rects_i = rects{i};
        detScores{i} = rects_i(5,:);
        ubs_i = ubAnno{i}; % annotated upper body
        isTruePos_i = -ones(1, size(rects_i, 2));
        for j=1:size(ubs_i,2)
            ub = ubs_i(:,j);
            overlap = HW2_Utils.rectOverlap(rects_i, ub);
            % change threshold from 0.5 to 0.3
            isTruePos_i(overlap >= 0.3) = 1;
        end
        isTruePos{i} = isTruePos_i;
    end
    detScores = cat(2, detScores{:});
    isTruePos = cat(2, isTruePos{:});

    % Sort the result by Score, choose the negative example with highest
    % score as hard negative examples
    [new_score,index]=sort(detScores,'descend');
    neg_index=find(isTruePos(index)==-1);
    len = length(neg_index);
    %choose the first 1000 neg example
    hard_neg_index=index(neg_index(1:min(len,1000)));

    B=HW2_Utils.l2Norm(All_features(:,hard_neg_index));
    % update NegD <- NegD\A U B
    NegD=[NegD_minus_A,B];

    % train the svm
    % (w,b) <- trainSVM(PosD,NegD)
    num_PosD=size(PosD,2);
    num_NegD=size(NegD,2);
    trData=double([PosD,NegD]);
    trLabel=[ones(num_PosD,1);-ones(num_NegD,1)];
    %clf = Kernel_SVM();
    clf = clf.fit(trData,trLabel,C);
    [w,b,obj_val,alpha] = clf.get_param();
    Train_obj_val(epoch)=obj_val;
    obj_val
    HW2_Utils.genRsltFile(w, b, 'val', 'valRslt.mat');
    ap=HW2_Utils.cmpAP('valRslt.mat','val');
    Val_ap(epoch)=ap;
    ap
end

%% code for 4.3
List = 1:10
% objective_value Curve
figure(1)
obj_fig=plot(List,Train_obj_val)
xlabel('epoch')
ylabel('objective_value')
saveas(obj_fig,'obj_fig.png')
$ AP Curve
figure(2)
ap_fig=plot(List,Val_ap)
xlabel('epoch')
ylabel('Val_ap')
saveas(ap_fig,'ap_fig.png')

%% save model
save('Train_obj_val.mat','Train_obj_val')
save('Val_ap.mat','Val_ap');
[w,b,~,~] = clf.get_param();
save('Weight.mat','w')
save('b.mat','b')
