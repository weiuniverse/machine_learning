%% code for 4.1
run('../../vlfeat-0.9.15/toolbox/vl_setup');
addpath('../hw2data/')
[trD, trLb, valD, valLbs, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
bi_clf = Kernel_SVM();
bi_clf = bi_clf.fit(trD,trLb,1);
[w,b,~,~]=bi_clf.get_param();
HW2_Utils.genRsltFile(w, b, 'val', 'valRslt.mat');
[ap,prec,rec]=HW2_Utils.cmpAP('valRslt.mat','val');

ap
pre_rec = plot(rec,prec);
title('precison-recall-curve')
xlabel('recall')
ylabel('precision')
saveas(pre_rec,'prec_rec.png');
