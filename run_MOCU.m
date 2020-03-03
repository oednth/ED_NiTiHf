%This code uses GPML toolbox
startup;

flag_nominal=1;
load('training_file.mat');
load('test_file.mat');
obj=Afms;
constr=Ms;
thr_2 = 300;%/1000; %Upper range for constraint
thr_1 = 200;%/1000; %Lower range for constraint

if flag_nominal==1
   Cv=Cv_nom;
   e_a=e_a_nom;
   x_Hf=x_Hf_nom;
   x_Ni=x_Ni_nom;
   x_Ti=x_Ti_nom;
end

constr = constr.*1000;
obj = obj.*1000;

e_a_b = e_a;
e_a_b = (e_a_b>7); 

X{1}=[x_Ni,x_Ti,x_Hf,Cv];
%X{1}=[x_Ni,x_Ti,x_Hf,Cv,e_a_b,1-e_a_b];

n=size(X{1},1);

p=size(Xnew{1},2);

e_a_b_test = e_a_test;
e_a_b_test = (e_a_b_test>7);

Xnew{1}=[x_Ni,x_Ti,x_Hf,Cv_test];
%Xnew{1}=[x_Ni,x_Ti,x_Hf,Cv_test,e_a_b_test,1-e_a_b_test];


%Confidence interval setting
ci_int=0.1;
threshhhh = 1 - (2.*ci_int);
coeff_confidence=norminv((1-ci_int),0,1);




%Fitting GPs

mp = {@meanConst};
hyp2.mean = mean(constr);
likfunc = @likGauss;

sff=std(constr)./sqrt(2);
lll = std(X{1})';
covfunc = {@covMaternard,5}; hyp2.cov = log([lll.*ones(p,1);sff]);
hyp2.lik = log(sqrt(var(constr))./100);
hyp2 = minimize(hyp2, @gp, -200, @infExact,mp, covfunc, likfunc,X{1}, constr);
[Ms_tmp21,Ms_tmp22,Ms_Mean_predicted_sh,Ms_Var_predicted_sh] = gp_new(hyp2, @infExact, mp, covfunc, likfunc, X{1} , constr , Xnew{1});
Ms_int = zeros(size(Xnew{1},1),2);
Ms_int(:,1) = Ms_Mean_predicted_sh - coeff_confidence.*sqrt(diag(Ms_Var_predicted_sh));
Ms_int(:,2) = Ms_Mean_predicted_sh + coeff_confidence.*sqrt(diag(Ms_Var_predicted_sh));

probs_sat=normcdf(thr_2,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)))-normcdf(thr_1,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)));
ind_feasible = find(probs_sat>=threshhhh);
%
mp3 = {@meanConst};
hyp3.mean = mean(obj);
likfunc3 = @likGauss;
sff3=std(obj)./sqrt(2);
lll3 = std(X{1})';
covfunc3 = {@covMaternard,5}; hyp3.cov = log([lll3.*ones(p,1);sff3]);
hyp3.lik = log(sqrt(var(obj))./100);
hyp3 = minimize(hyp3, @gp, -200, @infExact,mp3, covfunc3, likfunc3,X{1}, obj);
[Ms_tmp21_,Ms_tmp22_,Hyst_Mean_predicted_sh,Hyst_Var_predicted_sh] = gp_new(hyp3, @infExact, mp3, covfunc3, likfunc3, X{1} , obj , Xnew{1});
Hyst_int = zeros(size(Xnew{1},1),2);
Hyst_int(:,1) = Hyst_Mean_predicted_sh - coeff_confidence.*sqrt(diag(Hyst_Var_predicted_sh));
Hyst_int(:,2) = Hyst_Mean_predicted_sh + coeff_confidence.*sqrt(diag(Hyst_Var_predicted_sh));


% Pure exploitation (or IBR)
not_satisfy_constrs=normcdf(thr_2,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)))-normcdf(thr_1,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)));
not_satisfy_constrs = ones(size(not_satisfy_constrs))-not_satisfy_constrs;
Current_IBR_Val = not_satisfy_constrs.*Hyst_Mean_predicted_sh;
[sorted_Curr_IBR, ind_Curr_IBR]=sort(Current_IBR_Val);

%MOCU Experiment Design

X_feasible = Xnew{1}(ind_feasible,:);


%setting mean vectors, covariance matrices, and observation precisions
mean_cell = cell(1,2);
beta_cell = cell(1,2);
cov_cell = cell(1,2);
mean_cell{1}=-Hyst_Mean_predicted_sh;
mean_cell{2}=Ms_Mean_predicted_sh;
beta_meas = repmat((exp(2.* hyp3.lik)).^(-1),size(Hyst_Mean_predicted_sh,1),1);
beta_meas_c = repmat((exp(2.* hyp2.lik)).^(-1),size(Ms_Mean_predicted_sh,1),1);
beta_cell{1}=beta_meas;
beta_cell{2}=beta_meas_c;
cov_cell{1}=Hyst_Var_predicted_sh;
cov_cell{2}=Ms_Var_predicted_sh;


tic
[ind_batch_mocu,all_all_mocu] = constrained_mocu(mean_cell,beta_cell,cov_cell,1,1000,1,threshhhh,thr_1,thr_2);
toc
[sorted_all_all_mocu, ind_batch_mocu_sorted]=sort(all_all_mocu,'descend');

save('Results.mat','ind_batch_mocu','ind_batch_mocu_sorted','sorted_all_all_mocu','sorted_Curr_IBR','ind_Curr_IBR')

