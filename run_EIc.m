
startup;
flag_y_or_f=0;%y:0 (observation), f:1 (latent)
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


if flag_y_or_f==1
    satisfy_constrs=normcdf(thr_2,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)))-normcdf(thr_1,Ms_Mean_predicted_sh,sqrt(diag(Ms_Var_predicted_sh)));
else
    satisfy_constrs=normcdf(thr_2,Ms_Mean_predicted_sh,sqrt(diag(Ms_tmp22)))-normcdf(thr_1,Ms_Mean_predicted_sh,sqrt(diag(Ms_tmp22)));
end


curr_best_val = min(obj((constr<=thr_2)&(constr>=thr_1)));

if flag_y_or_f==1
    Current_EIc_Val = satisfy_constrs.*EI_normal(curr_best_val-Hyst_Mean_predicted_sh,sqrt(diag(Hyst_Var_predicted_sh)));
else
    Current_EIc_Val = satisfy_constrs.*EI_normal(curr_best_val-Hyst_Mean_predicted_sh,sqrt(diag(Ms_tmp22_)));
end

[sorted_Curr_EIc, ind_Curr_EIc]=sort(Current_EIc_Val,'descend');


save('Results.mat','sorted_Curr_EIc','ind_Curr_EIc')

