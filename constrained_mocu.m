%{
Constrained MOCU

notation for the following:
K is the number of alternatives.
M is the number of exoperimemnts in a batch
K x M stands for a matrix with K rows and M columns
Q_MC is the number of Monte Carlo samples

This function takes in
mu_A:   cell array containing predicted prior mean - each cell (K x 1)
beta_A: cell array containing measurement precision (1/lambda(x)) - each cell (K x 1)
covA:   cell array containing predicted initial covariance matrix - each cell (K,K)
M:      how many measurements will be made in a batch (scalar)
Q_MC:   how many MC samples will be used in batch approximation
flag_type: How to consider constraints 1:Multiply the objective.
2:Constraining search space
flag_thresh: Threshhold for constraining search space.

mu_A and beta_A and covA are cell arrays:
first cell for objective, second cell for constraint.


And returns
choices:    Alternatives picked at each iteration (1 x M)
All_MOCU:   MOCU calculated at each iteration (K x M)
%}

function [choices,All_MOCU]=constrained_mocu(mu_A,beta_A,covA,M,Q_MC,flag_type,flag_thresh,thr_1,thr_2)
mu_0=mu_A{1};
beta_W = beta_A{1};
covM=covA{1};

mu_C=mu_A{2};
beta_C = beta_A{2};
covC=covA{2};

K=length(mu_0); %number of available choices


choices=[];
All_MOCU=[];
sig_tild_all=[];
sig_tild_all_C=[];
x = 0;
for k=1:M 
    if k==1
    
        if flag_type==1
            [x,currKGs]=MC_Result_Constr_multiplication(mu_0,sig_tild_all,covM,beta_W,Q_MC,mu_C,sig_tild_all_C,covC,beta_C,thr_1,thr_2);
        else
           [x,currKGs]=MC_Result_Constr_constrained(mu_0,sig_tild_all,covM,beta_W,Q_MC,mu_C,sig_tild_all_C,covC,beta_C,flag_thresh,thr_1,thr_2); 
        end
        
        choices = [choices;x];
        All_MOCU = [All_MOCU,currKGs];
    else
        e_x=zeros(K,1);
        e_x(x)=1;
        sig_tild=covM(x,:)/sqrt(1/beta_W(x) + covM(x,x));
        covM = covM - (covM*e_x*e_x'*covM)/((1/beta_W(x)) + covM(x,x));
        sig_tild_C=covC(x,:)/sqrt(1/beta_C(x) + covC(x,x));
        covC = covC - (covC*e_x*e_x'*covC)/((1/beta_C(x)) + covC(x,x));
        if (any(isnan(sig_tild)) || any(any(isnan(covM))) || any(isnan(sig_tild_C)) || any(any(isnan(covC))) )
            warning('sig_tilde or Cov is NaN');
        end
        sig_tild_all = [sig_tild_all,sig_tild'];
        sig_tild_all_C = [sig_tild_all_C,sig_tild_C'];
        if flag_type==1
            [x,currKGs]=MC_Result_Constr_multiplication(mu_0,sig_tild_all,covM,beta_W,Q_MC,mu_C,sig_tild_all_C,covC,beta_C,thr_1,thr_2);
        else
           [x,currKGs]=MC_Result_Constr_constrained(mu_0,sig_tild_all,covM,beta_W,Q_MC,mu_C,sig_tild_all_C,covC,beta_C,flag_thresh,thr_1,thr_2); 
        end
        choices = [choices;x];
        All_MOCU = [All_MOCU,currKGs];
   end
    
end

    

end


% MC approximation of next experiment suggestion in a bacth when having
% constraints
% minimizing: maximizing(-theta * (1-P(satisfy)))
function [next_x,sum_x]=MC_Result_Constr_multiplication(thet_0,sig_all_b,covB,beta_W_loc,QQ,thet_C,sig_all_c,covCloc,beta_W_Cloc,thr_1,thr_2)

    x_length=length(thet_0);
    sum_x = zeros(x_length,1);
    bb=size(sig_all_b,2)+1;
    for i=1:x_length
        tic
        sig_tild_new=covB(i,:)/sqrt(1/beta_W_loc(i) + covB(i,i));
        sig_tild_new_C=covCloc(i,:)/sqrt(1/beta_W_Cloc(i) + covCloc(i,i));
        e_x_tmp=zeros(x_length,1);
        e_x_tmp(i)=1;
        covCloc_tmp = covCloc - (covCloc*e_x_tmp*e_x_tmp'*covCloc)/((1/beta_W_Cloc(i)) + covCloc(i,i));
        R = normrnd(0,1,QQ,bb);
        R_C = normrnd(0,1,QQ,bb);
        sig_tmpp = [sig_all_b,sig_tild_new'];
        sig_tmpp_C = [sig_all_c,sig_tild_new_C'];
        addition_term = zeros(x_length,QQ);
        not_satisfy_constrs = zeros(x_length,QQ);
        for j=1:QQ
         %mean and probability updates   
         addition_term(:,j) = sum(sig_tmpp.*repmat(R(j,:),x_length,1),2);   
         not_satisfy_constrs(:,j) = sum(sig_tmpp_C.*repmat(R_C(j,:),x_length,1),2);
        end
        not_satisfy_constrs=repmat(thet_C,1,QQ)+not_satisfy_constrs;
        not_satisfy_constrs=normcdf(thr_2,not_satisfy_constrs,repmat(sqrt(diag(covCloc_tmp)),1,QQ))-normcdf(thr_1,not_satisfy_constrs,repmat(sqrt(diag(covCloc_tmp)),1,QQ));
        not_satisfy_constrs = ones(size(not_satisfy_constrs))-not_satisfy_constrs;
        that_prime=repmat(thet_0,1,QQ)+addition_term;
        sum_x(i)=sum(max(that_prime.*not_satisfy_constrs));
        toc
    end
    [~,next_x]=max(sum_x);
end

% MC approximation of next experiment suggestion in a bacth when having
% constraints
% minimizing: maximizing(-theta) over a constrained set.
function [next_x,sum_x]=MC_Result_Constr_constrained(thet_0,sig_all_b,covB,beta_W_loc,QQ,thet_C,sig_all_c,covCloc,beta_W_Cloc,prob_thresh,thr_1,thr_2)

    x_length=length(thet_0);
    sum_x = zeros(x_length,1);
    bb=size(sig_all_b,2)+1;
    for i=1:x_length
        sig_tild_new=covB(i,:)/sqrt(1/beta_W_loc(i) + covB(i,i));
        sig_tild_new_C=covCloc(i,:)/sqrt(1/beta_W_Cloc(i) + covCloc(i,i));
        e_x_tmp=zeros(x_length,1);
        e_x_tmp(i)=1;
        covCloc_tmp = covCloc - (covCloc*e_x_tmp*e_x_tmp'*covCloc)/((1/beta_W_Cloc(i)) + covCloc(i,i));
        R = normrnd(0,1,QQ,bb);
        R_C = normrnd(0,1,QQ,bb);
        sig_tmpp = [sig_all_b,sig_tild_new'];
        sig_tmpp_C = [sig_all_c,sig_tild_new_C'];
        addition_term = zeros(x_length,QQ);
        satisfy_constrs = zeros(x_length,QQ);
        for j=1:QQ
         %mean and probability updates   
         addition_term(:,j) = sum(sig_tmpp.*repmat(R(j,:),x_length,1),2);   
         satisfy_constrs(:,j) = sum(sig_tmpp_C.*repmat(R_C(j,:),x_length,1),2);
        end
        satisfy_constrs=repmat(thet_C,1,QQ)+satisfy_constrs;
        satisfy_constrs=normcdf(thr_2,satisfy_constrs,repmat(sqrt(diag(covCloc_tmp)),1,QQ))-normcdf(thr_1,satisfy_constrs,repmat(sqrt(diag(covCloc_tmp)),1,QQ));
        binary_mults = ones(size(satisfy_constrs));
        binary_mults(satisfy_constrs<prob_thresh)=Inf;
        
        if any(sum(binary_mults==1)==0)
           warning('Threshold too tight'); 
        end
        
        that_prime=repmat(thet_0,1,QQ)+addition_term;
        that_prime = that_prime.*binary_mults;
        that_prime(that_prime==Inf)=-Inf;
        sum_x(i)=sum(max(that_prime));
        
    end
    [~,next_x]=max(sum_x);
end






