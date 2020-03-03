function [y1,y2] = chem_props(x_Ni_,x_Ti_,x_Hf_,V_Ni,V_Ti,V_Hf,flag_ave)
%Calculate chemical properties
%column inp & out
if any(x_Ni_>1)
   x_Ni_=x_Ni_./100;
   x_Ti_=x_Ti_./100;
   x_Hf_=x_Hf_./100;
end

N_Ni=28;
N_Ti=22;
N_Hf=72;

e_a = V_Ni.*x_Ni_ + V_Ti.*x_Ti_ + V_Hf.*x_Hf_;
Cv = e_a./(N_Ni.*x_Ni_ + N_Ti.*x_Ti_ + N_Hf.*x_Hf_);

if flag_ave == 1
   y1=e_a;
   y2=Cv;
else
   y1 = e_a; 
   y2=[];
end

end

