function ego_vals = EI_normal(z,s_sqrt)
%EI Calculation for normal
ego_vals = ( (s_sqrt).*(normpdf(z./s_sqrt)) ) + ( (z).*(normcdf(z./s_sqrt)) );

end