# Outflows
use pyspeckit to do MCMC fitting

------
mcmc.py

input:
outflows(array of plate , array of ifu)

output:

plate-ifu_mcmc_bins.png : plots of fitting result

plate-ifu_correlation_bins.png : correlation and distribution of each parameters

plate-ifu_bins.fits :
95% HPD interval  mean value , mcmc error and standard deviation of nd1,nd2,b,b_ism,v_wind,v_ism

where 
nd1:column number density of ISM
nd2:column number density of wind
b:doppler parameter of wind
b_ism:doppler parameter of ISM
v0:wind velocity
v_ism:ISM velocity
####

------
7443-12703.txt
qualified bin number of 7443-12703
choose by eye
####
