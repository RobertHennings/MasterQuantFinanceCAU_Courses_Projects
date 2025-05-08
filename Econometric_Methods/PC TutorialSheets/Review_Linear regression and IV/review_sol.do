clear
eststo clear

cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\group1\Review IV"
* import dataset
use malaria.dta
summarize

* a, b, c, d
* e) 2sls for 2 endogenous variables
ivregress 2sls lngdpc (gadp malrisk = lnmort maleco), vce(robust)
eststo md_2sls
* both estimated parameters are statistically significant and have plausible signs: GDP per capita increase with better Institution and decrease with higher risk for malaria. 

* ==== Compare with the results from task a (OLS)
regress  lngdpc gadp malrisk, vce(robust)
eststo md_ols
esttab md_ols md_2sls, se

* the estimated effect is significant and larger than in task a but in general the results obtained from OLS and 2SLS are consistent. 
* Standard errors increase when using instruments.

* f ) test for exogenous instruments
* using two instruments for two endogenous variables: just identified
* Sargan-Hansen test requires the overidentification -> we can't perform the test. 


* g) reproduce results from table 1 in the paper
ivregress 2sls lngdpc (rule malfal = lnmort maleco) if (popmill != 1 & oil !=1), small 
eststo md_3

* h) 
esttab md_2sls md_3, se
* We have different point estimates between e) and g) because we have different regressor set (gadp, malrisk) and (rule malfal) with different scalings/units.







