********************** d ********************
* re-load the dataset because the variable exper and expersq are changed in task c
clear
use mroz.dta 
*******************************
* ==== linear 
quiet reg inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)
* Wald test
test fatheduc motheduc
* Likelihood ratio test
* === unrestricted model
quiet reg inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)
scalar llu = e(ll)
* === restricted model (model under H0)
quiet reg inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)
scalar llr = e(ll)
* LR statistic
scalar LR = 2*(llu-llr)
di "LR statistic = " LR
di "p-value = " = chi2tail(2,LR) 

*********************************
* ==== logit
quiet logit inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
* Wald test
test fatheduc motheduc
* Likelihood ratio test
* === unrestricted model
quiet logit inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
scalar llu = e(ll)
* === restricted model (model under H0)
quiet logit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
scalar llr = e(ll)
* LR statistic
scalar LR = 2*(llu-llr)
di "LR statistic = " LR
di "p-value = " = chi2tail(2,LR) 
*** second way to perform the LR test for logit and probit model
* === unrestricted 
quiet logit inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
eststo md_1
* === restricted
quiet logit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
eststo md_2
lrtest md_1 md_2 










