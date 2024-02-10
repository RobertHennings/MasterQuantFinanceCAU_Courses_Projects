******************** e *******************
tab kidslt6
drop if kidslt6 == 3 /* only 3 observation */

* split quantitative variables into dummy
gen kid0 = 0
replace kid0 = 1 if kidslt6<1
gen kid1 = 0
replace kid1 = 1 if kidslt6 == 1
gen kid2 = 0
replace kid2 = 1 if kidslt6 == 2

** linear
* reg use kidlst6 (1)
reg inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)
scalar llr = e(ll)
* reg use dummy variables (2)
reg inlf fatheduc motheduc nwifeinc age educ exper expersq kid1 kid2 kidsge6, vce(robust)
scalar llu = e(ll)
* The model using kidslt6 is more restrictive than the model using dummy because the first model indicate that having one additional child only has the same impact no matter if the child is the first or the second child. 

* test the md2 against md1
* LR test
scalar LR = 2*(llu-llr)
dis "LR stat = " LR
dis "pvalue = " chi2tail(1,LR)
* Wald test
reg inlf fatheduc motheduc nwifeinc age educ exper expersq kid1 kid2 kidsge6, vce(robust)
test kid2 = 2*kid1
*********************************************
* logit
logit inlf fatheduc motheduc nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
scalar llr = e(ll)
* reg use dummy variables (2)
logit inlf fatheduc motheduc nwifeinc age educ exper expersq kid1 kid2 kidsge6, vce(oim)
scalar llu = e(ll)
* test the md2 against md1
* LR test
scalar LR = 2*(llu-llr)
dis "LR stat = " LR
dis "pvalue = " chi2tail(1,LR)
* Wald test
logit inlf fatheduc motheduc nwifeinc age educ exper expersq kid1 kid2 kidsge6, vce(oim)
test kid2 = 2*kid1








