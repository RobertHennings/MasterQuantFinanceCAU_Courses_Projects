clear

cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\group1\PC4"

use mroz.dta
summarize nwifeinc age educ exper kidslt6 kidsge6

* Note: careful about the ME of exper because of its squared term in regression. 
* Here, we consider the ME of nwifeinc, educ and age.
********************* a ********************
* === linear probability model with robust standard error
quiet reg inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)
eststo md_ols
* APEs/AMEs of nwifeinc, educ and age 
margins, dydx(nwifeinc educ age)
* PEMs/MEMs of nwifeinc, educ and age
margins, dydx(nwifeinc educ age) at((mean) nwifeinc educ age)

* === logit probability model with variance estimator based on observed information matrix
quiet logit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
eststo md_logit
* APEs/AMEs of nwifeinc, educ and age 
margins, dydx(nwifeinc educ age)
* PEMs/MEMs of nwifeinc, educ and age
margins, dydx(nwifeinc educ age) at((mean) nwifeinc educ age)

* === probit model with variance estimator based on observed information matrix
quiet probit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(oim)
eststo md_probit
* APEs/AMEs of nwifeinc, educ and age 
margins, dydx(nwifeinc educ age)
* PEMs/MEMs of nwifeinc, educ and age
margins, dydx(nwifeinc educ age) at((mean) nwifeinc educ age)

esttab md_ols md_logit md_probit, se













