***************** c**************
* we consider the logit model
logit inlf nwifeinc educ exper expersq kidslt6 kidsge6, vce(oim)

* analytical approach
gen exper1  = exper + 1
gen expersq1 = (exper+1)^2

* calculate the predicted probability from the baseline experience
gen G0_hat = exp(_b[_cons]+ _b[nwifeinc]*nwifeinc + _b[educ]*educ + _b[exper]*exper + _b[expersq]*expersq + _b[kidslt6]*kidslt6 + _b[kidsge6]*kidsge6) /(1+exp(_b[_cons]+ _b[nwifeinc]*nwifeinc + _b[educ]*educ + _b[exper]*exper + _b[expersq]*expersq + _b[kidslt6]*kidslt6 + _b[kidsge6]*kidsge6))

gen G1_hat = exp(_b[_cons]+ _b[nwifeinc]*nwifeinc + _b[educ]*educ + _b[exper]*exper1 + _b[expersq]*expersq1 + _b[kidslt6]*kidslt6 + _b[kidsge6]*kidsge6) /(1+exp(_b[_cons]+ _b[nwifeinc]*nwifeinc + _b[educ]*educ + _b[exper]*exper1 + _b[expersq]*expersq1 + _b[kidslt6]*kidslt6 + _b[kidsge6]*kidsge6))

gen delG = G1_hat - G0_hat
sum delG
gen APE_exper = r(mean)
dis "APE of exper = " APE_exper

* empirical approach
logit inlf nwifeinc educ exper expersq kidslt6 kidsge6, vce(oim)
predict inlf_hat_0

replace expersq = (exper+1)^2
replace exper = exper+1
predict inlf_hat_1

gen diff_inlf = inlf_hat_1 - inlf_hat_0
sum diff_inlf
gen APE_exper_empirical = r(mean)
dis " APE of exper (empirical way ) = " APE_exper_empirical
dis "APE of exper = " APE_exper







