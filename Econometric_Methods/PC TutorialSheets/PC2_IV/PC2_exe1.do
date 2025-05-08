clear
cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\group1\PC2"
use bwght.dta

summarize

* generate new weight variable in gram
gen bwght_gram = 28*bwght

* Define 4 groups: (1) no cigarettes, (2) 0<packs<0.5, (3) 0.5<=packs<1, (4) packs>=1
gen group = 1 if packs == 0
replace group = 2 if packs> 0 & packs <0.5
replace group =3 if packs >=0.5 & packs <1
replace group =4 if packs >=1
tab group, summarize(bwght_gram)

* a) family income is social-economic factor, and it might has impact both on bwght and packs. Include it can prevent omitted variable bias. 

* b) OLS estimation
regress bwght_gram packs male parity lfaminc, vce(robust)
* the effect is plausible: smoking one additional packs per day reduces the birth weight by 268gram (approximatly by 8% of the average birth weight). The effect is quantitatively relevant and statistically significant (pvalue < 5%)

* c)
* there is still omitted variable, for example personal's living habits, that leads to estimator biased.

* d)
* cigprice might has influences on cigarettes demand and is probably uncorrelate with the birth weight determinants.

* e) 2SLS estimation
ivregress 2sls bwght_gram (packs = cigprice) male parity lfaminc, vce(robust)
* the effect is implausible: positive and substantially large. It is not statistically significant, pvalue = 0.468.

* f) The difference between OLS and 2SLS is large. The 2SLS result is implausible and insignificant. 

* g)
* first-stage regression
regress packs cigprice male parity lfaminc, vce(robust)
test cigprice /* calculate F-statistic excluding instruments */
* the first-stage F statistic is very small (F = 0.89). Hence, cigprice is a weak instrument.

* second way:
ivregress 2sls bwght_gram (packs = cigprice) male parity lfaminc, vce(robust)
estat first /* postestimation command is used directly after the ivregress */









