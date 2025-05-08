* ===== Problem set 1 =========
*** clear data memory
clear 

*** import the dataset
* add the directory path
cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\PC1_group1\Pc1"

use mroz.dta

* open the dataset
browse
browse lwage

* describe and summarize
describe
summarize

***  exercise 1
* generate new variable
gen kidstotal = kidslt6 + kidsge6

* a)  run linear regression
regress lwage kidstotal, vce(robust)
* beta = -0.06 -> one additional child reduces the mother's wage by 6% on average, keeping elsething else constant.

* p-value = 0.021 < 0.05 , the effect is stastically significant at 5%.

* why should we use the ln(wage) instead of wage?
* interpretation: we are interested in the percentage change of wage.
* plot the kernel density
twoway kdensity lwage || kdensity wage
* when using lnwage, the emprical distribution of lnwage is closer to normal distribution.
* The variation within the lwage is more stable -> reduce the heterskedasticity. 

* b) Is the effect quantitatively relevant and statistically significant?
* The effect -6% is quantitatively relevant (think in terms of policy decision). The effect is statistically significant (pvalue < 5%)


* c) Report a 90% confidence interval for beta1. Interpret.
regress lwage kidstotal, vce(robust)  level(90)

* 90 confidence interval [-0.1  -0.017]: when repeating our study many times, then 90% of the times the true value will fall into this range. 

* the estimation effect -0.06 stays inside the 90% confidence interval -> we can almost sure that the true value of beta lies inside this CI.

*** d) is the kidstotal an important driver of the female wage?
* R-squared:  1.38% . This means the number of the children explain only 1.38% variation in the log(wage) -> the kidstotal is not important driver of the log(wage).

*** e) 
* in this linear regression, it is very likely that there is ommitted variable, for example: education.


* ================= Exe 2 ===============
* ====== a)
/* exper: important factor for wage, potentially correlated with the kidstotal eventhough the sign of correlation is unclear.
For example, married women with more kids may be older, it is likely that they will have more working experience -> positive correlation between exper and kidstotal.
On the other hand, the women have more childre, probably she has to stay at home to take care of their children and then she has less working experience -> negative correlation between exper and kidstotal.

* education: important driver of the wage. Women with higher education usually have less children -> potentially negative correlation between education and kidstotal

* age: potential driver of wages via working experience. Also, having positive correlation with kidstotal. 
*/

* b) re-estimate the model including the control variables
regress lwage kidstotal exper expersq educ age, vce(robust)

* b) and c)
* the effect now is 1.8% -> one additional child reduce the mother's wage by only 1.8%. The effect is less quantitatively relevant compared to the effect in exe 1, and also the effect is not statistically significant (pvalue: 0.536 > 5%)

* Apprarently, the introduction of exper, exper^2, educ and age as explanatory variables corrects the bias that appeared in the simple linear regression of exe 1.

* d ) 
* Is there any endogeniety to worry about?
* If you have strong argument to state that the OVB in your regression reduce to a negligible level, you can say that beta_1 is causal effect. Better to do some robustness check. 

* ============ Exe 3 ===========
* a)
* we expect that the young kids have stronger impact on the mother's wage than the old kids.
* the kids less than 6 years old need more care from the mother -> this prevent the mother from full-time work. On contrary, the kids from 6 until 18 can take care of themselves better, and the mother have more time to focus on her career. 

* b) and c ) 
regress lwage kidslt6 kidsge6 exper expersq educ age, vce(robust)
*  the effect of young kids (6%) is more quantitatively relevant than the effect of old kids (1.45%). However, both effects are not statistically significant. 

*d ) H0: beta_1 = beta_2 = 0
test kidslt6 kidsge6
* p-value = 0.78 > 5%, we fail to reject the null hypothesis. In conclusion, we don't find enough evidence to rejec that the effect of the old kids and young kids are jointly equal to zeros. 

*e ) LM test: H0: beta_1 = beta_2 = 0 (assuming homoscedasticity)

* Step 1: estimate the restricted model (the model under the null hypothesis), we obtain the residual (utilde)
regress lwage exper expersq educ age /* here we use the homoscedasticity se */
predict utilde, residuals /* obtain the residual in the regression */
* Step 2: estimate the auxiliary regression model
regress utilde kidslt6 kidsge6 exper expersq educ age
gen LMstat = e(N)*e(r2) /* calculate the LM statistic, gen = generate */
gen cv = invchi2(2,0.95) /*inverse of cdf from chi-squared distribution with 2 df evaluated at 95% */
gen pval = chi2tail(2,LMstat)

* display the result
disp "LM-test statistic = " LMstat 
disp "critical value (5%) = " cv
disp "p-value = " pval

* pvalue = 0.69>5% -> fail to reject the null hypothesis
* LM test and F test have consistent resutl.

* f ) H0: beta_1 = beta_2
regress lwage kidslt6 kidsge6 exper expersq educ age, vce(robust)
test kidslt6 = kidsge6
* pvalue = 0.66 > 5% -> fail to reject H0. 
* Conclusion: we don't have enough evidence to reject that the effects of young kids are equal to the effects of old kids.

























