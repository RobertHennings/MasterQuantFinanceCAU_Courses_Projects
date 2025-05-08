clear

cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\group1\PC3"

use fertil.dta

summarize

*** a)
* run OLS and view predicted value
reg children educ age agesq evermarr urban electric tv, robust
predict children_hat, xb /* predicted value y_hat*/
summarize children_hat
*  linear OLS generates the negative value of children (!)

* non-linear least squares
matrix B = [0,0,0,0,0,0,0,0] /* initial value */
nl (children = exp({b1}*educ + {b2}*age + {b3}*age^2 + {b4}*evermarr + {b5}*urban + {b6}*electric + {b7}*tv + {b8})), initial(B) variables(educ age agesq evermarr urban electric tv) vce(robust)
eststo mdl1

*** b) re-estimate using different starting values
matrix B = [0.5,0.1,0.1,0,0,0,0,0] /* initial value */
nl (children = exp({b1}*educ + {b2}*age + {b3}*age^2 + {b4}*evermarr + {b5}*urban + {b6}*electric + {b7}*tv + {b8})), initial(B) variables(educ age agesq evermarr urban electric tv) vce(robust)
eststo mdl2

esttab
* Starting values are used only for the iterative method for minimization problem and do not affect the estimators.

*** c) Average marginal effect (AME) / Average partial effect (APE)
margins, dydx(educ age)
* AME of age: one year older will have positive impact on woman's fertility on average, keeping others variable constant. This effect is statistically significant (pvalue = 0.000 < 5%)

*** d) Marginal effect at means (MEM/PEM)
margins, atmeans dydx(educ age)
* Interpretation: MEM of age means keeping all variables at their average values, one year older from its average 27 years old will have positive impact on woman's fertility. 
* problem: interpretation does not make sense when having dummy variables, for example married = 0.5(!).

*** e) Marginal effect for specific values
* educ
margins, at(educ=5 age=(15(5)45) evermarr=0 urban=0 electric=0 tv=0) dydx(educ)
marginsplot
/*
* keeping educ = 5, evermarr=0 urban=0 electric=0 tv=0
1) the educ has negative impact on woman's fertility across all age group. The effects are significant in all cases.
2) When the age increase, the marginal effect also increase (in absolute value)
*/
* ---------------
* age
margins, at(educ=5 age=(15(5)45) evermarr=0 urban=0 electric=0 tv=0) dydx(age)
marginsplot
* from the age 15 to 30, one year older have more positive impact on fertility, but when the woman is above 30, one year older will have less positive impact. 




