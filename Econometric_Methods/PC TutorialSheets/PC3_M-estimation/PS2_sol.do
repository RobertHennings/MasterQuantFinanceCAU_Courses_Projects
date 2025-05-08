clear
eststo clear

use crime.dta
summarize

*** a) robust regression
rreg crime poverty single
eststo mdl1 

*** b) OLS regression
reg crime poverty single
eststo mdl2

esttab mdl1 mdl2, se

*** c) 
rreg crime poverty  
predict crime_rreg 
reg crime poverty
predict crime_OLS
reg crime poverty if state != "dc" /* OLS excluding the DC*/
predict crime_OLS_exl_dc

** graph
twoway (scatter crime poverty) (line crime_rreg crime_OLS crime_OLS_exl_dc poverty), legend(label(2 "robust") label(3 "normal OLS") label(4 "OLS exl outlier"))





