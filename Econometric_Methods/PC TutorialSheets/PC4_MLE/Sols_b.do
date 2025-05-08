******************* b *****************
* ==== linear probability
quiet reg inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)

* ME (PE) of age evaluated at the first, second and third quartile
margins, dydx(age) at((p25) nwifeinc age educ exper expersq kidslt6 kidsge6) /*first quartile */
margins, dydx(age) at((p50) nwifeinc age educ exper expersq kidslt6 kidsge6) /*second quartile */ 
margins, dydx(age) at((p75) nwifeinc age educ exper expersq kidslt6 kidsge6) /*third quartile */
**********************************************
* ==== logit model
quiet logit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)

* ME (PE) of age evaluated at the first, second and third quartile
margins, dydx(age) at((p25) nwifeinc age educ exper expersq kidslt6 kidsge6) /*first quartile */
margins, dydx(age) at((p50) nwifeinc age educ exper expersq kidslt6 kidsge6) /*second quartile */ 
margins, dydx(age) at((p75) nwifeinc age educ exper expersq kidslt6 kidsge6) /*third quartile */

***********************************************
* ==== probit model
quiet probit inlf nwifeinc age educ exper expersq kidslt6 kidsge6, vce(robust)

* ME (PE) of age evaluated at the first, second and third quartile
margins, dydx(age) at((p25) nwifeinc age educ exper expersq kidslt6 kidsge6) /*first quartile */
margins, dydx(age) at((p50) nwifeinc age educ exper expersq kidslt6 kidsge6) /*second quartile */ 
margins, dydx(age) at((p75) nwifeinc age educ exper expersq kidslt6 kidsge6) /*third quartile */