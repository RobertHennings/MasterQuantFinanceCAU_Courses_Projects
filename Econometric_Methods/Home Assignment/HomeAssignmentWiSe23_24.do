/*---------------------------------------------------------------------------

Group Members:
- 1) Marque Mollenhauer, stu227420
- 2) Eric Kroll, stu243616
- 3) Josef Fella, stu245231
- 4) Robert Hennings, stu236320
- 5) Ahsan Muhammad, stu243716

---------------------------------------------------------------------------*/
clear
* Making sure the needed packages are installed
* ssc install estout

* ============== Excercise 1) ===============================================

**** a)
* Import and briefly describe the dataset:

cd "/Users/Robert_Hennings/Dokumente/Uni/Master/3.Semester/Econometric Methods/Home Assignment"
use HA_smoking.dta

* Get sense of data
describe
summarize

* Looking at the different distributions of the single variables
tabulate smoker female
* around 70% are non smokers
* within smokers and non-smokers the gender is roughly equally distributed
* within non-smokers slightly more imbalanced towards women
tabulate age smoker

tabulate age if hsdrop==1
tabulate age if hsgrad==1
tabulate age if colgrad==1

**** b)
* Estimate the regression using the OLS method with heteroscedasticity
* robust standard errors. Interpret βˆ1 . Is it statistically significant?

* New variable age^2
gen agesq = age * age

* Check if the operation worked properly
browse age agesq

* use heteroscedasticity robust standard errors
regress smoker smkban age agesq female hsdrop hsgrad colsome colgrad, vce(robust)
est store linear_model

/* Interpretation: Linear probability Model
H0: measures at the work place banning smoking have no effect on becoming a smoker, βˆ1 is equal to 0
H1: measures indeed have an effect, βˆ1 != 0

- If we look purely on the p-value of 0.000 we would argue that its statistically significant, since its below the significance level of 0.05 and even 0.01. So we have statistical evidence to reject our H0 (less smoking with smoking ban).
- On the other hand the coefficient is -0.048052, so smoking measures in place at the work area decrease the predicted probability of being a smoker by 4.8052 %. In absolute terms this is a quiet small magnitude. Rendering the effect strength of the measures taken questionable.

--> Statistically relevant BUT practical importance will be dependent of the study and scale of variable involved.

 */ 

 
**** c)
*Interpretation and Wald test:
* Given the coefficients:
esttab linear_model
/* Does probability of smoking increase or decrease?

Observing the educational related variables inlcuded in the model, one can see
the following trend: 
the higher the education of an individual the less likely or the smaller the chance
that this individual is a smoker, but all estimated parameters are positive
as high school drop outs have an effect of 0.2821909, high school grads: 0.2245546
attended a college: 0.1556154 and finally college grads are least likely ending up
as a smoker with: 0.0433244
where the effect or absolute difference is biggest between attended college and
the college grads, so finishing college is advised

--> Model implies higher probability for hsfropouts than for graduates
*/

* H0 Hypothesis: hsdrop and hsgrad have the same effect on smoker: hsdrop = hsgrad
* test if hsdrop and hsgrad have the same effect for the smoking probability
* Wald test
test hsdrop = hsgrad

/* Interpretation:
- H0: hsdrop = hsgrad (they are equal)
- H1: hsdrop, hsgrad have different effects

- Result: We reject the H0 --> F = 10.25 and p-value 0.0014 (less than 0.05 and even 0.01)
- So both variables are statistically significantly different from each other
*/


**** d)
* Predicted probability for the women: (logic is to store the values & coefficients and calc the probability)

* Set the values for the woman's characteristics
scalar smkban_value = 1
scalar age_value = 70
scalar agesq_value = age_value^2
scalar female_value = 1
scalar hsdrop_value = 0
scalar hsgrad_value = 0
scalar colsome_value = 0
scalar colgrad_value = 1

* Calculate the linear prediction
quiet regress smoker smkban age agesq female hsdrop hsgrad colsome colgrad, vce(robust)
generate linear_prediction = _b[_cons] + ///
							_b[smkban] * smkban_value + ///
                            _b[age] * age_value + ///
                            _b[agesq] * agesq_value + ///
                            _b[female] * female_value + ///
                            _b[hsdrop] * hsdrop_value + ///
                            _b[hsgrad] * hsgrad_value + ///
                            _b[colsome] * colsome_value + ///
                            _b[colgrad] * colgrad_value
di linear_prediction
* -0.01326874
* Test out version with marggins what yields the same results
margins, at(smkban=1 age=70 agesq=4900 female=1 hsdrop=0 hsgrad=0 colsome=0 colgrad=1)
* -.0132687


/* Interpretation:
- smoking measures in place at the work area decrease the predicted probability
  of being a smoker for this individual specified, by roughly 1.32 %

Problems:
- We only assume linear relationship of predictor and regressor
- Especially the positvie coefficient of education may suggest a more complex relationship --> not     	considered here
- Predicted probability isn't standardized -> Can yield values above 1 o below 0 
*/


* ============== Excercise 2) ===============================================

**** a)

*1) Linear Probability Model:
* Changing here the syntax changes nothing, only for better showing in comparison tab later
regress smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(robust)
eststo linear_model

*2) Probit Model:
probit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins, dydx(i.smkban) post
eststo probit_model
* dy/dx: -.048124

/* Effects statistically significant?
H0: The APE is not relevant, it should be 0
H1: The APE is relevant and has an effect, it shoud be !=0

Result: Since the p-value is 0.000 what is <0.05 and even 0.01, we reject the H0
		that the APE is not relevant, i.e. being equal to 0

Interpretation: the estimated average change in the probability of being a smoker is reduced by 4.81%
*/

*3) Logit Model:
logit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins, dydx(i.smkban) post
eststo logit_model
* dy/dx: -.0468536

/* Effects statistically significant?
H0: The APE is not relevant, it should be 0
H1: The APE is relevant and has an effect, it shoud be !=0

Result: Since the p-value is 0.000 what is <0.05 and even 0.01, we reject the H0
		that the APE is not relevant, i.e. being equal to 0
		
Interpretation: the estimated average change in the probability of being a smoker is reduced by 4.68%
*/

* Display a table with all three estimation results
esttab linear_model logit_model probit_model, keep(1.smkban)

**** b)
* Use the probit model
* analytical approach following PC Tutorial 4) MLE, Slides P.4/11 Task c)
* i) male, 40 years old, college graduate.
* First have a look at the filtered dataset
browse if female == 0 & age == 40 & colgrad == 1


* smkban=0 -> Baseline
* use the margins operator
quiet probit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins 0.smkban, at(age=40 agesq=1600 female=0 hsdrop=0 hsgrad=0 colsome=0 colgrad=1) post
* Save the estimated parameter for later use
scalar probit_i_0 = e(b)[1,1]
eststo probit_i_0_tab
* 0.1867095

* smkban=1 -> Compare
quiet probit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins 1.smkban, at(age=40 agesq=1600 female=0 hsdrop=0 hsgrad=0 colsome=0 colgrad=1) post
scalar probit_i_1 = e(b)[1,1]
eststo probit_i_1_tab
* 0.1468431

* Effect strenth as difference
disp probit_i_0 - probit_i_1
* .03986644
esttab probit_i_0_tab probit_i_1_tab

* ii) female, 20 years old, high school dropout.
* First have a look at the filtered dataset
browse if female == 1 & age == 20 & hsdrop == 1

* smkban=0 -> Baseline
* use the margins operator
probit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins 0.smkban, at(age=20 agesq=400 female=1 hsdrop=1 hsgrad=0 colsome=0 colgrad=0) post
* Save the estimated parameter for later use
scalar probit_ii_0 = e(b)[1,1]
eststo probit_ii_0_tab
* .3614542

* smkban=1 -> Compare
probit smoker i.smkban age agesq female hsdrop hsgrad colsome colgrad, vce(oim)
margins 1.smkban, at(age=20 agesq=400 female=1 hsdrop=1 hsgrad=0 colsome=0 colgrad=0) post
scalar probit_ii_1 = e(b)[1,1]
eststo probit_ii_1_tab
* .3034314

* Effect strenth as difference
disp probit_ii_0 - probit_ii_1
* .05802279
esttab probit_ii_0_tab probit_ii_1_tab

* Direct Comparison
esttab probit_i_0_tab probit_i_1_tab probit_ii_0_tab probit_ii_1_tab


// For Male, 40, College Graduate
display "Predicted Probability of a male, 40 Years Old, College Graduate without Smoking Ban: " probit_i_0
display "Predicted Probability of a male, 40 Years Old, College Graduate with Smoking Ban: " probit_i_1
display "Effect of Smoking Ban on Smoking Probability for Males: " probit_i_0 - probit_i_1
display "------------------------------------------------------------"
/*
Explanation:
Among 40-year-old males who are college graduates, the probability of smoking in the workplace without any measures taken is 18.67%. However, if a smoking ban is implemented, this probability decreases to 14.68%. Consequently, the smoking ban results in a reduction of smoking probability by 3.9% within this specific demographic group.
*/

// For Female, 20, High School Dropout
display "Predicted Probability of a female, 20 Years Old, High School Dropout without Smoking Ban: " probit_ii_0
display "Predicted Probability of a female, 20 Years Old, High School Dropouts with Smoking Ban: " probit_ii_1
display "Effect of Smoking Ban on Smoking Probability for Females: " probit_ii_0 - probit_ii_1
/*
Explanation:
For 20-year-old females who are high school dropouts, the lprobability of smoking in the workplace without any measures taken is 36.14%. If a smoking ban is implemented, this probability decreases to 30.34%. Consequently, the smoking ban results in a reduction of smoking probability by 5.8% within this specific demographic group.
*/
