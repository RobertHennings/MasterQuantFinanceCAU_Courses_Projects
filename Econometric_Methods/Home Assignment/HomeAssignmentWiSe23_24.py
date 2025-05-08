# Econometric Methods Home Assignment WiSe 23/24
"""
Group Members:
Josef Fella
Robert Hennings
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import scipy.stats as st
path = r"/Users/Robert_Hennings/Dokumente/Uni/Master/3.Semester/Econometric Methods/Home Assignment/"
data_file = "HA_smoking.dta"

##################################################################### Execise 1)
# a)
# Import and briefly describe the dataset.
smoking_data = pd.read_stata(path + data_file)

smoking_data.query("colgrad==1").hsdrop.value_counts()
smoking_data.describe()
smoking_data.info()
smoking_data.isna().value_counts()

# Look at the different class distributions in the base dataset
# Genders in %
(smoking_data.female.value_counts() / smoking_data.shape[0]) * 100
# Gender representation seems to be quite equal overall
plt.bar(x=["female", "male"], height=smoking_data.female.value_counts().to_list(), color="#9b0a7d")
plt.title(f"Gender representation in the overall dataset, N: {smoking_data.shape[0]}", fontsize=14)
plt.ylabel("Absolute Frequency")
plt.show()
# Gender distribution within the classes of smokers and non-smokers
plt.bar(x=["female", "male"], height=[smoking_data.query("smoker==0 and female==1").shape[0],
                                      smoking_data.query("smoker==0 and female==0").shape[0]],
                                    label="Non-smoker", color="grey")
plt.bar(x=["female", "male"], height=[smoking_data.query("smoker==1 and female==1").shape[0],
                                      smoking_data.query("smoker==1 and female==0").shape[0]],
                                    label="Smoker", color="#9b0a7d")
plt.title(f"Gender representation in the cond. dataset", fontsize=14)
plt.ylabel("Absolute Frequency")
plt.legend()
plt.show()

# Age distribution
(smoking_data.age.value_counts() / smoking_data.shape[0]) * 100

# Class distribution of age
plt.hist(smoking_data.age, bins=30, density=True, color="#9b0a7d")
plt.vlines(x=round(np.mean(smoking_data.age)), ymin=0, ymax= 0.050, color="black", linewidth=1)
plt.title("Histogram of Age")
plt.ylabel("Density")
plt.xlabel("Age Range")
plt.show()

# Looking at the educational variables
educ_df = pd.DataFrame()
for educ_var in smoking_data.columns[3:7]:
    educ_df[educ_var] = smoking_data[educ_var].value_counts()

# Class distribution of educational variables
plt.bar(x=smoking_data.columns[3:7], height=educ_df.loc[0], color="grey", label="0: No dropout")
plt.bar(x=smoking_data.columns[3:7], height=educ_df.loc[1], color="#9b0a7d", label="1: Dropout")
plt.title(f"Educational Variables: Class Distribution of: {smoking_data.columns[3:7].to_list()}", fontsize=10)
plt.ylabel("Absolute Frequency")
plt.legend()
plt.show()

# Masters degree holders have 0 row for every educational variable
# Analyze the share here
plt.hist(smoking_data.query("hsdrop==0 and hsgrad==0 and colsome==0 and colgrad==0").age, bins=30, density=True, color="#9b0a7d")
plt.vlines(x=round(np.mean(smoking_data.age)), ymin=0, ymax= 0.050, color="black", linewidth=1)
plt.title("Histogram of Age of Masters degree holders")
plt.ylabel("Density")
plt.xlabel("Age Range")
plt.show()
# The share of smokers and non smokers among them
plt.bar(x=["Non-Smoker", "Smoker"], height=smoking_data.query("hsdrop==0 and hsgrad==0 and colsome==0 and colgrad==0").smoker.value_counts().to_list(), color="#9b0a7d")
plt.title(f"Share of Smokers and Non Smokers among \n highest educational class, N: {smoking_data.query('hsdrop==0 and hsgrad==0 and colsome==0 and colgrad==0').shape[0]}", fontsize=14)
plt.ylabel("Absolute Frequency")
plt.show()

# b)
# Estimate the regression using the OLS method with heteroscedasticity
# robust standard errors. Interpret βˆ1. Is it statistically significant?


# Estimate the model
# Add the age^2 variable
smoking_data.insert(3, "age_sqrt", smoking_data.age.apply(lambda x: np.power(x, 2)))
# use heteroscedasticity robust standard errors and add constant intercept
smoking_data.insert(1, "constant_", 1)
lin_reg = sm.regression.linear_model.OLS(endog=smoking_data["smoker"], exog=smoking_data[smoking_data.columns[1:]])
lin_reg_fit = lin_reg.fit(cov_type="HC2")
lin_reg_fit.summary()

# Interpret beta_hat_1
# beta_hat_1 captures the effect of an existing smoking ban at the work place
# Remember this variable was binary
lin_reg_fit.params["smkban"]
# -0.04805196345001899
# H0: measures at the work place banning smoking have no effect on becoming a smoker, βˆ1 is equal to 0
# H1: measures indeed have an effect, βˆ1 != 0

# - If we look purely on the p-value of 0.000 we would argue that its statistically
# significant, since its below the significance level of 0.05 and even 0.01. So we
# have statistical evidence to reject our H0 (less smoking with smoking ban).
# - On the other hand the coefficient is -0.048052, so smoking measures in place
# at the work area decrease the predicted probability of being a smoker by 4.8052 %.
# In absolute terms this is a quiet small magnitude. Rendering the effect strength of the measures taken questionable.

# --> Statistically relevant BUT practical importance will be dependent of the study and scale of variable involved.

plt.plot(smoking_data.index, smoking_data.smoker, "o")
plt.plot(smoking_data.index, lin_reg_fit.fittedvalues, "ro")
plt.title("True values vs. Model fitted values", fontsize=12)
plt.xlabel(f"Data Point index, N: {smoking_data.shape[0]}")
plt.ylabel("True value vs. fit")
plt.show()
# Is beta_hat_1 statistically significant?
# Since its p-value is <0.05 we can argue that it is statistically significant
hypothesis_0 = "smkban = 0"
t_test_smkban = lin_reg_fit.t_test(hypothesis_0, use_t=True)
# p-value: 0.000
# Since the p-value is < 0.05 and even <0.01 we have to reject the H0 Hypothesis
# that the estimated beta parameter for the variable smkban is 0 or insignificant
z_score = ((0 - lin_reg_fit.params["smkban"]) / t_test_smkban.summary_frame()["std err"])
st.norm.pdf(z_score)
# Compare t value with critical value for test decision
abs(t_test_smkban.summary_frame()["t"])

# c)
# Does the probability of smoking increase or decrease with the level of education?
# Observe the educational related variables: hsdrop, hsgrad, colsome, colgrad,
lin_reg_fit.summary()
# Observing the educational related variables inlcuded in the model, one can see
# the following trend: 
# the higher the education of an individual the less likely or the smaller the chance
# that this individual is a smoker, but all estimated parameters are positive
# as high school drop outs have an effect of 0.2821909, high school grads: 0.2245546
# attended a college: 0.1556154 and finally college grads are least likely ending up
# as a smoker with: 0.0433244
# where the effect or absolute difference is biggest between attended college and
# the college grads, so finishing college is advised

# --> Model implies higher probability for hsfropouts than for graduates
abs(lin_reg_fit.params.diff())

# H0 Hypothesis: hsdrop and hsgrad have the same effect on smoker: hsdrop = hsgrad
# test if hsdrop and hsgrad have the same effect for the smoking probability
# Wald test
hypothesis_0 = ' = '.join(["hsdrop", "hsgrad"])
wald_test = lin_reg_fit.wald_test(hypothesis_0)
wald_test.statistic
# chi2 statistic: 10.2478391
# p-value: 0.0013684480785456001
# - H0: hsdrop = hsgrad (they are equal)
# - H1: hsdrop, hsgrad have different effects

# - Result: We reject the H0 --> F = 10.25 and p-value 0.0014 (less than 0.05 and even 0.01)
# - So both variables are statistically significantly different from each other


# d)
# Predict with fitted model and interprete
smkban_pred = 1
age_pred = 70
age_sqrt_pred = np.power(age_pred, 2)
hsdrop_pred, hsgrad_pred, colsome_pred = 0, 0, 0
colgrad_pred = 1
female_pred = 1
# Keep the correct order
smoking_data[smoking_data.columns[1:]]

x_pred = [1, smkban_pred, age_pred, age_sqrt_pred, hsdrop_pred, hsgrad_pred, colsome_pred, colgrad_pred, female_pred]

y_pred = lin_reg_fit.predict(x_pred)
y_pred
# -0.01326874

# - smoking measures in place at the work area decrease the predicted probability
#   of being a smoker for this individual specified, by roughly 1.32 %

# Problems:
# - We only assume linear relationship of predictor and regressor
# - Especially the positvie coefficient of education may suggest a more complex relationship --> not     	considered here
# - Predicted probability isn't standardized -> Can yield values above 1 o below 0

# Linear probability model is not appropriate in this case
smoking_data.query("age >= 70")
# 103 rows out of 10000 records
smoking_data.query("age >= 70 and smoker == 1").shape
# 15 records ou of the >= 70s that smoke
# and only two are college grads
# So these given settings represent an extreme case where the trained linear model
# with its average effect might lead to insufficient accuracy 

##################################################################### Execise 2)
# a)
# average partial effects of smoking ban on probability of smoking using i) probit and ii) logit model
# i) Probit Model
probit_mod = sm.Probit(endog=smoking_data["smoker"], exog=smoking_data[smoking_data.columns[1:]])
probit_mod_fit = probit_mod.fit()
probit_mod_fit.summary()
# Average Partial Effect (APE)
probit_mod_ape = probit_mod_fit.get_margeff(at="mean", method="dydx", atexog="smkban")
# estimate: smkban: -0.0485
probit_mod_ape.summary()
# 'dydx' - change in `endog` for a change in `exog`
# mkban: -0.0485

# ii) Logit Model
logit_mod = sm.Logit(endog=smoking_data["smoker"], exog=smoking_data[smoking_data.columns[1:]])
logit_mod_fit = logit_mod.fit()
logit_mod_fit.summary()
# Average Partial Effect (APE)
logit_mod_ape = logit_mod_fit.get_margeff(at="mean", method="dydx", atexog="smkban")
# estimate: smkban: 0.2652
logit_mod_ape.summary()
# 'dydx' - change in `endog` for a change in `exog`
# mkban: -0.0464


# Effect statistically significant?

# Display the results side by side in a compariosn table
param = "smkban"
lin_reg_eff = lin_reg_fit.params[param]
logit_eff = logit_mod_ape.summary_frame().loc[param, "dy/dx"]
probit_eff = probit_mod_ape.summary_frame().loc[param, "dy/dx"]

comp_eff_df = pd.DataFrame(index=["Linear Regression", "Logit Model", "Probit Model"],
                           data=[lin_reg_eff, logit_eff, probit_eff],
                           columns=["Average Partiial Effects (APE)"])
print(comp_eff_df)
# H0: The APE is not relevant, it should be 0
# H1: The APE is relevant and has an effect, it shoud be !=0

# Result: Since the p-value is 0.000 what is <0.05 and even 0.01, we reject the H0
# 		that the APE is not relevant, i.e. being equal to 0

# Interpretation: the estimated average change in the probability of being a smoker is reduced by 4.81%
# b)
# Probit Model to calculate the effect for groups specified
# Estimate the model new for the groups or predict for the groups?

# i) male, 40 years old, college graduate.
# Set the testing variables
constant = 1
smkban_i = 0
age_i = 40
age_sqrt_i = age_i**2
hsdrop_i = 0
hsgrad_i = 0
colsome_i = 0
colgrad_i = 1
female_i = 0

# Baseline Situation with no smoking ban in place
probit_mod_fit.get_prediction(exog=[constant, smkban_i, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).summary_frame()
base_i = probit_mod_fit.get_prediction(exog=[constant, smkban_i, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).predicted
# 0.18671

# Situation when a smoking ban is introduced
probit_mod_fit.get_prediction(exog=[constant, 1, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).summary_frame()
ban_i = probit_mod_fit.get_prediction(exog=[constant, 1, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).predicted
# 0.146843

# Effect strenth as difference
base_i - ban_i
# 0.03986644


# ii) female, 20 years old, high school dropout.
constant = 1
smkban_ii = 0
age_ii = 20
age_sqrt_ii = age_ii**2
hsdrop_ii = 1
hsgrad_ii = 0
colsome_ii = 0
colgrad_ii = 0
female_ii = 1

# Baseline Situation with no smoking ban in place
probit_mod_fit.get_prediction(exog=[constant, smkban_ii, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).summary_frame()
base_ii = probit_mod_fit.get_prediction(exog=[constant, smkban_ii, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).predicted
# 0.361454

# Situation when a smoking ban is introduced
probit_mod_fit.get_prediction(exog=[constant, 1, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).summary_frame()
ban_ii = probit_mod_fit.get_prediction(exog=[constant, 1, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).predicted
# 0.303431

# Effect strenth as difference
base_ii - ban_ii
# 0.05802279

# For Male, 40, College Graduate
print(f"Predicted Probability of a male, 40 Years Old, College Graduate without Smoking Ban: {round(base_i[0], 3)}")
print(f"Predicted Probability of a male, 40 Years Old, College Graduate with Smoking Ban: {round(ban_i[0], 3)}")
print(f"Effect of Smoking Ban on Smoking Probability for Males: {round(base_i[0] - ban_i[0], 3)}")

# Explanation:
# Among 40-year-old males who are college graduates, the probability of smoking in the workplace
# without any measures taken is 18.67%. However, if a smoking ban is implemented, this probability decreases to
# 14.68%. Consequently, the smoking ban results in a reduction of smoking probability by 3.9% within this specific demographic group.


# For Female, 20, High School Dropout
print(f"Predicted Probability of a female, 20 Years Old, High School Dropout without Smoking Ban: {round(base_ii[0], 3)}")
print(f"Predicted Probability of a female, 20 Years Old, High School Dropouts with Smoking Ban: {round(ban_ii[0], 3)}")
print(f"Effect of Smoking Ban on Smoking Probability for Females: {round(base_ii[0] - ban_ii[0], 3)}")

# Explanation:
# For 20-year-old females who are high school dropouts, the lprobability of smoking in the workplace without any
# measures taken is 36.14%. If a smoking ban is implemented, this probability decreases to 30.34%. Consequently,
# the smoking ban results in a reduction of smoking probability by 5.8% within this specific demographic group.


# Final Comparison
base_i_df = probit_mod_fit.get_prediction(exog=[constant, smkban_i, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).summary_frame()
ban_i_df = probit_mod_fit.get_prediction(exog=[constant, 1, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]).summary_frame()
i_df = pd.concat([base_i_df, ban_i_df]).reset_index(drop=True)
i_df["Effect_strength"] = i_df.predicted.diff()
print(i_df)

base_ii_df = probit_mod_fit.get_prediction(exog=[constant, smkban_ii, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).summary_frame()
ban_ii_df = probit_mod_fit.get_prediction(exog=[constant, 1, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]).summary_frame()

ii_df = pd.concat([base_ii_df, ban_ii_df]).reset_index(drop=True)
ii_df["Effect_strength"] = ii_df.predicted.diff()
print(ii_df)
