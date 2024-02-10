# Econometric Methods Home Assignment WiSe 23/24
"""
Group Members:
1. stu, stuXXXX
2. stu, stuXXXX
3. stu, stuXXXX
4. stu, stuXXXX
5. stu, stuXXXX
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
# The estimated parameter of the smkban has the desired sign and
# is about: -0.04805196345001899 what would mean that if a smoking ban exists at the work
# place (smkban=1) it reduces the likelihood that the individual smokes since
# the parameters reduces the result by its sign, its magnitude is rather small
# being roughly -0.05 what could indicate the strength of the taken measures at
# the work place
# Since we are working here with a linear probability model, we can state that
# beta_hat_1 can be interpreted as the change in the probability that the
# dependent variable smoker is 1, holding all other factors constant
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
# Does the probability of smoking increase or decre- ase with the level of education?
# Observe the educational related variables: hsdrop, hsgrad, colsome, colgrad,
lin_reg_fit.summary()
# Observing the educational related variables inlcuded in the model one can see
# the following trend: 
# the higher the education of an individual the less likely or the smaller the chance
# that this individual is a smoker, but all estimated parameters are positive
# as high school drop outs have an effect of 0.2822, high school grads: 0.2246
# attended a college: 0.1556 and finally college grads are least likely ending up
# as a smoker with: 0.0433
abs(lin_reg_fit.params.diff())
# where the effect or absolute difference is biggest between attended college and
# the college grads, so finishing college is advised

# Perform Wald test for joint parameter test
# test if hsdrop and hsgrad have the same effect for the smoking probability
hypothesis_0 = ' = '.join(["hsdrop", "hsgrad"])
wald_test = lin_reg_fit.wald_test(hypothesis_0)
wald_test.statistic
# chi2 statistic: 10.2478391
# p-value: 0.0013684480785456001
# Compare with critical value for test decision


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

# Estimate seems to be very low, so the bespoke individual seems to be very 
# unlikely to be a smoker and contradictingly has a negative sign

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
# b)
# Probit Model to calculate the effect for groups specified
# Estimate the model new for the groups or predict for the groups?

# i) male, 40 years old, college graduate.
exog_probit_i = smoking_data.query("female == 0 and age == 40 and colgrad == 1").copy().reset_index(drop=True)
probit_mod_i = sm.Probit(endog=exog_probit_i["smoker"], exog=exog_probit_i[exog_probit_i.columns[1:]])
probit_mod_i_fit = probit_mod_i.fit()
probit_mod_i_fit.summary()
# Average Partial Effect (APE)
probit_mod_i_ape = probit_mod_i_fit.get_margeff(at="mean", method="dydx", atexog="smkban")
# estimate: smkban: -0.1600
probit_mod_i_ape.summary()
# -0.0485


# ii) female, 20 years old, high school dropout.
exog_probit_ii = smoking_data.query("female == 1 and age == 20 and hsdrop == 1").copy().reset_index(drop=True)
probit_mod_ii = sm.Probit(endog=exog_probit_ii["smoker"], exog=exog_probit_ii[exog_probit_ii.columns[1:]])
probit_mod_ii_fit = probit_mod_ii.fit()
# numpy.linalg.LinAlgError: Singular matrix
probit_mod_ii_fit.summary()
# Average Partial Effect (APE)
probit_mod_ii_ape = probit_mod_ii_fit.get_margeff(at="mean", method="dydx", atexog="smkban")
# estimate: smkban:
probit_mod_ii_ape.summary()


# Alternative Approach for b)
# Use the already fitted Probit Model and just predict for the set variables
# i) male, 40 years old, college graduate.
smkban_i = 1
age_i = 40
age_sqrt_i = np.power(age_i, 2)
hsdrop_i, hsgrad_i, colsome_i = 0, 0, 0
colgrad_i = 1
female_i = 0
# Keep the correct order
smoking_data[smoking_data.columns[1:]]

x_i = [1, smkban_i, age_i, age_sqrt_i, hsdrop_i, hsgrad_i, colsome_i, colgrad_i, female_i]

probit_mod_fit.predict(x_i)
# 0.14684307


# ii) female, 20 years old, high school dropout.
smkban_ii = 1
age_ii = 20
age_sqrt_ii = np.power(age_ii, 2)
hsdrop_ii, hsgrad_ii, colsome_ii = 1, 0, 0
colgrad_ii = 0
female_ii = 0

x_ii = [1, smkban_ii, age_ii, age_sqrt_ii, hsdrop_ii, hsgrad_ii, colsome_ii, colgrad_ii, female_ii]

probit_mod_fit.predict(x_ii)
# 0.34125012