clear

cd "\\ukfwstat-s3\userhomes$\suwst148\Desktop\temporary\group1\PC2"
use CARD.dta

summarize

tab educ, summarize(wage)
scatter wage educ
* install package estout
* ssc install estout 

* a)  Estimate OLS
regress lwage educ exper expersq black south smsa reg661-reg668 smsa66, vce(robust)
eststo md1

* b) 
* === first-stage regression
regress educ nearc4 exper expersq black south smsa reg661-reg668 smsa66, vce(robust)
test nearc4
* educ and nearc4 has a practically and statistically significant partial correlation (pvalue = 0.000 < 5%)
* It is plausible, people living close to 4 year colleage in 1966 has higher education
* F-statistic = 14.14 > 10 -> nearc4 is strong instrument for educ



* c) estimate 2sls using nearc4 as instrument for education
ivregress 2sls lwage (educ = nearc4) exper expersq black south smsa reg661-reg668 smsa66, vce(robust)
eststo md2

esttab md1 md2, se
esttab md1 md2, se, using "results.tex" /* export the table */

* interpretation: one additional year of edu increase the income by 13%. It is statistically significant
* confidence interval: the 2SLS generate wider ci (higher variance ) compared to the OLS -> 2SLS is less efficient than the OLS. 

* d)  estimate 2sls using nearc4 and nearc2 as instruments
* ==== first-stage reg 
regress educ nearc4 nearc2 exper expersq black south smsa reg661-reg668 smsa66, vce(robust)
test nearc4 nearc2
 
regress educ nearc2 exper expersq black south smsa reg661-reg668 smsa66, vce(robust)
test nearc2

*  the partial corr between nearc2 and educ is not statistically significant. The F-stat (8.32) is smaller than in task b (F-stat = 14.14). Better use only nearc4 as instrumnet for education.

* ==== estimate 2sls with multiple instruments
ivregress 2sls lwage (educ = nearc4 nearc2) exper expersq black south smsa reg661-reg668 smsa66, vce(robust)

* e) regress iq test on nearc4
reg iq nearc4, vce(robust)
* result: people living near 4-year colleage have a significant higher IQ score. 
* Possible explanation: People with higher IQ score like to live near colleage because they plan to attend colleage/university. 

* f) add regional dummies
reg iq nearc4 reg661-reg668 smsa66, vce(robust)
* Result: IQ and nearc4 now appear uncorrelate. However, some regional dummies are highly significant -> Some regions may attract more intelligent people than others.
* If we exclude these regional dummies varibales in the wage equation, this will lead to instrumnet endogeneity (the exogenous condition of instrument is violated).















