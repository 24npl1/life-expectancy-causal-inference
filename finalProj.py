from cmath import nan
from statistics import median
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats.mstats import winsorize
from fcit import fcit


#read in our data into a data frame
df2 = pd.read_csv("finalDataToWins.csv")

#rename our columns for readability
orig_cols = list(df2.columns)
new_cols = []
for col in orig_cols:
    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').replace('-', '_').lower())
df2.columns = new_cols

#clean the data such that we drop any na data, or data where alpha < 0.05
mort_5_percentile = np.percentile(df2.adult_mortality.dropna(), 5)
df2.adult_mortality = df2.apply(lambda x: np.nan if x.adult_mortality < mort_5_percentile else x.adult_mortality, axis=1)
df2.infant_deaths = df2.infant_deaths.replace(0, np.nan)
#clean the bmi data for values which do not make realistic sense
df2.bmi = df2.apply(lambda x: np.nan if (x.bmi < 10 or x.bmi > 50) else x.bmi, axis=1)
#replace na with 0
df2.under_five_deaths = df2.under_five_deaths.replace(0, np.nan)
#drop the bmi column, as it does not make any sense
df2.drop(columns='bmi', inplace=True)

imputed_data = []

#clean the year column
for year in list(df2.year.unique()):
    year_data = df2[df2.year == year].copy()
    for col in list(year_data.columns)[3:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().mean()).copy()
    imputed_data.append(year_data)

df2 = pd.concat(imputed_data).copy()

#complete winsorization tests on the data
def test_wins(col, lower_limit=0, upper_limit=0):
    wins_data = winsorize(df2[col], limits=(lower_limit, upper_limit))
    wins_dict[col] = wins_data

wins_dict = {}
cont_vars = list(df2.columns)[3:]

#complete winsorizations for each column, where the limits reflects the range at which winsorization occurs
test_wins(cont_vars[0], lower_limit=.01)
test_wins(cont_vars[1], upper_limit=.04)
test_wins(cont_vars[2], upper_limit=.05)
test_wins(cont_vars[3], upper_limit=.0025)
test_wins(cont_vars[4], upper_limit=.135)
test_wins(cont_vars[5], lower_limit=.1)
test_wins(cont_vars[6], upper_limit=.19)
test_wins(cont_vars[7], upper_limit=.05)
test_wins(cont_vars[8], lower_limit=.1)
test_wins(cont_vars[9], upper_limit=.02)
test_wins(cont_vars[10], lower_limit=.105)
test_wins(cont_vars[11], upper_limit=.185)
test_wins(cont_vars[12], upper_limit=.105)
test_wins(cont_vars[13], upper_limit=.07)
test_wins(cont_vars[14], upper_limit=.035)
test_wins(cont_vars[15], upper_limit=.035)
test_wins(cont_vars[16], lower_limit=.05)
test_wins(cont_vars[17], lower_limit=.025, upper_limit=.005)


wins_df = df2.iloc[:, 0:3]
cont_vars = list(df2.columns)[3:]


for col in cont_vars:
    wins_df[col] = wins_dict[col]

#remove any and all columns which we not not need for analysis 
del(wins_df["country"])
del(wins_df["status"])
del(wins_df["year"])
del(wins_df["population"])
del(wins_df["hepatitis_b"])
del(wins_df["measles"])
del(wins_df["hiv/aids"])
del(wins_df["thinness_1_19_years"])
del(wins_df["thinness_5_9_years"])
del(wins_df["polio"])
del(wins_df["diphtheria"])
del(wins_df["infant_deaths"])

#finnaly, export our data to a csv for further manipulation
wins_df.to_csv('winsorizedFinalData1.csv')



def backdoor_mean(Y, A, Z, value, data):
    """
    Compute the counterfactual mean E[Y(a)] for a given value of a via backdoor adjustment
    
    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names to adjust for
    value: float corresponding value to set A to
    data: pandas dataframe
    
    Return
    ------
    ACE: float corresponding to the causal effect
    """
    
    formula = Y + "~" + A
    if len(Z) > 0:
        formula += " + " + "+".join(Z)
    
    model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    data_a = data.copy()
    data_a[A] = value
    return np.mean(model.predict(data_a))

def aipw(Y, A, Z, data, trim=False):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via AIPW

    Inputs
    ------
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    Z: list of variable names included the backdoor adjustment set
    data: pandas dataframe
    trim: boolean determining whether to trim the propensity scores or not

    Return
    ------
    ACE: float corresponding to the causal effect
    """
    #Build our formula
    if len(Z) != 0:
        f_A = str(A) + "~" + "+".join(Z)
        f_Y = str(Y) + "~" + str(A) + "+" + "+".join(Z)

        #Binary treatment, thus we use linear regression
        reg_ipw = sm.GLM.from_formula(formula=f_A, data=data, family=sm.families.Binomial()).fit()
    
    else:
        reg_ipw = sm.GLM.from_formula(formula= A + "~ 1", data=data, family=sm.families.Binomial()).fit()
        f_Y = str(Y) + "~" + str(A)

    #use our model to predict our data
    propensity = reg_ipw.predict(data)

    #we add the propensity data to our original data set 
    data["propensity"] = propensity

    #we trim if need be
    if trim:
        data = data[(data["propensity"] < 0.9) & (data["propensity"] > 0.1)]


    #create new data set with replacement of the treatment
    data_1 = data.copy()
    data_1[A] = 1

    data_0 = data.copy()
    data_0[A] = 0

    
    #we use continous regression on each of our data sets and create point estimates
    reg_y =  sm.GLM.from_formula(formula=f_Y, data=data, family=sm.families.Gaussian()).fit()

    #get our backdoor ACE
    y_predict_1 = reg_y.predict(data_1)
    y_predict_0 = reg_y.predict(data_0)
    
    dataPrime = 1 - data[A]
    propPrime = 1 - propensity

    #now we simply calcualte the IPW adjusted ACE
    ACE = np.mean((data[A] / propensity) * (data[Y] - y_predict_1) + (y_predict_1)) - np.mean((dataPrime / propPrime) * (data[Y] - y_predict_0) + (y_predict_0))             
    return ACE


def compute_confidence_intervals(Y, A, Z, data, method_name, num_bootstraps=10, alpha=0.05, value=None):
    """
    Compute confidence intervals for backdoor adjustment via bootstrap
    
    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    
    for i in range(num_bootstraps):
        
        # resample the data with replacement
        data_sampled = data.sample(len(data), replace=True)
        data_sampled.reset_index(drop=True, inplace=True)
        
        # add estimate from resampled data
        if method_name == "aipw":
            estimates.append(aipw(Y, A, Z, data_sampled))
            
        elif method_name == "backdoor_mean":
            estimates.append(backdoor_mean(Y, A, Z, value, data_sampled))

        else:
            print("Invalid method")
            estimates.append(1)

    # calculate the quantiles
    quantiles = np.quantile(estimates, q=[Ql, Qu])
    q_low = quantiles[0]
    q_up = quantiles[1]
    
    return q_low, q_up

def get_numpy_matrix(data, variables):
    """
    Takes a pandas dataframe and a list of variable names, and returns
    just the raw matrix for those specific variables
    """
    
    
    matrix = data[variables].to_numpy()

    # if there's only one variable, ensure we return a matrix with one column
    # rather than just a column vector
    if len(variables) == 1:
        return matrix.reshape(len(data),)
    return matrix
    
def testEdge(data, var1, var2):
    test = fcit.test(np.vstack(data[var1]), np.vstack(data[var2]))
    return "{} -> {}: {}".format(var1, var2, test)

def main():
    """
    Add code to the main function as needed to compute the desired quantities.
    """

    np.random.seed(100)

    data = wins_df
    
    #sensitivity testing
    # print(testEdge(data, "total_expenditure", "life_expectancy"))
    # print(testEdge(data, "percentage_expenditure", "total_expenditure"))
    # print(testEdge(data, "total_expenditure", "percentage_expenditure"))
    # print(testEdge(data, "percentage_expenditure", "gdp"))
    # print(testEdge(data, "gdp", "percentage_expenditure"))

    medianLife = np.median(data["life_expectancy"])
    print(min(data["total_expenditure"]))
    print(max(data["total_expenditure"]))

    #values over a range of all the values of that life expectancy takes in the data
    values = np.arange(0.37, 11.66, 0.01)

    point_estimates = []

    #create point estimates for the backdoor mean at each possibile value of life expectancy
    for value in values:
        point_estimates.append(backdoor_mean("life_expectancy", "total_expenditure", ["percentage_expenditure","gdp"], value, data))

    ci_lower = []
    ci_upper = []

    #create confidence intervals for the backdoor mean at each possibile value of life expectancy
    for value in values:
        ci = compute_confidence_intervals("life_expectancy", "total_expenditure", ["percentage_expenditure", "gdp"],
                                        data, method_name="backdoor_mean", value=value)
        ci_lower.append(ci[0])
        ci_upper.append(ci[1])

    #generate the plot from the point estimates and the confidence intervals we have caluclated
    point_estimates = np.array(point_estimates)
    ci_lower, ci_upper = np.array(ci_lower), np.array(ci_upper)
    fig, ax = plt.subplots()
    ax.plot(values, point_estimates, label = 'ACE Point Estimates')
    ax.fill_between(values, ci_lower, ci_upper, color='b', alpha=.1, label = "Confidence Intervals")
    #create a legend and labels for the plot
    ax.legend()
    plt.xlabel("Health Expenditure (% of Total Expenditure)")
    plt.ylabel("Predicted Life Expectancy (Years)")
    plt.title("ACE via Backdoor Mean with Confidence Intervals")

    #save the our plot
    fig.savefig("lifeExpectPlot3.pdf")
    
    #Get the ACE estimate from the slope of the point estimates 
    #slope, intercept = np.polyfit(np.log(point_estimates), np.log(values), 1)
    slope = (max(point_estimates) - min(point_estimates)) / (max(values) - min(values))

    print("Our estimated causual effect is " + str(slope) + " years")






if __name__ == "__main__":
    main()
