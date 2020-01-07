import pandas
from joblib import load
import xgboost

class iaForCredits() :

    def __init__(self,DataFrame):
        """ Import your csv with new inputs, make a prediction by Ia and create .csv on the root of folder """

        self.importID = DataFrame['id']
        self.importfile = DataFrame
    
        df = self.importfile

        #Pre_Processing of the DataFrame
        df['IncomePerPerson'] = df['MonthlyIncome'] / ( df['NumberOfDependents'] + 1 )
        df.loc[df.age > 80, 'isOld'] = '1' 
        df.loc[df.age <= 80, 'isOld'] = '0'
        df['MonthlyDebt'] = df['MonthlyIncome'] * df['DebtRatio']
        df['MonthlyBalance'] = df['MonthlyIncome'] - df['MonthlyDebt']
        df['DebtPerPerson'] = df['MonthlyDebt'] / ( df['NumberOfDependents'] + 1 )
        df['BalancePerPerson'] = df['MonthlyBalance'] / ( df['NumberOfDependents'] + 1 )
        df['NumberOfTime30-89DaysPastDueNotWorse'] = df['NumberOfTime30-59DaysPastDueNotWorse'] + df['NumberOfTime60-89DaysPastDueNotWorse']
        df['NumbersOfOpen-NumberRealEstate'] = df['NumberOfOpenCreditLinesAndLoans'] - df['NumberRealEstateLoansOrLines']
        df = df.fillna(0)

        #load model ia and make prediction
        model = load('xgb_model_ia.joblib')
        y_prod_proba = model.predict(df)
        self.y_prod_scores = y_prod_proba[:,1]
        
        #format and create the output file
        xId = pandas.DataFrame(self.y_prod_scores)
        xId['Id'] = self.importID
        xId = xId[['Id',0]]
        xId.columns = ['Id','Probability']
        xId.to_csv('forkagg.csv', index=False)

        #just print when it's done
        print("Prediction is done !! ")        
