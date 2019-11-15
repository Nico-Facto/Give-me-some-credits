import pandas


def traitementDF(importfile,rev,nod,dr,age,noo,nre,n3059,n6089,n90):

    df_old = importfile
    df_old = df_old[0:0]#clean du DF

    input_data = { "RevolvingUtilizationOfUnsecuredLines": rev, "NumberOfDependents": nod, "DebtRatio": dr, "age": age, "NumberOfOpenCreditLinesAndLoans": noo,
     "NumberRealEstateLoansOrLines": nre, "NumberOfTime30-59DaysPastDueNotWorse": n3059, "NumberOfTime60-89DaysPastDueNotWorse": n6089, 
     "NumberOfTimes90DaysLate": n90 }

    df = df_old.append(input_data,ignore_index=True,sort=False)

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

