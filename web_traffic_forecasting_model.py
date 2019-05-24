
from datetime import datetime
import numpy as np
import pandas as pd
from fbprophet import Prophet


print('Script started at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Read the initial CSV files into dataframes
df_train = pd.read_csv('train_2.csv', index_col='Page', header=0)
df_key = pd.read_csv('key_2.csv', index_col='Page', header=0)


site_name = df_train.index
df_train = df_train.transpose()

df_train = df_train.fillna(method='ffill')
df_train = df_train.fillna(method='bfill')
df_train = np.log(df_train[df_train != 0])
df_train = df_train.fillna(0)



start_idx = 0

end_idx = 99


date_format = "%m/%d/%Y"
first_date = datetime.strptime('9/1/2017', date_format)
start_date = datetime.strptime('9/13/2017', date_format)
end_date = datetime.strptime('11/13/2017', date_format)
prediction_period = (end_date - first_date).days + 1
prediction_subperiod = (end_date - start_date).days + 1


index = pd.date_range(start=start_date, end=end_date, freq='D')
df_pred = pd.DataFrame()


for i in range(start_idx, end_idx):

    df = df_train.iloc[:,i]

    # Format input for Prophet model
    df = df.reset_index()
    df.columns.values[:] = ['ds', 'y']

    if np.sum(df['y']) > 0:

        # Include yearly seasonality
        model = Prophet(yearly_seasonality=True)
        model.fit(df)

        # Make daily forecasts until end of prediction period
        future = model.make_future_dataframe(periods=prediction_period)
        forecast = model.predict(future)

        forecast.index = forecast['ds']



        pred = pd.DataFrame(forecast['yhat'][forecast['ds'] >= start_date])

    else:

        pred = pd.DataFrame(np.zeros((prediction_subperiod,1)), index=index, columns=['yhat'])

    pred.rename(columns = {'yhat':site_name[i]}, inplace=True)

    df_pred = pd.concat([df_pred, pred], axis=1)


    print(i) if i % 10 == 0 else False



df_final = pd.DataFrame(df_pred.unstack())
df_final.rename(columns = {0:'Visits'}, inplace=True)
df_final.reset_index(inplace=True)
df_final['key'] = df_final['level_0'] + '_' + df_final['ds'].astype(str)


df_final = pd.merge(df_final, df_key, how='inner', left_on='key',
                    right_on=None, left_index=False, right_index=True)

# Override negative predictions with zeros
df_final['Visits'] = df_final['Visits'].apply(lambda x: 0 if x < 0 else x)

# Exponentiate the predicitions and then round up to zero decimal places
df_final['Visits'] = np.round(np.exp(df_final['Visits']), decimals=0)

# Create the submission file
df_submission = df_final.to_csv('submission_2.csv', index=False,
                                columns=['Id','Visits'])

end_run_time = datetime.now()

print('Script completed at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
