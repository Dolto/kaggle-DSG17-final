import pandas as pd
import fbprophet

train = pd.read_csv('data/train_resampled.csv', sep=';')
test = pd.read_csv('data/test.csv', sep=';')

def get_sample_train(material):
    sample_train = train[(train['Material'] == material)][['Month', 'OrderQty']]
    sample_train.rename(columns={'Month': 'ds', 'OrderQty': 'y'}, inplace=True)
    sample_train.ds = pd.to_datetime(sample_train.ds)
    return sample_train

def get_sample_test(material):
    sample_test = test[(test['Material'] == material)][['date']].reset_index(drop=True)
    sample_test.rename(columns={'date': 'ds'}, inplace=True)
    sample_test.ds = pd.to_datetime(sample_test.ds)
    return sample_test

y_pred_m1 = []
y_pred_m2 = []
y_pred_m3 = []
just_one_row = 0

groups = test.groupby(['Material'])
for i, ((material), _) in enumerate(groups):
    print('%s/%s - [%s]' % (i,len(groups), material))
    # Filter data
    sample_train = get_sample_train(material)

    if len(sample_train.index) <= 1:
        just_one_row += 1
        print('--- just one line', just_one_row)
        y_pred_m1.append(0)
        y_pred_m2.append(0)
        y_pred_m3.append(0)
    else:
        try:
            # Create and fit a Prophet model
            model = fbprophet.Prophet(weekly_seasonality=False)
            model.fit(sample_train)
            # Get data to predict
            sample_test = get_sample_test(material)
            # Predict new values
            preds = model.predict(sample_test)
            # Appends predictions for each month
            y_pred_m1.append(preds.yhat[0])
            y_pred_m2.append(preds.yhat[1])
            y_pred_m3.append(preds.yhat[2])
        except:
            y_pred_m1.append(0)
            y_pred_m2.append(0)
            y_pred_m3.append(0)

y_pred = y_pred_m1 + y_pred_m2 + y_pred_m3

submission = pd.DataFrame(data={
    'id': range(len(y_pred)),
    'demand': y_pred
})
submission.to_csv('submission_prophet.csv', sep=';', index=False)
