from utils import *
from utils.utils import *
from utils.models import *

import lightgbm as lgbm
from sklearn.metrics import r2_score, mean_absolute_error

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

def train_and_predict(PATH,n_round,with_preprosessing=False):
    pre = "whit" if with_preprosessing else "without"
    print(f"Working with round: {n_round}")
    train_data, tournament_data, features = load_proses_data(n_round, preprocessing=with_preprosessing)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    ids = tournament_data['id']

    kf = KFoldEra(n_splits=4, shuffle=False, seed=123)
    folds = kf.split(train_data)

    X_train = train_data[features]
    y_train = train_data[TARGET_NAME]
    X_test = validation_data[features]
    y_test = validation_data[TARGET_NAME]

    X_tournament = tournament_data[features]

    y_train_pred = np.zeros(X_train.shape[0])

    y_pred_val = []
    y_pred_tournament = []

    params = {
        'bagginf_freq':5,
        'bagging_fraction':0.1,
        'boost_from_average':'false',
        'colsample_bytree':0.12,
        'boost':'gbdt',
        'learning_rate': 0.03,
        'min_sum_hessian_in_leaf':0.1,
        'num_leaves': 9,
        'num_threads':4,
        'metric':'mae',
        'feature_fraction': 0.1,
        'max_depth': -1,
        'objective': 'rmse'}
    
    for k, (tr_idx, vl_idx) in enumerate(folds):
    
        print(f'Fold {k}')
        x_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_vl, y_vl = X_train.iloc[vl_idx], y_train.iloc[vl_idx]
        
        d_train = lgbm.Dataset(x_tr, label = y_tr)
        d_valid = lgbm.Dataset(x_vl, label = y_vl)
        
        model = lgbm.train(params, d_train, 100, valid_sets = [d_train, d_valid])

        y_hat = model.predict(x_vl)
        y_hat_val = model.predict(X_test)
        y_hat_tournament = model.predict(X_tournament)

        y_train_pred[vl_idx] += model.predict(x_vl)
        print(f"Correlation train of fold: {correlation_score(y_vl, y_hat)}")
        print(f"Correlation valid of fold: {correlation_score(y_test, y_hat_val)}")

        y_pred_val.append(y_hat_val)
        y_pred_tournament.append(y_hat_tournament)
    
    correlation_train_score = correlation_score(y_train, y_train_pred)
    correlation_valid_score = correlation_score(y_test, np.average(y_pred_val, axis=0))
    print(f"Correlation score of train data: {correlation_train_score}")
    print(f"Correlation score of validation data: {correlation_valid_score}")

    me, con = check_correlation_consistency_lgbm(model,train_data,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'consistency of train data ({pre}_preprosessing): {con}',margin=[0.02, 0.002], save=False, path=PATH)

    me, con = check_correlation_consistency_lgbm(model,validation_data,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'consistency of valid data ({pre}_preprosessing): {con}',margin=[0.02, 0.002], save=True, path=PATH)

    ##TOURNAMENT FILE
    tournament_data[PREDICTION_NAME] = np.average(y_pred_tournament, axis=0)

    results_df = pd.DataFrame(data={PREDICTION_NAME:tournament_data[PREDICTION_NAME]})
    joined = pd.DataFrame(ids).join(results_df)
    print(f"# Writing predictions to {TOURNAMENT_NAME}_submissions.csv...")
    joined.to_csv(f"{PATH}/{TOURNAMENT_NAME}_submission.csv", index=False)

    return 'Done!'

@click.command()
@click.option("--n_round")
@click.option("--preprosessing")
def main(n_round, preprosessing):
    init = time.time()
    PATH = f'../../submission/round {n_round}/LGBM'
    os.makedirs(exist_ok=True, name=PATH)

    if preprosessing.lower() == "true": preprosessing=True 
    else: preprosessing=False
    result = train_and_predict(PATH,n_round,preprosessing)
    final = time.time()
    print(result+ f': {final-init}')


if __name__ == "__main__":
    main()