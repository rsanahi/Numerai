from utils import *
from utils.utils import *
from utils.models import *

import lightgbm as lgbm
from sklearn.metrics import r2_score, mean_absolute_error

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

def train_and_predict(PATH,n_round):
    print(f"Working with round: {n_round}")
    raw_train, tournament_data, features = load_data(n_round)
    ids = tournament_data['id']

    valid_raw = tournament_data[tournament_data.data_type == "validation"]

    d_train = lgbm.Dataset(raw_train[features], label = raw_train[TARGET_NAME])
    d_valid = lgbm.Dataset(valid_raw[features], label = valid_raw[TARGET_NAME])

    params = {
        'bagginf_freq':5,
        'bagging_fraction':0.1,
        'boost_from_average':'false',
        'colsample_bytree':0.12,
        'boost':'gbdt',
        'learning_rate': 0.05,
        'min_sum_hessian_in_leaf':1.0,
        'num_leaves': 9,
        'num_threads':4,
        'metric':'mae',
        'feature_fraction': 0.1,
        'max_depth': -1,
        'objective': 'rmse'}
    
    clf = lgbm.train(params, d_train, 100, valid_sets = [d_train, d_valid])

    yhat_pred = clf.predict(raw_train[features])
    train_correlation = correlation_score(raw_train[TARGET_NAME], yhat_pred)
    me, con = check_correlation_consistency(clf,raw_train,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'Train consistency: {con}, correlation: {train_correlation}',margin=[0.02, 0.002],save=True,path=PATH)

    yhat_pred = clf.predict(valid_raw[features])
    valid_correlation = correlation_score(valid_raw[TARGET_NAME], yhat_pred)
    me, con = check_correlation_consistency(clf,valid_raw,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'Validation consistency: {con}, correlation: {valid_correlation}',margin=[0.02, 0.002],save=True,path=PATH)

    tournament_data[PREDICTION_NAME] = clf.predict(tournament_data[features])

    # Check the per-era correlations on the validation set

    results_df = pd.DataFrame(data={PREDICTION_NAME:tournament_data[PREDICTION_NAME]})
    joined = pd.DataFrame(ids).join(results_df)
    print(f"# Writing predictions to {TOURNAMENT_NAME}_submissions.csv...")
    joined.to_csv(f"{PATH}/{TOURNAMENT_NAME}_submission.csv", index=False)

    return 'Done!'

@click.command()
@click.argument("n_round")
def main(n_round):
    init = time.time()
    PATH = f'../../submission/round {n_round}/LGBM'
    os.makedirs(exist_ok=True, name=PATH)
    result = train_and_predict(PATH,n_round)
    final = time.time()
    print(result+ f': {final-init}')


if __name__ == "__main__":
    main()