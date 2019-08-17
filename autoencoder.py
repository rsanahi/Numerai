from utils.utils import *
from utils.models import *
import os, click

TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[TARGET_NAME],
	    df[PREDICTION_NAME].rank(pct=True, method="first")
        )[0,1]

def train_and_predict(PATH,n_round):
    print(f"Working with round: {n_round}")
    raw_train, tournament_data, features = load_data(n_round)
    train_, test_, te_eras, tr_eras = split_v2(raw_train, verbose=0)

    train_features,train_labels = train_[features],train_[TARGET_NAME]
    test_features,test_labels = test_[features], test_[TARGET_NAME]

    input_size = 310
    hidden_size = 128
    code_size = 32

    neuronal_network = nn_model()
    autoencoder = autoencoder_()
    history = autoencoder.fit(train_features, train_features, epochs=10, batch_size=1024, validation_data=(test_features, test_features))
    his = history.history
    plot_history(his,['loss', 'mean_absolute_error'],title=f'autoencoder {n_round}', path=PATH)

    autoencoder_train_predict = autoencoder.predict(train_features)
    history = neuronal_network.fit(autoencoder_train_predict, train_labels, epochs=10, batch_size=1024, validation_data=(test_features, test_labels))
    his = history.history
    plot_history(his,['loss', 'mean_absolute_error'],title=f'neuronal_network {n_round}', path=PATH)

    tournament_data[PREDICTION_NAME] = neuronal_network.predict(tournament_data[features])

    # Check the per-era correlations on the validation set
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(score)

    tournament_data[PREDICTION_NAME].to_csv(TOURNAMENT_NAME + "_submission.csv")

    return 'Done!'

@click.command()
@click.argument("n_round")
def main(n_round):
    PATH = f'../../submission/round {n_round}/autoencoder'
    os.makedirs(exist_ok=True, name=PATH)
    result = train_and_predict(PATH,n_round)
    print(result)


if __name__ == "__main__":
    main()