from utils import *
from utils.utils import *
from utils.models import *



def make_preprosessing(PATH,n_round):
    print(f"Working with round: {n_round}")
    train_data, tournament_data, features = load_data(n_round)

    feature_groups = {
        g: [c for c in train_data if c.startswith(f"feature_{g}")]
        for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
        }

    print("Train data pca preprosessing")
    pca_data = PCA_preprosessing(train_data, feature_groups, features)

    print("Train data Autoencoder feature reduction")
    encoded_train = autoencoder_(train_data[features],input_size=310, hidden_size=64, code_size=50)
    encoded_train = encoded_train.add_prefix('feature_')

    df = pd.concat([encoded_train,pca_data], axis=1)
    features_preprosessing = df.columns

    df = pd.DataFrame(save_memo(df,features_preprosessing))

    print("Saving train data with preprosessing")
    df.to_csv('numerai_train_data.csv')

    gc.collect()

    print("Tournament data pca preprosessing")
    pca_data = PCA_preprosessing(tournament_data, feature_groups, features)

    print("Tournament data Autoencoder feature reduction")
    encoded_tournament = autoencoder_(tournament_data[features],input_size=310, hidden_size=64, code_size=50)
    encoded_tournament = encoded_tournament.add_prefix('feature_')

    tournament = pd.concat([encoded_tournament,pca_data], axis=1)

    df = pd.DataFrame(save_memo(tournament,features_preprosessing))

    print("Saving tournament data with preprosessing")
    df.to_csv('numerai_tournament_data.csv')

    return "Done!"


@click.command()
@click.argument("n_round")
def main(n_round):
    init = time.time()
    PATH = f'../../submission/round {n_round}/autoencoder'
    os.makedirs(exist_ok=True, name=PATH)
    result = make_preprosessing(PATH,n_round)
    final = time.time()
    print(result+ f': {final-init}')


if __name__ == "__main__":
    main()