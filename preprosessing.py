from utils import *
from utils.utils import *
from utils.models import *



def make_preprosessing(PATH,n_round):
    print(f"Working with round: {n_round}")
    train_data, tournament_data, features = load_data(n_round)
    train_ids = train_data.id

    feature_groups = {
        g: [c for c in train_data if c.startswith(f"feature_{g}")]
        for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
        }
    train_data.to_feather(f'../../../raw_data/round {n_round}/train-tmp')
    tournament_data(f"../../../raw_data/round {n_round}/tournament-tmp")

    del tournament_data
    gc.collect()

    print("Train data pca preprosessing")
    pca_data, pca = PCA_preprosessing(train_data, feature_groups, features)

    print("Train data Autoencoder feature reduction")
    input_size = 310
    code_size = 50

    input_data = Input(shape=(310,))
    encoded1 = Dense(310,input_shape=(310,))(input_data)
    encoded2 = Activation('relu')(encoded1)
    encoded3 = BatchNormalization()(encoded2)
    code = Dense(code_size)(encoded3)
    decoded1 = Dense(310)(code)
    decoded2= Activation('relu')(decoded1)
    decoded3 =BatchNormalization()(decoded2)
    decoded4 = Dense(310, activation='sigmoid')(decoded3)

    autoencoder = Model(inputs=input_data, outputs=decoded4)
    autoencoder.compile(optimizer = RMSprop(), loss = 'binary_crossentropy', metrics=['mae'])
    history = autoencoder.fit(train_data[features], train_data[features], epochs=10, batch_size=1024)

    reduce_feature = Model(inputs = input_data, outputs = code)
    encoded_input = Input(shape = (input_size, ))

    encoded_train = pd.DataFrame(reduce_feature.predict(train_data[features]))
    encoded_train = encoded_train.add_prefix('feature_')

    df = pd.concat([encoded_train,pca_data, train_data['target_kazutsugi']], axis=1)
    features_preprosessing = df.columns

    df = pd.DataFrame(save_memo(df,features_preprosessing))

    print("Saving train data with preprosessing")
    df.to_csv(f'../../../raw_data/round {n_round}/numerai_training_preprosessing_data.csv')

    del train_data
    gc.collect()

    tournament_data = pd.read_feather(f"../../../raw_data/round {n_round}/tournament-tmp")
    tournament_data_type = tournament_data.data_type
    tournament_id = tournament_data.id

    print("Tournament data pca preprosessing")
    pca_data = PCA_preprosessing(tournament_data, feature_groups, features, pca)

    print("Tournament data Autoencoder feature reduction")
    encoded_tournament = pd.DataFrame(reduce_feature.predict(tournament_data[features]))
    encoded_tournament = encoded_tournament.add_prefix('feature_')

    tournament = pd.concat([tournament_id, tournament_data_type, encoded_tournament,pca_data], axis=1)

    df = pd.DataFrame(save_memo(tournament,features_preprosessing))

    print("Saving tournament data with preprosessing")
    df.to_csv(f'../../../raw_data/round {n_round}/numerai_tournament_preprosessing_data.csv')

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