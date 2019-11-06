from utils import *
from utils.utils import *
from utils.models import *


TOURNAMENT_NAME = "kazutsugi"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

def identity_block(X, hidden_layers=1):
    X = Dense(hidden_layers)(X)

    return X

def linear_block(X, hidden_layers=1):
 
    X_shortcut = X
    X = Dense(hidden_layers)(X_shortcut)
    X = PReLU()(X)
    X = BatchNormalization()(X)
    X = Dropout(0.1)(X)
        
    return X

def ResNet(input_shape = (310,), classes=1):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Stage 1    
    X_stage_1 = linear_block(X_input, hidden_layers=310)
    
    output_1 = Add()([X_stage_1,X_input])
    
    #Stage 2
    X_init = identity_block(output_1, 100)
    X_stage_2 = linear_block(output_1, hidden_layers=100)
    
    output_2 = Add()([X_stage_2,X_init])
    
    #Stage 3
    X_init = identity_block(output_2, 50)
    X_stage_3 = linear_block(output_2, hidden_layers=50)
    
    output_3 = Add()([X_stage_3,X_init])
    
    #Stage 4
    X_init = identity_block(output_3, 25)
    X_stage_4 = linear_block(output_3, hidden_layers=25)
    
    output_4 = Add()([X_stage_4,X_init])
    # output layer
    X = Dense(classes, activation='sigmoid')(output_4)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet')

    return model

def train_model(X_train, y_train, X_val, y_val):
    opt = RMSprop()
    model = ResNet()
    model.compile(optimizer=opt, loss='mse')
    er = EarlyStopping(patience=8, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    model.fit(X_train, y_train, epochs=10, callbacks=[er], validation_data=[X_val, y_val], batch_size=1024)
    return model

def make_predictions(PATH,n_round,with_preprosessing=False):
    pre = "whit" if with_preprosessing else "without"
    print(f"Working with round: {n_round}")
    train_data, tournament_data, features = load_proses_data(n_round, with_preprosessing)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    ids = tournament_data['id']

    X_train = train_data[features]
    y_train = train_data[TARGET_NAME]
    X_test = validation_data[features]
    y_test = validation_data[TARGET_NAME]

    X_tournament = tournament_data[features]

    folds = 4
    seed = 222
    kf = KFold(n_splits = folds, shuffle = True, random_state=seed)
    y_train_pred = np.zeros(X_train.shape[0])
    models = []

    y_pred_val = []
    y_pred_tournament = []

    for tr_idx, vl_idx in kf.split(X_train, y_train):
        x_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_vl, y_vl = X_train.iloc[vl_idx], y_train.iloc[vl_idx]
        
        model = train_model(x_tr, y_tr, x_vl, y_vl)
        y_hat = model.predict(x_vl)[:, 0]
        y_hat_val = model.predict(X_test)[:, 0]
        y_hat_tournament = model.predict(X_tournament)[:, 0]

        y_train_pred[vl_idx] += model.predict(x_vl)[:,0]
        print(f"Correlation train of fold: {correlation_score(y_vl, y_hat)}")
        print(f"Correlation valid of fold: {correlation_score(y_test, y_hat_val)}")

        y_pred_val.append(y_hat_val)
        y_pred_tournament.append(y_hat_tournament)
        models.append(model)

    correlation_train_score = correlation_score(y_train, y_train_pred)
    correlation_valid_score = correlation_score(y_test, np.average(y_pred_val, axis=0))
    print(f"Correlation score of train data: {correlation_train_score}")
    print(f"Correlation score of validation data: {correlation_valid_score}")

    me, con = check_correlation_consistency(model,train_data,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'consistency of train data ({pre}_preprosessing): {con}',margin=[0.02, 0.002], save=True)

    me, con = check_correlation_consistency(model,validation_data,correlation_score, features=features, target=TARGET_NAME)
    basic_plot(me,title=f'consistency of valid data ({pre}_preprosessing): {con}',margin=[0.02, 0.002], save=True)

    ##TOURNAMENT FILE 
    tournament_data[PREDICTION_NAME] = np.average(y_pred_tournament, axis=0)

    results_df = pd.DataFrame(data={PREDICTION_NAME:tournament_data[PREDICTION_NAME]})
    joined = pd.DataFrame(ids).join(results_df)
    print(f"# Writing predictions to {TOURNAMENT_NAME}_submissions.csv...")

    joined.to_csv(f"{PATH}/{TOURNAMENT_NAME}_resnet_{pre}_preprosessing_submission.csv", index=False)


@click.command()
@click.argument("n_round")
def main(n_round):
    init = time.time()
    PATH = f'../../submission/round {n_round}/autoencoder'
    os.makedirs(exist_ok=True, name=PATH)
    result = make_predictions(PATH,n_round)
    final = time.time()
    print(result+ f': {final-init}')


if __name__ == "__main__":
    main()