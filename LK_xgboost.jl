using XGBoost

const DATAPATH = joinpath(@__DIR__, "")
train_X = DMatrix(joinpath(DATAPATH, "train_x.txt"))
train_Y = DMatrix(joinpath(DATAPATH, "train_y.txt"))
test_X = DMatrix(joinpath(DATAPATH, "test_x.txt"))
num_round = 2
bst = xgboost(train_X, num_round, label = train_Y, eta = 1, max_depth = 2, booster = "gbtree", obj = "reg:linear")
preds = predict(bst, test_X)
print(preds[1])
