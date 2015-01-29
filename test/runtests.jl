srand(1)
using ExperimentalSVM
iris = readcsv(joinpath(dirname(@__FILE__), "iris.csv"))
irisX = float(iris[:, 1:4]')
p, n = size(irisX)
train = bitrand(n)
irisY = [species == "setosa" ? 1.0 : -1.0 for species in  iris[:, 5]]
iris_model_pegasos_optim = svm(irisX[:,train], irisY[train])
@assert (predict(iris_model_pegasos_optim, irisX[:, ~train])) == irisY[~train]
iris_model_cddual_optim = cddual(irisX[:,train], irisY[train])
@assert (predict(iris_model_cddual_optim, irisX[:, ~train])) == irisY[~train]
iris_model_cddual_shrinking_optim = cddual_shrinking(irisX[:,train], irisY[train])
@assert (predict(iris_model_cddual_shrinking_optim, irisX[:, ~train])) == irisY[~train]
rcvY, rcvX = read_svm_data(joinpath(dirname(@__FILE__), "rcv1_train.binary"))
rcv_model_pegasos_optim = svm(rcvX, rcvY)
rcv_model_cddual_optim = cddual(rcvX, rcvY)
rcv_model_cddual_shrinking_optim = cddual_shrinking(rcvX, rcvY)
