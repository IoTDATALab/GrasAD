set -e

cd ../..

echo "Run fedopt on synthetic."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/asyn_dp_synthetic_mapas.yaml \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.optimizer.lr 1. \
