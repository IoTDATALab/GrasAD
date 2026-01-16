set -e

cd ../..

echo "Run fedopt on shakespeare."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/asyn_dp_shakespeare_mapas.yaml \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.optimizer.lr 1.
